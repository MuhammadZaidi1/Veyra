"""One volatile shared-memory frame slot, with nonblocking GUI reads.

The pixel/geometry payload uses POSIX shared memory rather than a pipe feeder or
temporary-file-backed multiprocessing array. Only scalar statistics use a
RawArray. A crashed writer can leave its lock held, but cannot block the viewer.
"""

from __future__ import annotations

from multiprocessing.shared_memory import SharedMemory

import numpy as np


class FrameMailbox:
    """A single replaceable RGB frame and its corresponding estimated geometry.

    Construct in the parent with a multiprocessing context, pass to a spawned
    worker, and close in both processes. Only the creating parent calls unlink.
    All calls to ``read_latest`` belong to one consumer process.
    """

    MAX_WIDTH = 1920
    MAX_HEIGHT = 1080
    MAX_POINTS = 478
    MAX_EDGES = 2048
    PIXEL_BYTES = MAX_WIDTH * MAX_HEIGHT * 3
    POINT_BYTES = MAX_POINTS * 3 * np.dtype(np.float32).itemsize
    EDGE_BYTES = MAX_EDGES * 2 * np.dtype(np.int32).itemsize

    def __init__(self, context):
        self._shm = SharedMemory(
            create=True, size=self.PIXEL_BYTES + self.POINT_BYTES + self.EDGE_BYTES
        )
        self._name = self._shm.name
        # sequence, width, height, point count, FPS, latency, capture time, edges
        self._metadata = context.RawArray("d", 8)
        self._lock = context.Lock()
        self._last_seq = 0
        self._owner = True
        self._closed = False
        self._unlinked = False

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_shm"] = None
        state["_owner"] = False
        state["_last_seq"] = 0
        return state

    def _memory(self):
        if self._closed:
            raise RuntimeError("Frame mailbox is closed")
        if self._shm is None:
            self._shm = SharedMemory(name=self._name)
        return self._shm.buf

    def publish(self, packet) -> bool:
        """Replace the slot; drop this update if a reader holds the lock."""
        rgb = np.asarray(packet["rgb"])
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Preview must be an H x W x 3 uint8 RGB array")
        height, width = rgb.shape[:2]
        if not (0 < width <= self.MAX_WIDTH and 0 < height <= self.MAX_HEIGHT):
            raise ValueError("Preview exceeds the 1920 x 1080 shared-memory slot")
        points = packet["points"]
        if points is None:
            points = np.empty((0, 3), dtype=np.float32)
        else:
            points = np.asarray(points, dtype=np.float32)
        if (points.ndim != 2 or points.shape[1] != 3 or len(points) > self.MAX_POINTS
                or not np.isfinite(points).all()):
            raise ValueError("Geometry must contain at most 478 finite XYZ points")
        edges = np.asarray(packet["edges"], dtype=np.int32)
        if edges.size == 0:
            edges = np.empty((0, 2), dtype=np.int32)
        if edges.ndim != 2 or edges.shape[1] != 2 or len(edges) > self.MAX_EDGES:
            raise ValueError("Mesh must contain at most 2048 landmark connections")
        max_index = len(points) if len(points) else self.MAX_POINTS
        if edges.size and (edges.min() < 0 or edges.max() >= max_index):
            raise ValueError("Mesh connection references a missing landmark")
        scalars = [float(packet[key]) for key in ("fps", "inference_ms", "captured_at")]
        if not np.isfinite(scalars).all() or min(scalars) < 0:
            raise ValueError("Frame statistics must be finite and nonnegative")
        if not self._lock.acquire(timeout=0.01):
            return False
        try:
            memory = self._memory()
            np.ndarray(rgb.shape, dtype=np.uint8, buffer=memory)[:] = rgb
            np.ndarray(points.shape, dtype=np.float32, buffer=memory,
                       offset=self.PIXEL_BYTES)[:] = points
            np.ndarray(edges.shape, dtype=np.int32, buffer=memory,
                       offset=self.PIXEL_BYTES + self.POINT_BYTES)[:] = edges
            self._metadata[1:] = [width, height, len(points), *scalars, len(edges)]
            self._metadata[0] += 1
            return True
        finally:
            self._lock.release()

    def read_latest(self):
        """Copy a new complete update, or return immediately if busy/unchanged."""
        if not self._lock.acquire(block=False):
            return None
        try:
            sequence = int(self._metadata[0])
            if sequence == self._last_seq:
                return None
            _, width, height, count, fps, latency, captured, edge_count = self._metadata
            width, height, count, edge_count = map(int, (width, height, count, edge_count))
            memory = self._memory()
            rgb = np.ndarray((height, width, 3), dtype=np.uint8, buffer=memory).copy()
            points = (np.ndarray((count, 3), dtype=np.float32, buffer=memory,
                                 offset=self.PIXEL_BYTES).copy() if count else None)
            edges = np.ndarray((edge_count, 2), dtype=np.int32, buffer=memory,
                               offset=self.PIXEL_BYTES + self.POINT_BYTES).copy()
            self._last_seq = sequence
            return {
                "kind": "frame", "rgb": rgb, "points": points, "edges": edges,
                "fps": fps, "inference_ms": latency, "captured_at": captured,
                "resolution": (width, height),
            }
        finally:
            self._lock.release()

    def close(self):
        """Release this process's mapping; does not remove the shared object."""
        if self._shm is not None:
            self._shm.close()
        self._closed = True

    def unlink(self):
        """Remove the shared object once, in the owning parent process."""
        if not self._owner:
            raise RuntimeError("Only the creating parent can unlink the frame mailbox")
        if not self._unlinked:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass
            self._unlinked = True

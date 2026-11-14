"""A spawned GUI lifecycle fixture: synthetic RAM data, never a webcam."""

import time

import numpy as np


def run_synthetic_worker(config, mailbox, stop, status):
    """Publish a tiny invented triangle over a black preview at 30 Hz."""
    del config
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    points = np.array(((-0.2, -0.15, 0.0), (0.2, -0.15, 0.0), (0.0, 0.2, 0.1)),
                      dtype=np.float32)
    edges = np.array(((0, 1), (1, 2), (2, 0)), dtype=np.int32)
    # Bound even an orphaned fixture if a lifecycle regression kills its parent.
    deadline = time.perf_counter() + 10
    try:
        status.send({"kind": "status", "message": "Synthetic lifecycle fixture; no camera"})
        while not stop.is_set() and time.perf_counter() < deadline:
            mailbox.publish({
                "kind": "frame", "rgb": rgb, "points": points, "edges": edges,
                "fps": 30.0, "inference_ms": 0.0, "captured_at": time.perf_counter(),
                "resolution": (640, 480),
            })
            stop.wait(1 / 30)
    finally:
        mailbox.close()
        status.close()

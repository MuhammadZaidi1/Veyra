"""Camera and CPU inference run in an isolated, disposable worker process."""

import sys
import time

import numpy as np

from .core import Config, FrameRate, VideoClock, relative_geometry
from .mesh import valid_face

CAMERA_HELP = (
    "Camera unavailable. In System Settings > Privacy & Security > Camera, "
    "allow the app hosting Python (Terminal or VS Code), then quit and reopen "
    "that app. Close other camera apps and try --camera 0 or --camera 1. "
    "If your device has no webcam, connect one first."
)


def make_landmarker(model):
    import mediapipe as mp

    return mp.tasks.vision.FaceLandmarker.create_from_options(
        mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )


def mesh_connections():
    from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections as C

    mesh = {(min(c.start, c.end), max(c.start, c.end)) for c in C.FACE_LANDMARKS_TESSELATION}
    contours = {(c.start, c.end) for c in C.FACE_LANDMARKS_CONTOURS}
    irises = {(c.start, c.end) for c in C.FACE_LANDMARKS_LEFT_IRIS + C.FACE_LANDMARKS_RIGHT_IRIS}
    return (
        np.array(sorted(mesh | irises), dtype=np.int32),
        np.array(sorted(contours | irises), dtype=np.int32),
    )


def annotate(frame, normalized, edges, contours, mirror):
    import cv2

    height, width = frame.shape[:2]
    if normalized is not None:
        pixels = np.rint(normalized[:, :2] * [width, height]).astype(np.int32)
        # Clipping keeps the overlay safe for landmarks outside the image.
        pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
        valid_edges = edges[(edges < len(pixels)).all(axis=1)]
        valid_contours = contours[(contours < len(pixels)).all(axis=1)]
        cv2.polylines(frame, pixels[valid_edges], False, (95, 135, 95), 1, cv2.LINE_AA)
        cv2.polylines(frame, pixels[valid_contours], False, (70, 225, 190), 1, cv2.LINE_AA)
        for x, y in pixels:
            cv2.circle(frame, (int(x), int(y)), 1, (210, 255, 235), -1, cv2.LINE_AA)
    if mirror:
        frame = cv2.flip(frame, 1)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def extract_landmarks(result):
    """Treat partial, missing or nonfinite detections as tracking loss."""
    try:
        if not result.face_landmarks:
            return None
        points = np.array([(p.x, p.y, p.z) for p in result.face_landmarks[0]], dtype=np.float32)
        return points if valid_face(points) else None
    except (AttributeError, TypeError, ValueError):
        return None


def run_worker(config: Config, mailbox, stop, status):
    capture = None
    landmarker = None
    try:
        import cv2
        import mediapipe as mp

        cv2.setNumThreads(1)
        status.send({"kind": "status", "message": "Loading local CPU face model…"})
        landmarker = make_landmarker(config.model)
        edges, contours = mesh_connections()
        status.send({"kind": "status", "message": "Opening camera — allow macOS camera access if prompted…"})
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        capture = cv2.VideoCapture(config.camera, backend)
        if not capture.isOpened():
            raise RuntimeError(CAMERA_HELP)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        capture.set(cv2.CAP_PROP_FPS, config.fps)
        clock, rate = VideoClock(), FrameRate()
        start = time.perf_counter()
        misses = 0
        while not stop.is_set():
            frame_start = time.perf_counter()
            ok, frame = capture.read()
            if not ok or frame is None:
                misses += 1
                if misses >= 10:
                    raise RuntimeError("Camera stopped returning frames. " + CAMERA_HELP)
                status.send({"kind": "status", "message": "Camera frame unavailable; retrying…"})
                stop.wait(0.1)
                continue
            misses = 0
            height, width = frame.shape[:2]
            # Drivers may ignore the requested size. Keep inference/IPC bounded.
            fit = min(1.0, 1920 / width, 1080 / height)
            if fit < 1:
                frame = cv2.resize(frame, (max(1, int(width * fit)), max(1, int(height * fit))),
                                   interpolation=cv2.INTER_AREA)
                height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            inference_start = time.perf_counter()
            result = landmarker.detect_for_video(image, clock.tick(inference_start - start))
            latency = (time.perf_counter() - inference_start) * 1000
            normalized = extract_landmarks(result)
            geometry = None
            if normalized is not None:
                geometry = relative_geometry(normalized, width, height)
            frame_edges = (edges[(edges < len(geometry)).all(axis=1)]
                           if geometry is not None and len(geometry) < 478 else edges)
            preview = annotate(frame, normalized, edges, contours, config.mirror)
            done = time.perf_counter()
            mailbox.publish({
                "kind": "frame", "rgb": preview, "points": geometry,
                "edges": frame_edges, "fps": rate.update(done),
                "inference_ms": latency, "captured_at": frame_start,
                "resolution": (width, height),
            })
            stop.wait(max(0, 1 / config.fps - (time.perf_counter() - frame_start)))
    except Exception as exc:
        status.send({"kind": "error", "message": str(exc)})
    finally:
        if capture is not None:
            capture.release()
        if landmarker is not None:
            landmarker.close()
        mailbox.close()
        status.close()


def self_test(model):
    """Exercise the real CPU model on synthetic black images; no camera access."""
    import mediapipe as mp

    edges, _ = mesh_connections()
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((480, 640, 3), dtype=np.uint8))
    with make_landmarker(model) as landmarker:
        timings = []
        for timestamp in (0, 33, 66):
            start = time.perf_counter()
            result = landmarker.detect_for_video(image, timestamp)
            timings.append((time.perf_counter() - start) * 1000)
            if result.face_landmarks:
                raise RuntimeError("Unexpected face on synthetic blank input")
    print(f"CPU model self-test passed: 3 blank frames, no face, {len(edges)} mesh edges; "
          f"inference {min(timings):.1f}–{max(timings):.1f} ms. No camera opened.")

"""Numerical helpers; no camera, GUI, network, or persistence."""

from collections import deque
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "face_landmarker.task"
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"


def validate_model(model: Path) -> None:
    if not model.is_file():
        raise ValueError(f"Model missing: {model}\nRun .venv/bin/python scripts/download_model.py")
    if hashlib.sha256(model.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError("Model checksum mismatch. Remove the damaged model and run "
                         ".venv/bin/python scripts/download_model.py again.")


@dataclass(frozen=True)
class Config:
    model: Path = DEFAULT_MODEL
    camera: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    mirror: bool = True


def relative_geometry(landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    """Center x/y/z in image-width units, with up +y and toward viewer +z.

    MediaPipe z is estimated relative depth on x's scale. There is no camera
    calibration, metric reconstruction, rigid pose correction, or depth sensor.
    """
    points = np.asarray(landmarks, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Landmarks must be a finite N x 3 array")
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    if not len(points):
        return points.copy()
    result = points * np.array([1.0, -height / width, -1.0], dtype=np.float32)
    return result - result.mean(axis=0)


class VideoClock:
    """Millisecond timestamps that remain strictly increasing at any frame rate."""

    def __init__(self):
        self.last_ms = -1

    def tick(self, elapsed_seconds: float) -> int:
        self.last_ms = max(self.last_ms + 1, int(elapsed_seconds * 1000))
        return self.last_ms


class FrameRate:
    """Processed capture + inference + overlay throughput over recent frames."""

    def __init__(self, window: int = 60):
        self.times = deque(maxlen=window)

    def update(self, now: float) -> float:
        self.times.append(now)
        span = self.times[-1] - self.times[0]
        return (len(self.times) - 1) / span if span > 0 else 0.0

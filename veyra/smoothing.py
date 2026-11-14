"""Bounded RAM-only adaptive smoothing; no averaging across facial vertices."""

import math
import time

import numpy as np

from .mesh import valid_face


def alpha(cutoff, dt):
    return 1.0 / (1.0 + 1.0 / (2.0 * math.pi * cutoff * dt))


class OneEuroFilter:
    """Velocity-adaptive low-pass filter in image-width relative units.

    One scalar cutoff per vertex preserves the direction of that vertex's
    movement. Fast movement raises the cutoff to reduce expression/head lag.
    """

    def __init__(self, min_cutoff=4.0, beta=20.0, derivative_cutoff=1.0):
        self.configure(min_cutoff, beta, derivative_cutoff)

    def configure(self, min_cutoff, beta=20.0, derivative_cutoff=1.0):
        values = (min_cutoff, beta, derivative_cutoff)
        if not all(math.isfinite(v) for v in values) or min_cutoff <= 0 or beta < 0 or derivative_cutoff <= 0:
            raise ValueError("Smoothing cutoffs must be positive and beta nonnegative")
        self.min_cutoff, self.beta, self.derivative_cutoff = values
        self.reset()

    def reset(self):
        self.previous = self.filtered = self.derivative = None
        self.timestamp = None

    def update(self, points, timestamp):
        if points is None:
            self.reset()
            return None
        points = np.asarray(points, dtype=np.float32)
        if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
                or not np.isfinite(points).all() or not math.isfinite(timestamp)):
            self.reset()
            raise ValueError("Smoothing requires finite XYZ points and a finite timestamp")
        dt = timestamp - self.timestamp if self.timestamp is not None else 0
        if self.previous is None or points.shape != self.previous.shape or not 0 < dt <= 0.25:
            self.previous = points.copy()
            self.filtered = points.copy()
            self.derivative = np.zeros_like(points)
        else:
            velocity = (points - self.previous) / dt
            a = alpha(self.derivative_cutoff, dt)
            self.derivative += a * (velocity - self.derivative)
            cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.derivative, axis=1, keepdims=True)
            self.filtered += alpha(cutoff, dt) * (points - self.filtered)
            np.copyto(self.previous, points)
        self.timestamp = timestamp
        return self.filtered.copy()


class GeometryStream:
    """Filter and compare the same incoming frames, resetting on tracking loss."""

    def __init__(self, enabled=True, cutoff=4.0, beta=20.0):
        self.filter = OneEuroFilter(cutoff, beta)
        self.enabled = enabled
        self.reset()

    def reset(self):
        self.filter.reset()
        self.previous_raw = self.previous_output = None
        self.raw_motion = self.output_motion = 0.0
        self.filter_ms = 0.0
        self.comparisons = 0

    def configure(self, enabled, cutoff):
        self.enabled = enabled
        self.filter.configure(cutoff, self.filter.beta)
        self.reset()

    def update(self, points, timestamp):
        if not valid_face(points) or not math.isfinite(timestamp):
            self.reset()
            return None
        raw = np.asarray(points, dtype=np.float32)
        if self.previous_raw is not None and self.previous_raw.shape != raw.shape:
            self.reset()
        # Reset motion statistics too when acquisition resumes after a gap.
        last_time = self.filter.timestamp
        if last_time is not None and not 0 < timestamp - last_time <= 0.25:
            self.reset()
        start = time.perf_counter()
        output = self.filter.update(raw, timestamp) if self.enabled else raw
        if not self.enabled:
            self.filter.timestamp = timestamp
        self.filter_ms = (time.perf_counter() - start) * 1000
        if self.previous_raw is not None and self.previous_raw.shape == raw.shape:
            raw_motion = float(np.mean(np.sum((raw - self.previous_raw) ** 2, axis=1)))
            output_motion = float(np.mean(np.sum((output - self.previous_output) ** 2, axis=1)))
            # Rolling exponential mean of displacement energy, never a scan log.
            weight = 1.0 if not self.comparisons else 0.1
            self.raw_motion += weight * (raw_motion - self.raw_motion)
            self.output_motion += weight * (output_motion - self.output_motion)
            self.comparisons += 1
        self.previous_raw = raw.copy()
        self.previous_output = output.copy()
        return output

    def comparison_text(self):
        if not self.comparisons:
            return "Motion comparison: waiting for consecutive tracked frames"
        return (f"Motion RMS raw → shown: {math.sqrt(self.raw_motion):.5f} → "
                f"{math.sqrt(self.output_motion):.5f} rel/frame · "
                f"filter {self.filter_ms:.2f} ms" + ("" if self.enabled else " · smoothing off"))

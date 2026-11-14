"""Shared orbit controls for the accelerated surface and legacy wireframe."""

import math
import numpy as np
from PySide6 import QtCore, QtGui


def rotate_points(points: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    """Rotate XYZ points around the display's vertical and horizontal axes.

    Angles are degrees. Input coordinates use X right, Y up, and Z toward the
    viewer. This affects inspection only; it does not estimate head pose.
    """
    yaw_radians, pitch_radians = math.radians(yaw), math.radians(pitch)
    cy, sy = math.cos(yaw_radians), math.sin(yaw_radians)
    cp, sp = math.cos(pitch_radians), math.sin(pitch_radians)
    yaw_matrix = np.array(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)))
    pitch_matrix = np.array(((1.0, 0.0, 0.0), (0.0, cp, -sp), (0.0, sp, cp)))
    return np.asarray(points) @ yaw_matrix.T @ pitch_matrix.T


class OrbitControls:
    def reset_view(self) -> None:
        self.yaw = 0.0
        self.pitch = 0.0
        self.zoom = 1.0
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag_position = event.position()
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drag_position is not None:
            delta = event.position() - self._drag_position
            self._drag_position = event.position()
            self.yaw = (self.yaw + delta.x() * 0.5) % 360.0
            self.pitch = max(-89.0, min(89.0, self.pitch + delta.y() * 0.5))
            self.update()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag_position = None
            self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        delta = event.angleDelta().y() or event.pixelDelta().y()
        factor = math.exp(max(-5.0, min(5.0, delta / 120.0 * 0.15)))
        self.zoom = max(0.2, min(5.0, self.zoom * factor))
        self.update()
        event.accept()


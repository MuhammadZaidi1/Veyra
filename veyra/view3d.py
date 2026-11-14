"""Local, software-rendered inspection of estimated relative face geometry.

Qt paints the projected wireframe directly. No OpenGL context, network service,
or persistent copy of the landmarks is required.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


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


class _FaceCanvas(QtWidgets.QWidget):
    """Project a live mesh with a persistent, user-controlled inspection view."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 320)
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.points: np.ndarray | None = None
        self.edges = np.empty((0, 2), dtype=np.intp)
        self.reference_span: float | None = None
        self.yaw = 0.0
        self.pitch = 0.0
        self.zoom = 1.0
        self._drag_position: QtCore.QPointF | None = None

    def update_face(self, points: np.ndarray | None, edges: np.ndarray) -> None:
        """Replace the current in-memory geometry, clearing it for no face."""
        if points is None:
            self.points = None
            self.edges = np.empty((0, 2), dtype=np.intp)
            self.update()
            return

        coordinates = np.asarray(points, dtype=np.float64)
        connections = np.asarray(edges, dtype=np.intp)
        if (
            coordinates.ndim != 2
            or coordinates.shape[1] != 3
            or len(coordinates) == 0
            or not np.isfinite(coordinates).all()
        ):
            raise ValueError("3D face points must be a nonempty, finite N by 3 array.")
        if connections.size == 0:
            connections = np.empty((0, 2), dtype=np.intp)
        if connections.ndim != 2 or connections.shape[1] != 2:
            raise ValueError("Mesh connections must be an N by 2 array.")
        if connections.size and (connections.min() < 0 or connections.max() >= len(coordinates)):
            raise ValueError("A mesh connection references a missing landmark.")

        self.points = coordinates.copy()
        self.edges = connections.copy()
        if self.reference_span is None:
            # Fix the initial fit so later expression/head movements do not
            # continuously change the user's zoom or inspection orientation.
            radius = float(np.linalg.norm(coordinates, axis=1).max())
            self.reference_span = max(2.0 * radius, 1e-3)
        self.update()

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

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor("#101923"))

        if self.points is None:
            painter.setPen(QtGui.QColor("#c3ceda"))
            painter.drawText(
                self.rect(),
                QtCore.Qt.AlignmentFlag.AlignCenter,
                "No face detected\nLook toward the camera to show the live mesh.",
            )
            return

        rotated = rotate_points(self.points, self.yaw, self.pitch)
        scale = min(self.width(), self.height()) * 0.78 * self.zoom / self.reference_span
        projected = rotated[:, :2] * np.array((scale, -scale))
        projected += np.array((self.width() / 2.0, self.height() / 2.0))

        depths = rotated[:, 2]
        depth_span = max(float(np.ptp(depths)), 1e-9)
        depth_bins = np.clip(((depths - depths.min()) / depth_span * 11).astype(int), 0, 11)
        if len(self.edges):
            edge_bins = ((depth_bins[self.edges[:, 0]] + depth_bins[self.edges[:, 1]]) // 2)
            for depth_bin in range(12):
                path = QtGui.QPainterPath()
                for start, end in self.edges[edge_bins == depth_bin]:
                    path.moveTo(float(projected[start, 0]), float(projected[start, 1]))
                    path.lineTo(float(projected[end, 0]), float(projected[end, 1]))
                painter.setPen(QtGui.QPen(QtGui.QColor(57, 194, 194, 55 + depth_bin * 14), 0.9))
                painter.drawPath(path)

        for depth_bin in range(12):
            visible_points = projected[depth_bins == depth_bin]
            polygon = QtGui.QPolygonF(
                [QtCore.QPointF(float(x), float(y)) for x, y in visible_points]
            )
            pen = QtGui.QPen(QtGui.QColor(164, 239, 224, 90 + depth_bin * 15), 2.2)
            pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawPoints(polygon)

        painter.setPen(QtGui.QColor("#8c9eae"))
        painter.drawText(14, self.height() - 14, "Orthographic view · relative coordinates")


class Face3DWindow(QtWidgets.QWidget):
    """Independent window for rotating the currently estimated face mesh.

    Call methods on the Qt main thread. Closing this window emits ``closed`` so
    the application can release its camera and close the companion preview.
    """

    closed = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__(None)
        self.setWindowTitle("Veyra — Estimated relative 3D face")
        self.resize(700, 740)
        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("Estimated relative 3D landmarks")
        title_font = title.font()
        title_font.setPointSize(title_font.pointSize() + 4)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        layout.addWidget(QtWidgets.QLabel("Not measured depth • no physical units"))
        self.canvas = _FaceCanvas(self)
        layout.addWidget(self.canvas, 1)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Drag to rotate · scroll to zoom · R to reset"), 1)
        reset_button = QtWidgets.QPushButton("Reset view")
        reset_button.clicked.connect(self.canvas.reset_view)
        controls.addWidget(reset_button)
        layout.addLayout(controls)
        self.status_label = QtWidgets.QLabel("Waiting for a face…")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self._shortcuts = []
        for key, callback in (
            ("R", self.canvas.reset_view),
            ("Q", self.close),
            ("Esc", self.close),
        ):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)

    def update_face(self, points: np.ndarray | None, edges: np.ndarray) -> None:
        self.canvas.update_face(points, edges)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.canvas.update_face(None, np.empty((0, 2), dtype=np.intp))
        self.closed.emit()
        super().closeEvent(event)

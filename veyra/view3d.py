"""Minimal live surface viewer; camera and landmark data never leave RAM."""

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .render_common import rotate_points  # Preserve the v0.1 public helper.
from .render_software import WireCanvas


class Face3DWindow(QtWidgets.QWidget):
    closed = QtCore.Signal()
    smoothing_changed = QtCore.Signal(bool, float)

    def __init__(self, renderer="auto", smoothing=True, cutoff=4.0, mode="Solid"):
        super().__init__(None)
        self.setWindowTitle("Veyra — Estimated 3D facial surface")
        self.resize(720, 800)
        self._layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("Estimated relative 3D facial surface")
        title.setStyleSheet("font-size: 19px; font-weight: bold;")
        self._layout.addWidget(title)
        self._layout.addWidget(QtWidgets.QLabel("Not measured depth · no physical units"))
        platform = QtGui.QGuiApplication.platformName().lower()
        self.accelerated = renderer != "software" and platform not in ("offscreen", "minimal")
        if self.accelerated:
            from .render_gl import SurfaceCanvas
            self.canvas = SurfaceCanvas(self)
            self.canvas.ready.connect(self._renderer_ready)
            self.canvas.failed.connect(self._renderer_failed)
        else:
            self.canvas = WireCanvas(self)
        self._layout.addWidget(self.canvas, 1)
        controls = QtWidgets.QHBoxLayout()
        self.display_mode = QtWidgets.QComboBox()
        self.display_mode.addItems(("Wireframe", "Solid", "Solid + Wireframe"))
        self.display_mode.setCurrentText(mode if self.accelerated else "Wireframe")
        self.display_mode.currentTextChanged.connect(self._set_mode)
        controls.addWidget(self.display_mode)
        self.smoothing = QtWidgets.QCheckBox("Smoothing")
        self.smoothing.setToolTip("Additional viewer smoothing. The preview and MediaPipe's built-in single-face video smoothing remain unchanged.")
        self.smoothing.setChecked(smoothing)
        controls.addWidget(self.smoothing)
        controls.addWidget(QtWidgets.QLabel("Response"))
        self.cutoff = QtWidgets.QDoubleSpinBox()
        self.cutoff.setRange(1.0, 10.0)
        self.cutoff.setSingleStep(0.5)
        self.cutoff.setDecimals(1)
        self.cutoff.setSuffix(" Hz")
        self.cutoff.setValue(cutoff)
        self.cutoff.setToolTip("Higher response reduces lag; lower response smooths more. Fast motion adapts automatically.")
        controls.addWidget(self.cutoff)
        controls.addStretch()
        reset = QtWidgets.QPushButton("Reset view")
        reset.clicked.connect(lambda: self.canvas.reset_view())
        controls.addWidget(reset)
        self._layout.addLayout(controls)
        lighting = QtWidgets.QHBoxLayout()
        lighting.addWidget(QtWidgets.QLabel("Light direction"))
        self.light_angle = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.light_angle.setRange(-85, 85)
        self.light_angle.setValue(-35)
        lighting.addWidget(self.light_angle, 1)
        lighting.addWidget(QtWidgets.QLabel("Intensity"))
        self.light_strength = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.light_strength.setRange(0, 150)
        self.light_strength.setValue(100)
        lighting.addWidget(self.light_strength, 1)
        self._layout.addLayout(lighting)
        self.renderer_label = QtWidgets.QLabel("OpenGL initializing…" if self.accelerated else "Software wireframe fallback")
        self.renderer_label.setWordWrap(True)
        self._layout.addWidget(self.renderer_label)
        self.render_metrics = QtWidgets.QLabel("Render CPU: —    GPU: —    Normals: —")
        self.render_metrics.setToolTip("CPU submission excludes GPU completion. GPU time is queried asynchronously when available.")
        self._layout.addWidget(self.render_metrics)
        self.comparison_label = QtWidgets.QLabel("Motion comparison: waiting for tracked frames")
        self.comparison_label.setWordWrap(True)
        self.comparison_label.setToolTip("RMS frame-to-frame vertex displacement on the same input frames. Includes intentional movement; this is not physical accuracy or a pure jitter measurement.")
        self._layout.addWidget(self.comparison_label)
        self.status_label = QtWidgets.QLabel("Waiting for a face…")
        self.status_label.setWordWrap(True)
        self._layout.addWidget(self.status_label)
        footer = QtWidgets.QLabel("Drag to rotate · scroll to zoom · R to reset\nOpen face mask only; rear view shows the same estimated sheet.")
        footer.setWordWrap(True)
        self._layout.addWidget(footer)
        self.smoothing.toggled.connect(self._smoothing_control)
        self.cutoff.valueChanged.connect(self._smoothing_control)
        self.light_angle.valueChanged.connect(self._set_light)
        self.light_strength.valueChanged.connect(self._set_light)
        self._shortcuts = []
        for key, callback in (("R", lambda: self.canvas.reset_view()), ("Q", self.close), ("Esc", self.close)):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)
        self._set_mode(self.display_mode.currentText())
        self._set_light()
        self._apply_capabilities()
        self._metrics_timer = QtCore.QTimer(self)
        self._metrics_timer.timeout.connect(self.refresh_render_metrics)
        self._metrics_timer.start(250)
        if self.accelerated:
            QtCore.QTimer.singleShot(1500, self._check_context)

    def _smoothing_control(self, *_):
        self.smoothing_changed.emit(self.smoothing.isChecked(), self.cutoff.value())

    def _set_mode(self, mode):
        if self.accelerated:
            self.canvas.set_mode(mode)

    def _set_light(self, *_):
        if self.accelerated:
            self.canvas.set_light(self.light_angle.value(), self.light_strength.value() / 100)

    def _apply_capabilities(self):
        for widget in (self.display_mode, self.light_angle, self.light_strength):
            widget.setEnabled(self.accelerated)

    def _renderer_ready(self, info):
        self.renderer_label.setText("OpenGL surface · " + info)

    def _renderer_failed(self, error):
        # Replace a failed GL widget only after its paint/initialization callback returns.
        QtCore.QTimer.singleShot(0, lambda: self._fallback(error))

    def _check_context(self):
        if self.accelerated and not self.canvas.isValid():
            self._fallback("A compatible OpenGL context is unavailable")

    def _fallback(self, error):
        if not self.accelerated:
            return
        old = self.canvas
        points, edges = old.points, old.edges
        self.accelerated = False
        self.canvas = WireCanvas(self)
        self.canvas.yaw, self.canvas.pitch, self.canvas.zoom = old.yaw, old.pitch, old.zoom
        self._layout.replaceWidget(old, self.canvas)
        if points is not None:
            self.canvas.update_face(points, edges)
        old.cleanup()
        old.hide()
        old.deleteLater()
        self.display_mode.setCurrentText("Wireframe")
        self.renderer_label.setText("Surface unavailable; software wireframe fallback. " + error)
        self._apply_capabilities()

    def refresh_render_metrics(self):
        if not self.accelerated or self.canvas.points is None:
            self.render_metrics.setText("Render CPU: —    GPU: —    Normals: —")
            return
        gpu = self.canvas.gpu_draw_ms
        gpu_text = f"{gpu:.2f} ms" if gpu is not None else "unavailable"
        self.render_metrics.setText(f"Render {self.canvas.render_fps:.1f} FPS · CPU: {self.canvas.cpu_draw_ms:.2f} ms · GPU: {gpu_text}"
                                    f"    Normals: {self.canvas.normal_ms:.2f} ms")

    def update_face(self, points, edges):
        self.canvas.update_face(points, edges)

    def set_status(self, text):
        self.status_label.setText(text)

    def closeEvent(self, event):
        self._metrics_timer.stop()
        self.canvas.update_face(None, np.empty((0, 2), dtype=np.intp))
        if self.accelerated:
            self.canvas.cleanup()
        self.closed.emit()
        super().closeEvent(event)

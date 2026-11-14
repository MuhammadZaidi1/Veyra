"""Qt main-thread presentation; all webcam data stays in volatile memory."""

import multiprocessing as multiprocessing
import signal
import sys
import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .core import Config
from .tracking import run_worker
from .transport import FrameMailbox
from .view3d import Face3DWindow


class PreviewWindow(QtWidgets.QWidget):
    closed = QtCore.Signal()

    def __init__(self, mirror: bool):
        super().__init__()
        self.setWindowTitle("Veyra — Live camera and landmarks")
        self.resize(800, 700)
        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("Veyra v0.1 · local face geometry research")
        title.setStyleSheet("font-size: 19px; font-weight: bold;")
        layout.addWidget(title)
        layout.addWidget(QtWidgets.QLabel(
            "CPU inference · single face · preview " + ("mirrored" if mirror else "unmirrored")))
        self.preview = QtWidgets.QLabel("Waiting for camera…")
        self.preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(320, 240)
        self.preview.setStyleSheet("background: #101923; color: #dce7ee;")
        layout.addWidget(self.preview, 1)
        self.metrics = QtWidgets.QLabel("FPS: —    Inference: — ms    Frame age: — ms")
        self.metrics.setStyleSheet("font-family: monospace; font-size: 14px;")
        layout.addWidget(self.metrics)
        self.status = QtWidgets.QLabel("Starting…")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addWidget(QtWidgets.QLabel("Frames and geometry stay in memory. Q / Esc / close either window to quit."))
        quit_button = QtWidgets.QPushButton("Stop camera and quit")
        quit_button.clicked.connect(self.close)
        layout.addWidget(quit_button)
        self._image = None
        self._shortcuts = []
        for key in ("Q", "Esc"):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.activated.connect(self.close)
            self._shortcuts.append(shortcut)

    def update_image(self, rgb):
        height, width = rgb.shape[:2]
        # Copy detaches Qt from the incoming NumPy frame's lifetime.
        self._image = QtGui.QImage(rgb.data, width, height, rgb.strides[0],
                                   QtGui.QImage.Format.Format_RGB888).copy()
        self._redraw()

    def clear_image(self, text):
        self._image = None
        self.preview.clear()
        self.preview.setText(text)

    def _redraw(self):
        if self._image is not None:
            pixmap = QtGui.QPixmap.fromImage(self._image)
            self.preview.setPixmap(pixmap.scaled(
                self.preview.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._redraw()

    def closeEvent(self, event):
        self.clear_image("Camera stopped")
        self.closed.emit()
        super().closeEvent(event)


class Controller(QtCore.QObject):
    def __init__(self, app, config: Config, duration: float = 0):
        super().__init__()
        self.app = app
        self.preview = PreviewWindow(config.mirror)
        self.geometry = Face3DWindow()
        self.preview.closed.connect(self.request_quit)
        self.geometry.closed.connect(self.request_quit)
        context = multiprocessing.get_context("spawn")
        self.mailbox = FrameMailbox(context)
        self.stop = context.Event()
        self.status, sender = context.Pipe(duplex=False)
        self.worker = context.Process(target=run_worker,
                                      args=(config, self.mailbox, self.stop, sender),
                                      name="VeyraCamera", daemon=True)
        try:
            self.worker.start()
        except BaseException:
            sender.close()
            self.status.close()
            self.mailbox.close()
            self.mailbox.unlink()
            raise
        sender.close()
        self.error = False
        self.stopped = False
        self.last_frame = None
        self.frame_count = 0
        self.last_metrics = None
        self.status_open = True
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(16)
        app.aboutToQuit.connect(self.shutdown)
        if duration:
            QtCore.QTimer.singleShot(int(duration * 1000), self.request_quit)
        self.preview.show()
        self.geometry.show()

    def set_status(self, message):
        self.preview.status.setText(message)
        self.geometry.set_status(message)

    def show_error(self, message):
        if self.error:
            return
        self.error = True
        self.stop.set()
        self.preview.clear_image("Camera stopped")
        self.preview.metrics.setText("FPS: —    Inference: — ms    Frame age: — ms")
        self.geometry.update_face(None, np.empty((0, 2), dtype=int))
        self.set_status(message)
        print("Veyra: " + message, file=sys.stderr, flush=True)

    def poll(self):
        while self.status_open and self.status.poll():
            try:
                packet = self.status.recv()
            except EOFError:
                self.status_open = False
                break
            if packet["kind"] == "error":
                self.show_error(packet["message"])
            elif not self.error:
                self.set_status(packet["message"])
        latest = self.mailbox.read_latest()
        if latest is not None and not self.error:
            self.frame_count += 1
            self.last_frame = time.perf_counter()
            self.preview.update_image(latest["rgb"])
            self.geometry.update_face(latest["points"], latest["edges"])
            age = max(0, (time.perf_counter() - latest["captured_at"]) * 1000)
            metrics = (f"FPS: {latest['fps']:.1f}    Inference: {latest['inference_ms']:.1f} ms"
                       f"    Frame age: {age:.0f} ms")
            self.preview.metrics.setText(metrics)
            self.last_metrics = metrics
            width, height = latest["resolution"]
            if self.frame_count == 1:
                print(f"Camera active: {width} × {height}; CPU inference; frames stay in RAM.", flush=True)
            face = (f"Tracking {len(latest['points'])} estimated landmarks"
                    if latest["points"] is not None else "No face detected — look toward the camera")
            self.preview.status.setText(f"{face} · {width} × {height} · CPU")
            self.geometry.set_status(f"{face} · {metrics}")
        if not self.worker.is_alive() and not self.error:
            self.show_error(f"Camera worker exited (code {self.worker.exitcode}). "
                            "Check terminal output, camera access, and the local model.")
        elif self.last_frame is not None and time.perf_counter() - self.last_frame > 2 and not self.error:
            self.preview.clear_image("Waiting for fresh camera frames…")
            self.geometry.update_face(None, np.empty((0, 2), dtype=int))
            self.preview.metrics.setText("FPS: 0.0    Inference: — ms    Frame age: stale")
            self.set_status("Camera frames stalled; close the window to stop and restart.")

    def request_quit(self):
        if self.stopped:
            return
        self.shutdown()
        self.app.quit()

    def shutdown(self):
        if self.stopped:
            return
        self.stopped = True
        self.timer.stop()
        self.stop.set()
        self.worker.join(timeout=2)
        if self.worker.is_alive():
            self.worker.terminate()
            self.worker.join(timeout=2)
        if self.worker.is_alive():
            self.worker.kill()
            self.worker.join(timeout=1)
        self.preview.clear_image("Camera stopped")
        self.geometry.update_face(None, np.empty((0, 2), dtype=int))
        self.status.close()
        self.mailbox.close()
        self.mailbox.unlink()
        self.worker.close()
        self.preview.close()
        self.geometry.close()
        if self.last_metrics:
            print(f"Displayed {self.frame_count} previews. {self.last_metrics}", flush=True)
        print("Veyra stopped; camera worker released.", flush=True)


def run(config: Config, duration: float) -> int:
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("Veyra")
    controller = Controller(app, config, duration)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous:
        signal.signal(sig, lambda *_: controller.request_quit())
    try:
        app.exec()
    finally:
        controller.shutdown()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 1 if controller.error else 0

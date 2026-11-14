"""Offline checks on public model reference data and invented motion only.

No camera access, captured data, geometry export, or benchmark data files.
"""

import math
import time
import zipfile

import numpy as np

from .mesh import SURFACE_EDGES, TRIANGLES
from .smoothing import OneEuroFilter


def reference_geometry(model):
    """Read the public model's generic reference mesh in memory, not a scan."""
    from mediapipe.tasks.cc.vision.face_geometry.proto import geometry_pipeline_metadata_pb2
    metadata = geometry_pipeline_metadata_pb2.GeometryPipelineMetadata()
    with zipfile.ZipFile(model) as archive:
        metadata.ParseFromString(archive.read('geometry_pipeline_metadata_landmarks.binarypb'))
    points = np.asarray(metadata.canonical_mesh.vertex_buffer, dtype=np.float32).reshape(-1, 5)[:, :3].copy()
    if points.shape != (468, 3) or not np.isfinite(points).all():
        raise ValueError('Unexpected public reference mesh layout')
    points -= points.mean(axis=0)
    points *= 0.5 / np.ptp(points[:, 0])
    return points


def smoothing_benchmark():
    """Repeatable synthetic noise reduction and step response at 30 Hz."""
    rng = np.random.default_rng(42)
    base = np.zeros((468, 3), dtype=np.float32)
    filter_ = OneEuroFilter()
    raw_energy, filtered_energy, timings = [], [], []
    for index in range(360):
        raw = base + rng.normal(0, 0.001, base.shape).astype(np.float32)
        start = time.perf_counter()
        result = filter_.update(raw, index / 30)
        timings.append((time.perf_counter() - start) * 1000)
        if index >= 30:
            raw_energy.append(float(np.mean(raw ** 2)))
            filtered_energy.append(float(np.mean(result ** 2)))
    raw_rms, filtered_rms = math.sqrt(np.mean(raw_energy)), math.sqrt(np.mean(filtered_energy))
    filter_.reset()
    for index in range(30):
        filter_.update(base, index / 30)
    reached = None
    for index in range(30, 60):
        result = filter_.update(base + 0.05, index / 30)
        if float(result.mean()) >= 0.045 and reached is None:
            reached = (index - 30 + 1) * 1000 / 30
    return {'raw_rms': raw_rms, 'filtered_rms': filtered_rms,
            'reduction_percent': 100 * (1 - filtered_rms / raw_rms),
            'filter_p50_ms': float(np.median(timings)), 'step_90_ms': reached}


def benchmark(model):
    from PySide6 import QtCore, QtGui, QtWidgets
    from .render_gl import SurfaceCanvas, configure_surface_format
    from .render_software import WireCanvas

    points = reference_geometry(model)
    configure_surface_format()
    app = QtWidgets.QApplication([])
    window = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(window)
    wire, surface = WireCanvas(), SurfaceCanvas()
    for canvas in (wire, surface):
        canvas.setFixedSize(600, 600)
        layout.addWidget(canvas)
        canvas.update_face(points, SURFACE_EDGES)
    window.show()
    outcome = [1]

    def measure():
        try:
            if not surface.isValid() or surface.render_error:
                raise RuntimeError(surface.render_error or 'no OpenGL context')
            print('Reference: public model template, 468 vertices,', len(TRIANGLES), 'triangles; no camera.')
            print('Renderer:', surface.renderer_info)
            dpr = surface.devicePixelRatioF()
            pixmap = QtGui.QPixmap(round(600*dpr), round(600*dpr))
            pixmap.setDevicePixelRatio(dpr)
            print('Completed offscreen paint, 600x600 logical pixels, DPR', dpr,
                  '(includes geometry update; excludes window compositing).')
            for name, canvas in (('v0.1 software wireframe', wire), ('v0.2 smooth solid', surface)):
                samples = []
                for frame in range(90):
                    start = time.perf_counter()
                    canvas.yaw = math.sin(frame / 30) * 20
                    canvas.update_face(points, SURFACE_EDGES)
                    if canvas is wire:
                        wire.render(pixmap)
                    else:
                        surface.makeCurrent()
                        surface.paintGL()
                        surface.gl.glFinish()  # Benchmark only; no pixel capture/readback.
                        error = surface.gl.glGetError()
                        if error:
                            raise RuntimeError(f'OpenGL error during benchmark: {error}')
                        surface.doneCurrent()
                    if frame >= 10:
                        samples.append((time.perf_counter() - start) * 1000)
                print(f'{name}: update + completed paint p50 {np.median(samples):.2f} ms, '
                      f'p95 {np.percentile(samples, 95):.2f} ms')
            print('GPU draw:', f'{surface.gpu_draw_ms:.3f} ms' if surface.gpu_draw_ms is not None else 'timer unavailable')
            print('Synthetic 30 Hz smoothing:', smoothing_benchmark())
            if surface.render_error or surface._draw_count < 20:
                raise RuntimeError(surface.render_error or 'Not enough completed surface draws')
            outcome[0] = 0
        except Exception as error:
            print('Benchmark failed:', error)
        finally:
            surface.cleanup()
            window.close()
            app.quit()

    QtCore.QTimer.singleShot(300, measure)
    app.exec()
    return outcome[0]

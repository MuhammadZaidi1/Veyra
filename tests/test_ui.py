"""Offscreen Qt checks using synthetic arrays only; no camera or saved images."""

import os

# This setting belongs to this isolated test process, never the application.
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import unittest

import numpy as np
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from veyra.app import PreviewWindow
from veyra.view3d import Face3DWindow, rotate_points


class GeometryRotationTests(unittest.TestCase):
    def test_rotation_preserves_distances_and_uses_all_three_axes(self):
        points = np.array(((1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 3.0)))
        rotated = rotate_points(points, 35, 22)
        np.testing.assert_allclose(np.linalg.norm(rotated, axis=1), np.linalg.norm(points, axis=1))
        np.testing.assert_allclose(
            np.linalg.norm(rotated[1:] - rotated[0], axis=1),
            np.linalg.norm(points[1:] - points[0], axis=1),
        )
        np.testing.assert_allclose(rotate_points(points, 0, 0), points)
        np.testing.assert_allclose(rotate_points(points[:1], 90, 0), ((0, 0, -1),), atol=1e-15)


class WindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.window = Face3DWindow()
        self.points = np.array(((-0.2, -0.15, -0.05), (0.2, -0.15, -0.05), (0, 0.2, 0.1)))
        self.edges = np.array(((0, 1), (1, 2), (2, 0)), dtype=int)
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.app.processEvents()

    def test_synthetic_mesh_renders_and_no_face_clears_geometry(self):
        self.window.update_face(self.points, self.edges)
        pixmap = QtGui.QPixmap(self.window.size())
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        self.window.render(pixmap)
        self.assertFalse(pixmap.isNull())
        self.assertGreater(pixmap.toImage().pixelColor(0, 0).alpha(), 0)
        self.window.update_face(None, self.edges)
        self.assertIsNone(self.window.canvas.points)
        self.assertEqual(self.window.canvas.edges.shape, (0, 2))
        self.window.render(pixmap)

    def test_drag_zoom_and_live_update_preserve_user_view_then_reset(self):
        self.window.update_face(self.points, self.edges)
        canvas = self.window.canvas
        reference_span = canvas.reference_span
        QtTest.QTest.mousePress(canvas, QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(80, 80))
        QtTest.QTest.mouseMove(canvas, QtCore.QPoint(120, 100))
        QtTest.QTest.mouseRelease(canvas, QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(120, 100))
        self.assertNotEqual(canvas.yaw, 0)
        self.assertNotEqual(canvas.pitch, 0)
        wheel = QtGui.QWheelEvent(
            QtCore.QPointF(100, 100), QtCore.QPointF(100, 100),
            QtCore.QPoint(), QtCore.QPoint(0, 120),
            QtCore.Qt.MouseButton.NoButton, QtCore.Qt.KeyboardModifier.NoModifier,
            QtCore.Qt.ScrollPhase.ScrollUpdate, False,
        )
        QtWidgets.QApplication.sendEvent(canvas, wheel)
        self.assertGreater(canvas.zoom, 1)
        old_view = canvas.yaw, canvas.pitch, canvas.zoom
        self.window.update_face(self.points * 1.1, self.edges)
        self.assertEqual((canvas.yaw, canvas.pitch, canvas.zoom), old_view)
        self.assertEqual(canvas.reference_span, reference_span)
        self.window.findChild(QtWidgets.QPushButton).click()
        self.assertEqual((canvas.yaw, canvas.pitch, canvas.zoom), (0, 0, 1))

    def test_invalid_edges_and_nonfinite_points_rejected(self):
        for points, edges in (
            (self.points, np.array(((0, 3),))),
            (self.points, np.array(((-1, 0),))),
            (self.points, np.array((0, 1, 2))),
            (np.array(((0, 0, np.inf),)), np.empty((0, 2), dtype=int)),
        ):
            with self.subTest(points=points, edges=edges):
                with self.assertRaises(ValueError):
                    self.window.update_face(points, edges)

    def test_preview_owns_image_copy_and_can_clear_it(self):
        preview = PreviewWindow(mirror=True)
        try:
            rgb = np.zeros((12, 16, 3), dtype=np.uint8)
            rgb[:, :, 0] = 200
            preview.update_image(rgb)
            rgb[:] = 0
            self.assertEqual(preview._image.pixelColor(0, 0).red(), 200)
            preview.clear_image("Camera stopped")
            self.assertIsNone(preview._image)
            self.assertEqual(preview.preview.text(), "Camera stopped")
        finally:
            preview.close()

    def test_closing_geometry_emits_shutdown_signal_and_drops_points(self):
        self.window.update_face(self.points, self.edges)
        signals = []
        self.window.closed.connect(lambda: signals.append(True))
        self.window.close()
        self.assertTrue(signals)
        self.assertIsNone(self.window.canvas.points)

    def test_smoothing_controls_emit_configuration_without_resetting_inspection_view(self):
        received = []
        self.window.smoothing_changed.connect(lambda enabled, cutoff: received.append((enabled, cutoff)))
        self.window.canvas.yaw = 30
        self.window.smoothing.setChecked(False)
        self.window.cutoff.setValue(7)
        self.assertEqual(received[-1], (False, 7))
        self.assertEqual(self.window.canvas.yaw, 30)

    def test_software_fallback_preserves_wireframe_and_marks_surface_controls_unavailable(self):
        self.assertFalse(self.window.accelerated)
        self.assertEqual(self.window.display_mode.currentText(), 'Wireframe')
        self.assertFalse(self.window.display_mode.isEnabled())
        self.assertFalse(self.window.light_strength.isEnabled())
        self.assertIn('fallback', self.window.renderer_label.text())


if __name__ == "__main__":
    unittest.main()

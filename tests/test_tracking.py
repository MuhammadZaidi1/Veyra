"""Worker lifecycle tests with synthetic arrays and a fully mocked camera."""

import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import cv2
import numpy as np

from veyra.core import Config, relative_geometry
from veyra import tracking


class _Stop:
    def __init__(self):
        self.stopped = False
        self.checks = 0
        self.waits = []

    def is_set(self):
        self.checks += 1
        # A regression cannot turn a synthetic test into an endless loop.
        return self.stopped or self.checks > 20

    def set(self):
        self.stopped = True

    def wait(self, seconds):
        self.waits.append(seconds)
        return self.stopped


class _Mailbox:
    def __init__(self, stop):
        self.stop = stop
        self.packets = []
        self.closed = False

    def publish(self, packet):
        self.packets.append(packet)
        self.stop.set()
        return True

    def close(self):
        self.closed = True


class _Status:
    def __init__(self):
        self.packets = []
        self.closed = False

    def send(self, packet):
        self.packets.append(packet)

    def close(self):
        self.closed = True


class WorkerLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.stop = _Stop()
        self.mailbox = _Mailbox(self.stop)
        self.status = _Status()
        self.capture = mock.Mock()
        self.capture.isOpened.return_value = True
        self.frame = np.zeros((12, 16, 3), dtype=np.uint8)
        self.frame[:, :8] = (20, 40, 60)
        self.frame[:, 8:] = (80, 100, 120)
        self.capture.read.return_value = (True, self.frame)
        self.detector = mock.Mock()
        self.detector.detect_for_video.return_value = SimpleNamespace(face_landmarks=[])
        self.edges = np.array(((0, 1), (1, 2), (2, 0)), dtype=np.int32)
        self.fake_mp = SimpleNamespace(
            Image=lambda **kwargs: SimpleNamespace(**kwargs),
            ImageFormat=SimpleNamespace(SRGB="synthetic-srgb"),
        )

    def run_worker(self):
        with (
            mock.patch.dict(sys.modules, {"mediapipe": self.fake_mp}),
            mock.patch.object(cv2, "VideoCapture", return_value=self.capture) as camera,
            mock.patch.object(cv2, "setNumThreads"),
            mock.patch.object(tracking, "make_landmarker", return_value=self.detector),
            mock.patch.object(tracking, "mesh_connections", return_value=(self.edges, self.edges)),
        ):
            tracking.run_worker(Config(), self.mailbox, self.stop, self.status)
        camera.assert_called_once()

    def assert_resources_released(self):
        self.capture.release.assert_called_once_with()
        self.detector.close.assert_called_once_with()
        self.assertTrue(self.status.closed)
        self.assertTrue(self.mailbox.closed)

    def errors(self):
        return [packet["message"] for packet in self.status.packets if packet["kind"] == "error"]

    def test_no_face_publishes_live_mirrored_preview_without_geometry_and_releases(self):
        expected = cv2.cvtColor(cv2.flip(self.frame, 1), cv2.COLOR_BGR2RGB)
        self.run_worker()
        self.assertEqual(self.errors(), [])
        self.assertEqual(len(self.mailbox.packets), 1)
        packet = self.mailbox.packets[0]
        self.assertIsNone(packet["points"])
        np.testing.assert_array_equal(packet["rgb"], expected)
        self.assertEqual(packet["resolution"], (16, 12))
        self.assertGreaterEqual(packet["inference_ms"], 0)
        self.assert_resources_released()

    def test_camera_permission_failure_reports_fix_and_releases(self):
        self.capture.isOpened.return_value = False
        self.run_worker()
        self.assertEqual(len(self.errors()), 1)
        self.assertIn("Privacy & Security > Camera", self.errors()[0])
        self.capture.read.assert_not_called()
        self.detector.detect_for_video.assert_not_called()
        self.assertEqual(self.mailbox.packets, [])
        self.assert_resources_released()

    def test_inference_exception_reports_error_and_releases(self):
        self.detector.detect_for_video.side_effect = RuntimeError("synthetic inference failure")
        self.run_worker()
        self.assertEqual(self.errors(), ["synthetic inference failure"])
        self.assertEqual(self.mailbox.packets, [])
        self.assert_resources_released()

    def test_repeated_capture_failures_stop_without_publishing_stale_frames(self):
        self.capture.read.return_value = (False, None)
        self.run_worker()
        self.assertEqual(self.capture.read.call_count, 10)
        self.assertEqual(len(self.errors()), 1)
        self.assertIn("Camera stopped returning frames", self.errors()[0])
        self.detector.detect_for_video.assert_not_called()
        self.assertEqual(self.mailbox.packets, [])
        self.assert_resources_released()

    def test_synthetic_face_publishes_centered_geometry_and_mesh_overlay(self):
        normalized = np.resize(np.array(((0.25, 0.25, 0.03), (0.75, 0.25, 0.02), (0.5, 0.75, -0.03))), (478, 3))
        self.detector.detect_for_video.return_value = SimpleNamespace(
            face_landmarks=[[SimpleNamespace(x=x, y=y, z=z) for x, y, z in normalized]]
        )
        before = self.frame.copy()
        self.run_worker()
        self.assertEqual(self.errors(), [])
        packet = self.mailbox.packets[0]
        np.testing.assert_allclose(packet["points"], relative_geometry(normalized, 16, 12))
        np.testing.assert_array_equal(packet["edges"], self.edges)
        self.assertTrue(np.isfinite(packet["points"]).all())
        self.assertFalse(np.array_equal(packet["rgb"], cv2.cvtColor(cv2.flip(before, 1), cv2.COLOR_BGR2RGB)))
        self.assert_resources_released()

    def test_invalid_partial_and_missing_detections_are_tracking_loss(self):
        for landmarks in ([], [[SimpleNamespace(x=0, y=0, z=0)]],
                          [[SimpleNamespace(x=np.nan, y=0, z=0)] * 478],
                          [[SimpleNamespace(x=0, y=0)] * 478]):
            with self.subTest(landmarks_count=len(landmarks)):
                self.assertIsNone(tracking.extract_landmarks(SimpleNamespace(face_landmarks=landmarks)))

    def test_partial_detection_still_releases_camera_and_publishes_empty_geometry(self):
        self.detector.detect_for_video.return_value = SimpleNamespace(face_landmarks=[[SimpleNamespace(x=0, y=0, z=0)]])
        self.run_worker()
        self.assertIsNone(self.mailbox.packets[0]["points"])
        self.assertEqual(self.errors(), [])
        self.assert_resources_released()

    def test_tracking_loss_and_recovery_publish_new_geometry_without_freezing(self):
        face = [SimpleNamespace(x=.4, y=.4, z=.01)] * 478
        self.detector.detect_for_video.side_effect = [SimpleNamespace(face_landmarks=[face]),
                                                     SimpleNamespace(face_landmarks=[]),
                                                     SimpleNamespace(face_landmarks=[face])]
        self.capture.read.side_effect = [(True, self.frame.copy()) for _ in range(3)]
        def publish(packet):
            self.mailbox.packets.append(packet)
            if len(self.mailbox.packets) == 3:
                self.stop.set()
        self.mailbox.publish = publish
        self.run_worker()
        self.assertIsNotNone(self.mailbox.packets[0]['points'])
        self.assertIsNone(self.mailbox.packets[1]['points'])
        self.assertIsNotNone(self.mailbox.packets[2]['points'])
        self.assert_resources_released()

    def test_468_vertex_detections_do_not_index_missing_iris_landmarks(self):
        self.edges = np.array(((0, 1), (1, 2), (468, 477)), dtype=np.int32)
        self.detector.detect_for_video.return_value = SimpleNamespace(face_landmarks=[
            [SimpleNamespace(x=.4, y=.4, z=.01)] * 468])
        self.run_worker()
        self.assertEqual(self.errors(), [])
        self.assertLess(self.mailbox.packets[0]['edges'].max(), 468)
        self.assert_resources_released()


if __name__ == "__main__":
    unittest.main()

"""Numerical and startup checks without a camera or stored face data."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np

from veyra.__main__ import parse_args
from veyra.core import FrameRate, VideoClock, relative_geometry, validate_model


class RelativeGeometryTests(unittest.TestCase):
    def test_aspect_ratio_and_depth_direction(self):
        # Equal normalized x/y changes represent different pixel distances on
        # a nonsquare image; z shares x's scale and nearer z is more negative.
        landmarks = np.array(((0.2, 0.2, -0.1), (0.6, 0.6, 0.1)), dtype=np.float32)
        points = relative_geometry(landmarks, 640, 480)
        np.testing.assert_allclose(points.mean(axis=0), 0, atol=1e-7)
        np.testing.assert_allclose(points[1] - points[0], (0.4, -0.3, -0.2), atol=1e-7)
        self.assertGreater(points[0, 2], points[1, 2])

    def test_centering_is_translation_invariant_without_rescaling(self):
        landmarks = np.array(((0.1, 0.3, -0.2), (0.4, 0.7, 0.1), (0.8, 0.5, 0.0)))
        translated = landmarks + (0.05, -0.1, 0.2)
        original = relative_geometry(landmarks, 1280, 720)
        np.testing.assert_allclose(original, relative_geometry(translated, 1280, 720), atol=1e-7)
        self.assertAlmostEqual(float(np.ptp(original[:, 0])), 0.7, places=6)

    def test_empty_face_and_invalid_geometry(self):
        self.assertEqual(relative_geometry(np.empty((0, 3)), 640, 480).shape, (0, 3))
        for points, width, height in (
            (np.zeros((3, 2)), 640, 480),
            (np.array(((0.0, np.nan, 0.0),)), 640, 480),
            (np.zeros((3, 3)), 0, 480),
            (np.zeros((3, 3)), 640, -1),
        ):
            with self.subTest(width=width, height=height, points=points):
                with self.assertRaises(ValueError):
                    relative_geometry(points, width, height)


class TimingTests(unittest.TestCase):
    def test_video_clock_is_strictly_monotonic_for_duplicate_or_earlier_times(self):
        clock = VideoClock()
        timestamps = [clock.tick(t) for t in (0, 0, 0.0001, 0.033, 0.030, 0.1)]
        self.assertEqual(timestamps[0], 0)
        self.assertTrue(all(b > a for a, b in zip(timestamps, timestamps[1:])))
        self.assertEqual(timestamps[-1], 100)

    def test_fps_counts_frame_intervals_and_rolls_its_window(self):
        rate = FrameRate(window=3)
        self.assertEqual(rate.update(0.0), 0.0)
        self.assertAlmostEqual(rate.update(0.05), 20.0)
        self.assertAlmostEqual(rate.update(0.1), 20.0)
        self.assertAlmostEqual(rate.update(0.2), 2 / 0.15)
        self.assertEqual(len(rate.times), 3)


class StartupTests(unittest.TestCase):
    def test_missing_and_corrupt_models_have_actionable_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "face_landmarker.task"
            with self.assertRaisesRegex(ValueError, "Model missing.*"):
                validate_model(model)
            model.write_bytes(b"synthetic invalid model; no face data")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                validate_model(model)

    def test_invalid_cli_values_fail_before_camera_startup(self):
        for arguments in (
            ["--camera", "-1"],
            ["--width", "0"],
            ["--height", "1100"],
            ["--fps", "0"],
            ["--fps", "61"],
            ["--duration", "-1"],
            ["--duration", "nan"],
            ["--duration", "inf"],
            ["--smoothing-cutoff", "nan"],
            ["--smoothing-cutoff", "0"],
            ["--smoothing-beta", "-1"],
            ["--smoothing-beta", "inf"],
            ["--renderer", "unsupported"],
        ):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    parse_args(arguments)
                self.assertEqual(result.exception.code, 2)

    def test_default_cli_has_cpu_prototype_capture_settings(self):
        args = parse_args([])
        self.assertEqual((args.camera, args.width, args.height, args.fps), (0, 640, 480, 30))
        self.assertFalse(args.no_mirror)
        self.assertEqual(args.duration, 0)


if __name__ == "__main__":
    unittest.main()

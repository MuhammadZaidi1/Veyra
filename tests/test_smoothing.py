"""Invented signals test jitter reduction, responsiveness and tracking resets."""

import unittest

import numpy as np

from veyra.diagnostics import smoothing_benchmark
from veyra.smoothing import GeometryStream, OneEuroFilter


class SmoothingTests(unittest.TestCase):
    def test_constant_geometry_does_not_drift(self):
        filter_ = OneEuroFilter()
        points = np.full((478, 3), 0.12, dtype=np.float32)
        for frame in range(90):
            np.testing.assert_array_equal(filter_.update(points, frame / 30), points)

    def test_synthetic_noise_reduced_and_step_reaches_ninety_percent_promptly(self):
        result = smoothing_benchmark()
        self.assertLess(result["filtered_rms"], result["raw_rms"] * 0.75)
        self.assertLessEqual(result["step_90_ms"], 100)

    def test_motion_adaptation_responds_faster_than_fixed_filter_without_overshoot(self):
        adaptive, fixed = OneEuroFilter(beta=20), OneEuroFilter(beta=0)
        start = np.zeros((3, 3), dtype=np.float32)
        adaptive.update(start, 0); fixed.update(start, 0)
        a = adaptive.update(start + 0.05, 1 / 30)
        b = fixed.update(start + 0.05, 1 / 30)
        self.assertTrue(np.all(a > b))
        self.assertTrue(np.all((a >= 0) & (a <= 0.05)))

    def test_disabled_output_is_unchanged_and_toggle_resets_old_filter_history(self):
        stream = GeometryStream(enabled=False)
        points = np.zeros((478, 3), dtype=np.float32)
        stream.update(points, 0)
        output = stream.update(points + 0.1, 1 / 30)
        np.testing.assert_array_equal(output, points + 0.1)
        self.assertAlmostEqual(stream.raw_motion, stream.output_motion)
        stream.configure(True, 4)
        np.testing.assert_array_equal(stream.update(points + 0.2, 2 / 30), points + 0.2)

    def test_loss_invalid_landmarks_and_recovery_clear_history(self):
        stream = GeometryStream()
        points = np.zeros((478, 3), dtype=np.float32)
        for invalid in (None, np.zeros((467, 3)), np.full((478, 3), np.nan)):
            stream.update(points, 0)
            stream.update(points + 0.01, 1 / 30)
            self.assertIsNone(stream.update(invalid, 2 / 30))
            self.assertIsNone(stream.filter.previous)
            self.assertEqual(stream.comparisons, 0)
            np.testing.assert_array_equal(stream.update(points + 0.2, 3 / 30), points + 0.2)

    def test_capture_gap_and_repeated_timestamps_do_not_blend_stale_geometry(self):
        stream = GeometryStream()
        points = np.zeros((468, 3), dtype=np.float32)
        stream.update(points, 0)
        np.testing.assert_array_equal(stream.update(points + 0.3, 1), points + 0.3)
        np.testing.assert_array_equal(stream.update(points + 0.4, 1), points + 0.4)

    def test_higher_cutoff_gives_faster_response_and_parameters_validated(self):
        slow, fast = OneEuroFilter(min_cutoff=1, beta=0), OneEuroFilter(min_cutoff=10, beta=0)
        points = np.zeros((3, 3), dtype=np.float32)
        for f in (slow, fast):
            f.update(points, 0)
        self.assertTrue(np.all(fast.update(points + 1, 1/30) > slow.update(points + 1, 1/30)))
        for cutoff, beta in ((0, 1), (-1, 1), (float('nan'), 1), (4, -1)):
            with self.assertRaises(ValueError):
                OneEuroFilter(cutoff, beta)

    def test_landmark_count_change_resets_comparison_and_filter_state(self):
        stream = GeometryStream()
        points = np.zeros((468, 3), dtype=np.float32)
        stream.update(points, 0)
        stream.update(points + .01, 1/30)
        recovered = np.full((478, 3), .2, dtype=np.float32)
        np.testing.assert_array_equal(stream.update(recovered, 2/30), recovered)
        self.assertEqual(stream.comparisons, 0)

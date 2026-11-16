"""Bounds and diagnostic mathematics; these tests perform no optimization."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from scripts.canonical_control_geometry import ROOT, FRAME_WIDTH, REFERENCE, read_canonical
from scripts.canonical_learning_curve import check_plan, check_recorded_baseline, fitted_yaws, score
from veyra3d.model import Veyra3D


class LearningCurveTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((ROOT / "configs/veyra3d/canonical_learning_curve.json").read_text())

    def test_fresh_render_needs_no_separately_generated_baseline_file(self):
        with tempfile.TemporaryDirectory() as directory:
            check_recorded_baseline(directory, .01471475)
            path = Path(directory) / "constant-baseline.json"
            path.write_text(json.dumps({"synthetic_validation_relative_vertex_error": .01471475}))
            check_recorded_baseline(directory, .01471475)
            path.write_text(json.dumps({"synthetic_validation_relative_vertex_error": .1}))
            with self.assertRaisesRegex(ValueError, "baseline changed"):
                check_recorded_baseline(directory, .01471475)

    def test_step_cap_and_original_architecture_are_enforced(self):
        config = check_plan(self.plan)
        self.assertEqual(Veyra3D(config.model).decoder[-1].out_features, 468 * 3)

        for steps in (0, 201, True):
            changed = copy.deepcopy(self.plan)
            changed["experiments"][0]["max_steps"] = steps
            with self.assertRaisesRegex(ValueError, "200"):
                check_plan(changed)
        self.plan["execution"]["model"]["latent_dim"] = 128
        with self.assertRaisesRegex(ValueError, "architecture"):
            check_plan(self.plan)

    def test_strict_determinism_and_vertex_only_loss_are_enforced(self):
        self.plan["execution"]["deterministic"] = False
        with self.assertRaises(ValueError):
            check_plan(self.plan)
        self.plan["execution"]["deterministic"] = True
        self.plan["execution"]["loss"]["normal"] = .01
        with self.assertRaises(ValueError):
            check_plan(self.plan)

    def test_loss_and_reported_euclidean_error_have_distinct_scales(self):
        target = torch.zeros((1, 468, 3))
        predicted = target + .01
        result = score(predicted, target, beta=.02)
        self.assertAlmostEqual(result["loss"], .0025, places=7)
        self.assertAlmostEqual(result["relative_vertex_error"], np.sqrt(3) * .01, places=7)
        self.assertEqual(score(target, target, .02)["loss"], 0.)
        predicted[0, 13, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            score(predicted, target, .02)

    def test_yaw_diagnostic_recovers_exact_rotated_template(self):
        vertices, _ = read_canonical(REFERENCE / "canonical_face_model.obj")
        front = (vertices - vertices.mean(0)) * .01 / FRAME_WIDTH
        rotated = []
        expected = [-12, -6, 0, 6, 12]
        for yaw in expected:
            angle = np.deg2rad(yaw)
            rotation = np.array([[np.cos(angle), 0, -np.sin(angle)], [0, 1, 0],
                                 [np.sin(angle), 0, np.cos(angle)]])
            rotated.append(front @ rotation.T)
        np.testing.assert_allclose(fitted_yaws(np.stack(rotated), front), expected, atol=1e-12)

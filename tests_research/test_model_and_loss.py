"""All tensors are invented; none represents a webcam capture or real face."""

from dataclasses import replace
import unittest

import torch

from veyra3d.config import ExperimentConfig, LossConfig, ModelConfig
from veyra3d.data import SyntheticDataset
from veyra3d.model import Veyra3D
from veyra3d.objectives import GeometryObjective, MetricAccumulator
from veyra3d.runtime import initialize


class ModelTests(unittest.TestCase):
    def setUp(self):
        initialize(ExperimentConfig())

    def test_default_model_is_small_and_outputs_centered_finite_geometry(self):
        model = Veyra3D()
        output = model(torch.zeros(2, 3, 128, 128))
        self.assertEqual(output.shape, (2, 468, 3))
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output.mean(dim=1), torch.zeros(2, 3), atol=1e-7, rtol=0)
        self.assertLess(model.parameter_count, 1000000)

    def test_single_image_forward_backward_and_optimizer_step(self):
        model = Veyra3D(ModelConfig(64, 32))
        sample = SyntheticDataset(1, 64, 42)[0]
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        before = model.decoder[-1].weight.detach().clone()
        loss, _ = GeometryObjective()(model(sample['image'][None]), sample['vertices'][None], sample['valid'][None])
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.decoder[-1].weight))

    def test_input_contract_rejects_wrong_shapes_and_byte_images(self):
        model = Veyra3D()
        for image in (torch.zeros(1, 3, 64, 64), torch.zeros(1, 1, 128, 128),
                      torch.zeros(3, 128, 128), torch.zeros(0, 3, 128, 128),
                      torch.zeros(1, 3, 128, 128, dtype=torch.uint8)):
            with self.subTest(shape=image.shape), self.assertRaises(ValueError):
                model(image)


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.target = SyntheticDataset(1, 64, 42)[0]['vertices'][None]
        self.valid = torch.ones(1, 468, dtype=torch.bool)
        self.objective = GeometryObjective()

    def test_matching_geometry_has_near_zero_loss(self):
        prediction = self.target.clone().requires_grad_()
        loss, stats = self.objective(prediction, self.target, self.valid)
        self.assertLess(float(loss.detach()), 1e-6)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(int(stats['valid_vertices']), 468)

    def test_masked_nan_labels_do_not_affect_loss_or_invalid_vertex_gradients(self):
        valid = torch.zeros_like(self.valid)
        valid[:, 42] = True
        target = torch.full_like(self.target, float('nan'))
        target[:, 42] = .1
        prediction = torch.zeros_like(self.target, requires_grad=True)
        loss, _ = self.objective(prediction, target, valid)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(int(torch.count_nonzero(prediction.grad[~valid])), 0)
        self.assertGreater(float(prediction.grad[valid].abs().sum()), 0)

    def test_empty_supervision_has_finite_zero_objective(self):
        prediction = torch.randn_like(self.target, requires_grad=True)
        loss, _ = self.objective(prediction, torch.full_like(self.target, float('nan')), torch.zeros_like(self.valid))
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_invalid_or_nonfinite_supervised_geometry_rejected(self):
        for target, valid in ((self.target, self.valid.float()),
                              (torch.full_like(self.target, float('nan')), self.valid),
                              (self.target[:, :467], self.valid)):
            with self.assertRaises(ValueError):
                self.objective(self.target, target, valid)

    def test_metrics_weight_valid_vertices_across_unequal_batches(self):
        objective = GeometryObjective(LossConfig(edge=0, normal=0))
        accumulator = MetricAccumulator(objective.config)
        for error, count in ((.1, 1), (.2, 3)):
            valid = torch.zeros_like(self.valid)
            valid[:, :count] = True
            prediction = torch.zeros_like(self.target)
            target = torch.zeros_like(self.target)
            target[:, :count, 0] = error
            _, stats = objective(prediction, target, valid)
            accumulator.update(stats)
        self.assertAlmostEqual(accumulator.result()['relative_vertex_error'], (.1 + .2*3)/4, places=6)
        self.assertEqual(accumulator.result()['supervised_vertices'], 4)


class ConfigTests(unittest.TestCase):
    def test_config_roundtrip_and_unknown_or_invalid_values(self):
        config = ExperimentConfig.from_dict(ExperimentConfig().to_dict())
        self.assertEqual(config, ExperimentConfig())
        for change in ({'typo': 1}, {'batch_size': 0}, {'epochs': -1}, {'amp': 'false'},
                       {'model': {'image_size': 65}}, {'model': {'image_size': 128.0}},
                       {'learning_rate': float('nan')}, {'learning_rate': True},
                       {'loss': {'vertex': 0}}, {'data': {'kind': 'download'}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                ExperimentConfig.from_dict(change)

    def test_resume_fingerprint_allows_execution_changes_but_not_changed_training(self):
        config = ExperimentConfig()
        self.assertEqual(config.fingerprint(), replace(config, device='mps', epochs=3, run_dir='other').fingerprint())
        self.assertNotEqual(config.fingerprint(), replace(config, learning_rate=.1).fingerprint())
        self.assertNotEqual(config.fingerprint(), replace(config, seed=7).fingerprint())

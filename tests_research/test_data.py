"""Only colored temporary images and invented grid coordinates are generated."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from veyra3d.config import DataConfig, ExperimentConfig, ModelConfig
from veyra3d.data import ManifestDataset, SyntheticDataset, local_path, make_datasets, preprocess_rgb


def write_sample(root, name='one', group='opaque-a', color=(30, 90, 160), size=(40, 80)):
    Image.new('RGB', size, color).save(root / f'{name}.png')
    points = SyntheticDataset(1, 64, 42)[0]['vertices'].numpy()
    np.savez(root / f'{name}.npz', vertices=points, valid=np.ones(468, dtype=np.bool_))
    return {'image': f'{name}.png', 'geometry': f'{name}.npz', 'group': group}


class DataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='veyra3d-test-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def manifest(self, name, records):
        (self.root / name).write_text(''.join(json.dumps(record)+'\n' for record in records))

    def test_synthetic_samples_are_repeatable_independent_and_centered(self):
        dataset = SyntheticDataset(3, 64, 42)
        first, again, second = dataset[0], dataset[0], dataset[1]
        torch.testing.assert_close(first['image'], again['image'], rtol=0, atol=0)
        self.assertFalse(torch.equal(first['image'], second['image']))
        self.assertLessEqual(float(first['image'].abs().max()), 1)
        torch.testing.assert_close(first['vertices'].mean(dim=0), torch.zeros(3), atol=1e-7, rtol=0)
        datasets = make_datasets(ExperimentConfig())
        self.assertFalse(torch.equal(datasets['train'][0]['image'], datasets['val'][0]['image']))

    def test_tall_crop_is_letterboxed_and_all_xyz_scale_is_isotropic(self):
        record = write_sample(self.root)
        self.manifest('train.jsonl', [record])
        dataset = ManifestDataset(self.root, 'train.jsonl', 64)
        sample = dataset[0]
        self.assertEqual(sample['image'].shape, (3, 64, 64))
        torch.testing.assert_close(sample['vertices'], SyntheticDataset(1, 64, 42)[0]['vertices'] * .5)
        padding = sample['image'][:, :, :16]
        torch.testing.assert_close(padding, torch.full_like(padding, 128/127.5-1))
        self.assertTrue(sample['valid'].all())

    def test_missing_supervision_is_zeroed_but_never_inferred(self):
        record = write_sample(self.root)
        points = SyntheticDataset(1, 64, 42)[0]['vertices'].numpy()
        valid = np.ones(468, dtype=np.bool_); valid[10] = False
        points[10] = np.nan
        np.savez(self.root/'one.npz', vertices=points, valid=valid)
        self.manifest('train.jsonl', [record])
        sample = ManifestDataset(self.root, 'train.jsonl', 64)[0]
        self.assertFalse(bool(sample['valid'][10]))
        torch.testing.assert_close(sample['vertices'][10], torch.zeros(3))

    def test_bad_geometry_shapes_nonfinite_labels_and_empty_masks_rejected(self):
        record = write_sample(self.root)
        self.manifest('train.jsonl', [record])
        for points, valid in ((np.zeros((467,3), np.float32), np.ones(468, np.bool_)),
                              (np.full((468,3), np.nan, np.float32), np.ones(468, np.bool_)),
                              (np.zeros((468,3), np.float32), np.zeros(468, np.bool_)),
                              (np.ones((468,3), np.float32), np.ones(468, np.bool_))):
            np.savez(self.root/'one.npz', vertices=points, valid=valid)
            with self.assertRaises(ValueError):
                ManifestDataset(self.root, 'train.jsonl', 64)[0]

    def test_paths_cannot_escape_root_or_follow_external_symlinks(self):
        root = self.root/'dataset';root.mkdir()
        (self.root/'outside.txt').write_text('synthetic fixture')
        for path in ('../outside.txt', str(self.root/'outside.txt'), 'missing.png'):
            with self.assertRaises(ValueError):
                local_path(root, path)
        (root/'link.txt').symlink_to(self.root/'outside.txt')
        with self.assertRaises(ValueError):
            local_path(root, 'link.txt')

    def test_split_group_and_copied_image_content_leakage_rejected(self):
        first = write_sample(self.root, 'first', 'same-group')
        second = write_sample(self.root, 'second', 'same-group', color=(80,30,20))
        self.manifest('train.jsonl', [first]);self.manifest('val.jsonl', [second])
        config = ExperimentConfig(data=DataConfig(kind='manifest', root=str(self.root)))
        with self.assertRaisesRegex(ValueError, 'leakage'):
            make_datasets(config)
        second['group'] = 'different-group'
        (self.root/'second.png').write_bytes((self.root/'first.png').read_bytes())
        self.manifest('val.jsonl', [second])
        with self.assertRaisesRegex(ValueError, 'leakage'):
            make_datasets(config)

    def test_content_signature_changes_if_labels_change(self):
        record = write_sample(self.root)
        self.manifest('train.jsonl', [record])
        before = ManifestDataset(self.root, 'train.jsonl', 64).signature
        points = SyntheticDataset(1,64,42)[0]['vertices'].numpy() * .8
        np.savez(self.root/'one.npz', vertices=points, valid=np.ones(468, np.bool_))
        self.assertNotEqual(before, ManifestDataset(self.root, 'train.jsonl', 64).signature)

    def test_empty_unknown_and_duplicate_records_rejected(self):
        self.manifest('train.jsonl', [])
        with self.assertRaisesRegex(ValueError, 'empty'):
            ManifestDataset(self.root, 'train.jsonl', 64)
        record = write_sample(self.root)
        for records in ([record, record], [dict(record, unknown=1)]):
            self.manifest('train.jsonl', records)
            with self.assertRaises(ValueError):
                ManifestDataset(self.root, 'train.jsonl', 64)

    def test_exif_rotation_is_rejected_instead_of_misaligning_labels(self):
        image = Image.new('RGB', (40, 80))
        image.getexif()[274] = 6
        with self.assertRaisesRegex(ValueError, 'EXIF'):
            preprocess_rgb(image, 64)

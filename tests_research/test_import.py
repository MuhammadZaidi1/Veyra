"""Importer tests use invented grids and colored images, never facial data."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.import_registered_geometry import import_dataset
from test_data import write_sample
from veyra3d import GEOMETRY_CONTRACT


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.output = self.root / 'output'
        self.rows = []
        for index, split in enumerate(('train', 'val', 'test')):
            row = write_sample(self.source, split, f'person-{index}', (index*50, 70, 100))
            self.rows.append({**row, 'split':split})
        # This assertion metadata is a test fixture, not provenance for real data.
        self.review = {'contract':GEOMETRY_CONTRACT, 'label_source':'scan-derived', 'reviewed':True,
            'license_reference':'fixture only', 'registration_reference':'fixture only',
            'coordinate_conversion_reference':'fixture only'}
        self.save()

    def save(self):
        (self.source/'index.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in self.rows))
        (self.source/'review.json').write_text(json.dumps(self.review))

    def run_import(self):
        return import_dataset(self.source, 'index.jsonl', 'review.json', self.output)

    def test_valid_package_and_overwrite_refusal(self):
        self.assertEqual(self.run_import(), {'train':1, 'val':1, 'test':1})
        self.assertEqual(json.loads((self.output/'import-review.json').read_text()), self.review)
        with self.assertRaisesRegex(ValueError, 'must not exist'):
            self.run_import()

    def test_split_leakage_and_cleanup(self):
        self.rows[1]['group'] = self.rows[0]['group']
        self.save()
        with self.assertRaisesRegex(ValueError, 'leakage'):
            self.run_import()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob('.geometry-import-*')))

    def test_2d_labels_are_rejected(self):
        np.savez(self.source/'train.npz', vertices=np.zeros((468,2), np.float32), valid=np.ones(468, bool))
        with self.assertRaisesRegex(ValueError, '468,3'):
            self.run_import()
        self.assertFalse(self.output.exists())

    def test_unreviewed_or_pseudo_labels_are_rejected(self):
        self.review['label_source'] = 'pseudo-3D'
        self.save()
        with self.assertRaisesRegex(ValueError, 'scan-derived'):
            self.run_import()

    def test_source_escape_is_rejected(self):
        self.rows[0]['image'] = '../outside.png'
        self.save()
        with self.assertRaisesRegex(ValueError, 'escapes'):
            self.run_import()

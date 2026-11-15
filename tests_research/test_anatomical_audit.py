"""Optional local audit regression checks; no network, Blender, or training."""
import hashlib
import os
from pathlib import Path
import tempfile
import unittest

from scripts.audit_correspondence_anatomy import audit

ROOT=Path(__file__).resolve().parents[1]
ASSET_ROOT=Path(os.environ.get('APPDATA',''))/'Blender Foundation/Blender/5.3/extensions/blender_org/mpfb'
MAPPING=ROOT/'runs/veyra3d/correspondence-anatomical-final'
AVAILABLE=(MAPPING/'mapping-candidates.json').is_file() and (ASSET_ROOT/'data/3dobjs/base.obj').is_file()


@unittest.skipUnless(AVAILABLE,'Local pinned mapping/MPFB inspection fixtures unavailable')
class FocusedAnatomyAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.addClassCleanup(cls.temp.cleanup)
        cls.output=Path(cls.temp.name)/'audit'
        cls.mapping_hash=hashlib.sha256((MAPPING/'mapping-candidates.json').read_bytes()).hexdigest()
        cls.report=audit(MAPPING,ROOT/'runs/veyra3d/blender-prototype-002',ROOT/'third_party/mediapipe-reference',ROOT/'runs/veyra3d/correspondence-002',ASSET_ROOT,cls.output)

    def test_mouth_cap_warnings_are_not_misclassified_as_veyra_skin_failures(self):
        self.assertTrue(self.report['normal_warnings'])
        self.assertTrue(all(w['canonical_mouth_cap'] and not w['exists_in_veyra_skin'] for w in self.report['normal_warnings']))
        edge=next(w for w in self.report['edge_warnings'] if w['landmarks']==[13,14])
        self.assertFalse(edge['exists_in_veyra_skin'])
        self.assertFalse(edge['same_source_point'])
        self.assertGreater(edge['mapped_length'],0)

    def test_oval_and_mapping_integrity_are_checked_without_approval(self):
        self.assertTrue(self.report['face_oval']['canonical_boundary_matches_official_oval'])
        self.assertTrue(self.report['face_oval']['oval_is_boundary_of_veyra_skin'])
        self.assertEqual(self.report['mapping_sha256'],self.mapping_hash)
        self.assertEqual(hashlib.sha256((MAPPING/'mapping-candidates.json').read_bytes()).hexdigest(),self.mapping_hash)
        self.assertEqual(self.report['status'],'focused_audit_unvalidated')
        self.assertFalse(self.report['source_material_region']['is_verified_vermilion_boundary'])

    def test_audit_refuses_to_overwrite_existing_outputs(self):
        with self.assertRaisesRegex(ValueError,'fresh'):
            audit(MAPPING,ROOT/'runs/veyra3d/blender-prototype-002',ROOT/'third_party/mediapipe-reference',ROOT/'runs/veyra3d/correspondence-002',ASSET_ROOT,self.output)

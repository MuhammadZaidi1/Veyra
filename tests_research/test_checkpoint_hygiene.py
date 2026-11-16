"""Repository hygiene checks: no generated assets/private data in a checkpoint."""
import hashlib,json,re,subprocess,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class CheckpointHygieneTests(unittest.TestCase):
 def test_ignored_artifacts_and_public_reference_exception(self):
  paths=['renders/face.png','renders/geometry.json','private/person.json','private/notes.txt','.private/local.json','secrets/local.json','.aws/credentials','private/me.jpeg','private/ME.JPG','private/SCAN.OBJ','data/person/review.json','datasets/train.jsonl','checkpoints/a.pt','.venv-research/Scripts/python.exe','scratch/scene.blend','scratch/scene.blend1','scratch/render.exr','temp/a.json','a.tmp','Microsoft/Windows/PowerShell/ModuleAnalysisCache']
  r=subprocess.run(['git','check-ignore','--no-index',*paths],cwd=ROOT,capture_output=True,text=True)
  self.assertEqual(set(r.stdout.splitlines()),set(paths))
  r=subprocess.run(['git','check-ignore','--no-index','third_party/mediapipe-reference/canonical_face_model.obj'],cwd=ROOT,capture_output=True,text=True)
  self.assertEqual(r.returncode,1)
 def test_pinned_reference_and_notices_match(self):
  folder=ROOT/'third_party/mediapipe-reference';p=json.loads((folder/'provenance.json').read_text())
  for name,meta in p['files'].items():self.assertEqual(hashlib.sha256((folder/name).read_bytes()).hexdigest(),meta['sha256'])
  self.assertTrue((ROOT/'third_party/blender-scripts-GPL-3.0-or-later.txt').is_file())
 def test_portable_code_has_no_absolute_windows_paths(self):
  for folder in ['scripts','veyra3d']:
   for path in (ROOT/folder).glob('*.py'):
    self.assertIsNone(re.search(r'(?<![A-Za-z])[A-Za-z]:[\\/]',path.read_text(encoding='utf-8-sig')),str(path))
if __name__=='__main__':unittest.main()

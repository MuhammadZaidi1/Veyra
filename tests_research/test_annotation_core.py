import copy,json,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from annotation_core import rebuild,digest
class AnnotationTests(unittest.TestCase):
 def setUp(self):
  self.v=np.array([[0.,0,0],[1,0,0],[0,1,0]]);self.t=np.array([[0,1,2]],np.int32);self.ids=np.array([3,4,5]);self.b={'asset_sha256':'asset','topology_sha256':digest(self.t.tobytes()),'slots':[{'veyra_index':i,'source_triangle':0,'source_vertex_ids':[3,4,5],'barycentric':[1.,0,0]} for i in range(468)]};self.a={'base_mapping_sha256':'hash','asset_sha256':'asset','topology_sha256':self.b['topology_sha256'],'annotations':[{'veyra_index':17,'source_triangle':0,'barycentric':[.2,.3,.5],'decision':'accepted','reviewer':'unit-test fixture'}]}
 def run_rebuild(self,**kw):return rebuild(self.b,self.a,self.v,self.t,self.ids,'hash',**kw)
 def test_preserves_order_and_other_slots(self):
  r=self.run_rebuild();self.assertEqual(r['status'],'candidate_unvalidated');self.assertFalse(r['slots'][17]['human_reviewed']);self.assertEqual(r['slots'][:17],self.b['slots'][:17]);self.assertEqual(r['slots'][18:],self.b['slots'][18:])
 def test_no_silent_draft(self):
  self.a['annotations'][0]['decision']='unreviewed'
  with self.assertRaises(ValueError):self.run_rebuild()
  self.assertEqual(self.run_rebuild(allow_draft_preview=True)['annotation_applied_indices'],[])
 def test_requires_reviewer(self):
  self.a['annotations'][0]['reviewer']=''
  with self.assertRaises(ValueError):self.run_rebuild()
 def test_invalid_weights(self):
  self.a['annotations'][0]['barycentric']=[-1,1,1]
  with self.assertRaises(ValueError):self.run_rebuild()
 def test_wrong_source(self):
  self.a['base_mapping_sha256']='other'
  with self.assertRaises(ValueError):self.run_rebuild()
 def test_duplicate_annotation(self):
  self.a['annotations']*=2
  with self.assertRaises(ValueError):self.run_rebuild()
if __name__=='__main__':unittest.main()

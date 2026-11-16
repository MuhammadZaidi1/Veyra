"""Invented geometry and temporary core-delta fixtures; no Blender, datasets or training."""
import gzip,tempfile,unittest
from pathlib import Path
import numpy as np
from scripts.correspondence_gate import structural_checks,support_checks,quad_graph,shortest_mesh_path,semantic_decision
from scripts.makehuman_semantics import CoreTargets

class CorrespondenceGateTests(unittest.TestCase):
    def setUp(self):
        self.p=np.array([[0.,0,0],[1,0,0],[1,1,0],[0,1,0]]);self.f=np.array([[0,1,2],[0,2,3]])
    def check(self,p):return structural_checks(self.p,p,self.f)
    def test_identical_geometry_passes(self):self.assertTrue(self.check(self.p)['passed'])
    def test_reversal_is_rejected(self):
        p=self.p.copy();p[1]=[-1,0,0];r=self.check(p);self.assertFalse(r['passed']);self.assertTrue(r['reversals'])
    def test_zero_area_is_rejected_even_without_negative_normal_dot(self):
        p=self.p.copy();p[1]=[.5,.5,0];r=self.check(p);self.assertFalse(r['passed']);self.assertTrue(r['degenerate_triangles'])
    def test_severe_area_compression_rejected(self):
        p=self.p.copy();p[1]=[.5001,.5,0];self.assertTrue(self.check(p)['degenerate_triangles'])
    def test_large_edge_distortion_rejected(self):
        p=self.p.copy();p[1]=[10,0,0];self.assertTrue(self.check(p)['distorted_edges'])
    def test_collision_rejected(self):
        p=self.p.copy();p[3]=p[1];r=self.check(p);self.assertIn([1,3],r['collisions']);self.assertFalse(r['passed'])
    def test_mouth_order_checked_explicitly(self):
        r=structural_checks(self.p,self.p,self.f,[[1,0]]);self.assertEqual(r['mouth_order_failures'],[[1,0]])
    def test_same_pose_reference_avoids_global_rotation_false_positive(self):
        p=self.p[:,[0,2,1]];self.assertTrue(structural_checks(p,p,self.f)['passed'])
    def test_nonfinite_and_wrong_shape_fail_closed(self):
        p=self.p.copy();p[0,0]=np.nan;self.assertFalse(self.check(p)['passed']);self.assertFalse(self.check(self.p[:2])['passed'])
    def test_source_support_inversion_rejected(self):
        slots=[{'source_triangle':0,'barycentric':[.2,.3,.5]}];p=self.p.copy();p[1]=[-1,0,0];self.assertEqual(support_checks(self.p,p,self.f,slots),[0])
    def test_bad_barycentric_weights_rejected(self):
        with self.assertRaises(ValueError):support_checks(self.p,self.p,self.f,[{'source_triangle':0,'barycentric':[-.1,.6,.5]}])
    def test_only_true_quad_edges_used(self):
        graph=quad_graph(self.p,[[10,11,12,13]],np.array([10,11,12,13]),np.ones(4,bool));path=shortest_mesh_path(graph,0,2);self.assertEqual(len(path),3);self.assertNotIn(2,graph[0])
    def test_disconnected_paths_rejected(self):
        with self.assertRaises(ValueError):shortest_mesh_path({0:{},1:{}},0,1)
    def test_geometric_success_does_not_become_semantic_approval(self):
        self.assertEqual(semantic_decision(True,False),'retain_baseline_semantics_unresolved');self.assertEqual(semantic_decision(False,True),'reject_structural')
    def test_core_delta_axis_conversion_and_missing_source_ids(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'data/targets/test.target.gz';p.parent.mkdir(parents=True);p.write_bytes(gzip.compress(b'10 1 2 3\n999 2 4 6\n'))
            reader=CoreTargets(root,[10,11]);np.testing.assert_allclose(reader.load('test.target.gz'),[[.1,-.3,.2],[0,0,0]])
    def test_core_target_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):CoreTargets(d,[0]).load('../../outside.gz')

if __name__=='__main__':unittest.main()

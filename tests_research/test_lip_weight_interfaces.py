"""Exact surface-field interpolation tests, not anatomical-label tests."""
import unittest
import numpy as np
from scripts.evaluate_lip_weight_interfaces import interface_segments
class LipWeightInterfaceTests(unittest.TestCase):
    def test_intersections_have_convex_exact_field_values(self):
        v=np.array([[0.,0,0],[1,0,0],[0,1,0]]);t=np.array([[0,1,2]]);f=np.array([0.,1,0]);segments=interface_segments(v,t,f,.5)
        self.assertEqual(len(segments),1);index,w=segments[0];np.testing.assert_allclose(w.sum(1),1);self.assertTrue((w>=0).all());np.testing.assert_allclose(w@f[t[index]],.5)
    def test_flat_field_has_no_invented_interface(self):
        self.assertEqual(interface_segments(np.eye(3),np.array([[0,1,2]]),np.zeros(3),.5),[])
    def test_threshold_changes_the_source_surface_site(self):
        v=np.array([[0.,0,0],[1,0,0],[0,1,0]]);t=np.array([[0,1,2]]);f=np.array([0.,1,0]);low=interface_segments(v,t,f,.25)[0][1]@v;high=interface_segments(v,t,f,.75)[0][1]@v
        self.assertGreater(float(np.linalg.norm(high.mean(0)-low.mean(0))),.1)
if __name__=='__main__':unittest.main()

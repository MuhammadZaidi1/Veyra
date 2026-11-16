"""Synthetic mesh/math checks; no facial assets, Blender or training required."""
import unittest
import numpy as np
from scripts.surface_proposal_math import mesh_evidence,percentile,ordered_loop,solve_chain,segment_crossings
class SurfaceProposalTests(unittest.TestCase):
 def test_flat_mesh_has_no_curvature_or_morph_gradient(self):
  v=np.array([[0.,0,0],[1,0,0],[0,1,0],[1,1,0]]);t=np.array([[0,1,2],[1,3,2]])
  normals,c,g,response,area=mesh_evidence(v,t,[np.zeros_like(v)])
  np.testing.assert_allclose(c,0);np.testing.assert_allclose(g,0);np.testing.assert_allclose(normals,[[0,0,1]]*4)
 def test_no_signal_is_not_high_confidence(self):
  np.testing.assert_allclose(percentile([0,0],[0,0,0]),.5)
 def test_morph_gradient_detects_local_transition(self):
  v=np.array([[0.,0,0],[1,0,0],[0,1,0]]);t=np.array([[0,1,2]]);delta=np.zeros_like(v);delta[0,2]=.1
  _,_,gradient,_,_=mesh_evidence(v,t,[delta]);self.assertGreater(gradient[0],gradient[1]);self.assertGreater(gradient.min(),0)
 def test_ordered_loop_rejects_branch_and_disconnected_cycles(self):
  self.assertEqual(ordered_loop([(0,1),(1,2),(2,0)],0),[0,1,2])
  with self.assertRaises(ValueError):ordered_loop([(0,1),(1,2)],0)
  with self.assertRaises(ValueError):ordered_loop([(0,1),(1,2),(2,0),(3,4),(4,5),(5,3)],0)
 def test_joint_selection_keeps_neighbor_order(self):
  baseline=np.array([[0.,0,0],[1,0,0],[2,0,0]]);pools=[np.array([[0.,0,0]]),np.array([[3.,0,0],[1,0,0]]),np.array([[2.,0,0]])]
  self.assertEqual(solve_chain(pools,[np.array([0.]),np.array([-10.,1]),np.array([0.])],baseline,x_order=True),[0,1,0])
 def test_infeasible_chain_is_not_silently_relaxed(self):
  with self.assertRaises(ValueError):solve_chain([np.array([[1.,0,0]]),np.array([[0.,0,0]])],[np.zeros(1)]*2,np.array([[0.,0,0],[1,0,0]]),x_order=True)
 def test_detects_crossings_not_regular_square(self):
  v=np.array([[0.,0,0],[1,0,1],[0,0,1],[1,0,0]])
  self.assertEqual(len(segment_crossings(v,[0,1,2,3])),1);self.assertFalse(segment_crossings(v,[0,3,1,2]))
if __name__=='__main__':unittest.main()

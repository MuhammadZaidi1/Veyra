"""468 reference-evidence and masked-supervision checks; no model optimization or data download."""
import unittest
import numpy as np
import torch
from scripts.mediapipe_evidence_policy import eligible_mask,coverage,training_readiness,centered_camera_geometry,REGION_REQUIREMENTS
from veyra.topology_data import TRIANGLE_INDICES
from veyra3d.model import VERTEX_COUNT
from veyra3d.objectives import GeometryObjective

class EvidencePolicyTests(unittest.TestCase):
    def test_preserves_468_contract(self):self.assertEqual(VERTEX_COUNT,468);self.assertEqual(len(TRIANGLE_INDICES),852)
    def test_unverified_makehuman_not_promoted_by_excluding_disputed_points(self):
        mask=eligible_mask('makehuman_candidate','mh','canonical');self.assertEqual(int(mask.sum()),0);self.assertFalse(training_readiness(mask)['may_create_manifest'])
    def test_canonical_exact_hash_is_required(self):
        with self.assertRaises(ValueError):eligible_mask('canonical_reference','foreign','canonical')
        mask=eligible_mask('canonical_reference','canonical','canonical');self.assertEqual(coverage(mask,TRIANGLE_INDICES)['vertices'],468)
    def test_other_index_spaces_do_not_transfer_implicitly(self):
        with self.assertRaises(ValueError):eligible_mask('ict_light','ict','canonical')
    def test_all_regions_have_distinct_evidence_requirements(self):
        self.assertEqual(set(REGION_REQUIREMENTS),{'lip_upper','lip_lower','lip_corners','forehead','temples','jaw','chin'})
    def test_empty_mask_excludes_all_geometry_terms(self):
        self.assertEqual(coverage(np.zeros(468,bool),TRIANGLE_INDICES),{'vertices':0,'edges':0,'triangles':0})
    def test_partial_edges_and_triangles_require_every_endpoint(self):
        mask=np.zeros(468,bool);mask[:2]=True;tri=np.array([[0,1,2],[1,2,3]])
        self.assertEqual(coverage(mask,tri),{'vertices':2,'edges':1,'triangles':0})
    def test_coordinates_center_complete_known_geometry_and_preserve_isotropic_units(self):
        points=np.arange(468*3,dtype=float).reshape(468,3)/1000;camera=np.eye(4);camera[:3,3]=[1,2,3];actual=centered_camera_geometry(points,camera,.23)
        np.testing.assert_allclose(actual,(points-points.mean(0))/.23,atol=1e-13);np.testing.assert_allclose(actual.mean(0),0,atol=1e-13)
    def test_partial_geometry_cannot_define_centroid(self):
        with self.assertRaises(ValueError):centered_camera_geometry(np.zeros((412,3)),np.eye(4),.23)
    def test_untrusted_target_values_do_not_change_loss_or_gradients(self):
        # Invented numerical fixture; this is not a set of approved facial landmarks.
        torch.manual_seed(8);target=torch.randn(1,468,3);mask=torch.ones(1,468,dtype=torch.bool);mask[:,[0,17,61,291]]=False;initial=torch.randn_like(target);objective=GeometryObjective()
        prediction=initial.clone().requires_grad_();loss1,stats1=objective(prediction,target,mask);gradient1,=torch.autograd.grad(loss1,prediction)
        corrupted=target.clone();corrupted[~mask]=float('nan');prediction2=initial.clone().requires_grad_();loss2,stats2=objective(prediction2,corrupted,mask);gradient2,=torch.autograd.grad(loss2,prediction2)
        torch.testing.assert_close(loss1,loss2,rtol=0,atol=0);torch.testing.assert_close(gradient1,gradient2,rtol=0,atol=0);self.assertEqual(int(torch.count_nonzero(gradient1[~mask])),0)
        self.assertEqual(int(stats1['valid_vertices']),464);self.assertEqual(int(stats2['normal_count']),int(stats1['normal_count']))
if __name__=='__main__':unittest.main()

"""Geometry regression checks with invented data; no renders/network/training."""
import copy
import unittest
import numpy as np
from scripts.correspondence_math import barycentric,fit_similarity,radial_warp,evaluate_anchors,deformation_metrics,parse_obj


class CorrespondenceTests(unittest.TestCase):
    def test_similarity_recovers_known_rotation_translation_and_scale(self):
        points=np.array([[0,0,0],[1,0,0],[0,2,0],[0,0,3],[1,2,3]],float)
        rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        expected=points@rotation*2+[4,5,6]
        scale,r,t=fit_similarity(points,expected)
        np.testing.assert_allclose(scale*points@r+t,expected,atol=1e-12)
        self.assertGreater(np.linalg.det(r),0)

    def test_radial_warp_interpolates_controls_and_affine_translation(self):
        controls=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,1]],float)
        points=np.vstack((controls,[[.2,.3,.4]]))
        targets=controls+[.1,.2,.3]
        result=radial_warp(points,controls,targets)
        np.testing.assert_allclose(result,points+[.1,.2,.3],atol=1e-12)

    def test_barycentric_anchor_follows_deformed_support_surface(self):
        vertices=np.array([[0,0,0],[2,0,0],[0,2,0]],float)
        weights=barycentric([.5,1,0],vertices)
        np.testing.assert_allclose(weights,[.25,.25,.5])
        slots=[{'veyra_index':i,'source_triangle':0,'barycentric':weights.tolist()} for i in range(468)]
        deformed=vertices.copy();deformed[2,2]=3
        result=evaluate_anchors(deformed,np.array([[0,1,2]]),slots)
        np.testing.assert_allclose(result,np.tile([.5,1,1.5],(468,1)))

    def test_bad_order_weights_and_triangle_ids_are_rejected(self):
        vertices=np.array([[0,0,0],[1,0,0],[0,1,0]],float);triangles=np.array([[0,1,2]])
        slots=[{'veyra_index':i,'source_triangle':0,'barycentric':[.2,.3,.5]} for i in range(468)]
        for change in ({'veyra_index':4},{'source_triangle':-1},{'source_triangle':1},{'barycentric':[-1,1,1]},{'barycentric':[0,0,float('nan')]}):
            with self.subTest(change=change):
                bad=copy.deepcopy(slots);bad[0].update(change)
                with self.assertRaises(ValueError):evaluate_anchors(vertices,triangles,bad)

    def test_degenerate_and_outside_barycentric_points_rejected(self):
        with self.assertRaisesRegex(ValueError,'Degenerate'):barycentric([0,0,0],np.zeros((3,3)))
        with self.assertRaisesRegex(ValueError,'outside'):barycentric([2,2,0],np.array([[0,0,0],[1,0,0],[0,1,0]]))

    def test_flip_and_collapse_are_detected(self):
        points=np.array([[0,0,0],[1,0,0],[0,1,0]],float);triangles=np.array([[0,1,2]])
        reversed_points=points[[0,2,1]]
        self.assertEqual(deformation_metrics(points,reversed_points,triangles)['normal_flip_triangles'],1)
        collapsed=points.copy();collapsed[1]=collapsed[0]
        self.assertGreater(deformation_metrics(points,collapsed,triangles)['collapsed_or_stretched_edges'],0)

    def test_obj_parser_preserves_vertex_order_and_face_indices(self):
        text='\n'.join(f'v {i} 0 1' for i in range(468))+'\nf 3/1 1/2 2/3\n'
        vertices,triangles=parse_obj(text)
        np.testing.assert_array_equal(vertices[:,0],np.arange(468))
        np.testing.assert_array_equal(triangles,[[2,0,1]])
        with self.assertRaises(ValueError):parse_obj('v 0 0 0\nf 1 1 1')


class AnatomicalConstraintTests(unittest.TestCase):
    def test_quad_loop_traces_existing_edges_on_periodic_mesh(self):
        from scripts.anatomical_constraints import trace_quad_loop
        nx,ny=6,5
        index=lambda x,y:(y%ny)*nx+(x%nx)
        quads=[[index(x,y),index(x+1,y),index(x+1,y+1),index(x,y+1)] for y in range(ny) for x in range(nx)]
        self.assertEqual(trace_quad_loop(quads,0,1),list(range(nx)))
        with self.assertRaises(ValueError):trace_quad_loop([[0,1,2,3]],0,1)

    def test_piecewise_arclength_sampling_stays_on_curve_segment(self):
        from scripts.anatomical_constraints import segment_sample
        points=np.array([[0,0,0],[1,0,0],[1,2,0]],float)
        i,w=segment_sample(points,.5)
        self.assertEqual(i,1)
        np.testing.assert_allclose(w@points[i:i+2],[1,.5,0])
        self.assertGreaterEqual(w.min(),0)
        self.assertAlmostEqual(w.sum(),1)

    def test_harmonic_warp_preserves_fixed_boundaries_and_finite_interior(self):
        from scripts.anatomical_constraints import harmonic_warp
        points=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[.5,.5,0]],float)
        triangles=np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
        target=points[:4]+[0,0,.2]
        result=harmonic_warp(points,triangles,[0,1,2,3],target)
        np.testing.assert_allclose(result[:4],target,atol=1e-12)
        self.assertTrue(np.isfinite(result).all())
        self.assertGreater(result[4,2],.19)
        with self.assertRaisesRegex(ValueError,'Repeated'):
            harmonic_warp(points,triangles,[0,0],target[:2])


class AnatomicalPatchTests(unittest.TestCase):
    def test_cavity_flood_does_not_cross_closed_rim(self):
        from scripts.anatomical_constraints import enclosed_quad_patch
        # Closed cube: the ring separates its top quad from the other five.
        quads=[[0,1,2,3],[4,7,6,5],[0,4,5,1],[1,5,6,2],[2,6,7,3],[3,7,4,0]]
        patch=enclosed_quad_patch(quads,[0,1,2,3],4)
        self.assertEqual(set(patch),set(range(1,6)))
        self.assertNotIn(0,patch)

    def test_parameter_orientation_keeps_hard_controls_fixed(self):
        from scripts.anatomical_constraints import preserve_parameter_orientation
        reference=np.array([[0,0,0],[1,0,0],[1,0,1],[0,0,1],[.5,0,.5]],float)
        triangles=np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
        folded=reference.copy();folded[4]=[.5,0,-.2]
        result,report=preserve_parameter_orientation(reference,folded,triangles,[0,1,2,3])
        np.testing.assert_array_equal(result[:4],reference[:4])
        self.assertGreater(result[4,2],0)
        self.assertEqual(report['unresolved_parameter_triangles'],[])

    def test_body_quad_parser_rejects_changed_source_topology(self):
        from scripts.anatomical_constraints import body_quads
        text='g body\nf 1/1 2/2 3/3 4/4\ng helper\nf 1 2 3\n'
        self.assertEqual(body_quads(text),[[0,1,2,3]])
        with self.assertRaisesRegex(ValueError,'quad topology'):
            body_quads('g body\nf 1 2 3')

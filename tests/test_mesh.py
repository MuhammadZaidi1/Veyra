"""Connectivity and geometry checks without any captured face data."""

import ast
from collections import Counter
from importlib.metadata import distribution
import unittest

import numpy as np

from veyra.mesh import BOUNDARY_EDGES, SURFACE_EDGES, TRIANGLES, valid_face, validate_topology, vertex_normals
from veyra.render_common import rotate_points


class TopologyTests(unittest.TestCase):
    def test_full_face_topology_has_valid_unique_triangles_and_four_open_boundaries(self):
        self.assertEqual(TRIANGLES.shape, (852, 3))
        self.assertEqual(set(TRIANGLES.ravel()), set(range(468)))
        self.assertEqual(len(SURFACE_EDGES), 1322)
        self.assertEqual(len(BOUNDARY_EDGES), 88)
        self.assertFalse(TRIANGLES.flags.writeable)
        self.assertEqual(set(Counter(BOUNDARY_EDGES.ravel()).values()), {2})
        adjacency = {}
        for a, b in BOUNDARY_EDGES:
            adjacency.setdefault(int(a), set()).add(int(b))
            adjacency.setdefault(int(b), set()).add(int(a))
        remaining, loops = set(adjacency), 0
        while remaining:
            loops += 1
            pending = [remaining.pop()]
            while pending:
                for neighbor in adjacency[pending.pop()]:
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        pending.append(neighbor)
        self.assertEqual(loops, 4)  # Outer face, both eyes, and mouth remain open.

    def test_indices_exactly_match_pinned_mediapipe_ordered_triangle_cycles(self):
        source = distribution("mediapipe").locate_file("mediapipe/tasks/python/vision/face_landmarker.py")
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FaceLandmarksConnections")
        assignment = next(n for n in cls.body if isinstance(n, ast.AnnAssign) and n.target.id == "FACE_LANDMARKS_TESSELATION")
        edges = [tuple(ast.literal_eval(a) for a in n.args) for n in assignment.value.elts]
        faces = []
        for a, b, c in zip(edges[::3], edges[1::3], edges[2::3]):
            self.assertEqual((a[1], b[1], c[1]), (b[0], c[0], a[0]))
            faces.append((a[0], a[1], b[1]))
        np.testing.assert_array_equal(TRIANGLES, faces)

    def test_invalid_topology_rejected(self):
        for triangles in (
            [[0, 1, 468]], [[0, 0, 1]], [[0, 1, 2], [2, 1, 0]],
            [[0, 1, 2], [0, 1, 3]],  # Shared edge has inconsistent winding.
            [[0, 1, 2], [1, 0, 3], [0, 1, 4]],  # Three faces on one edge.
            [[-1, 0, 2]], [[0.0, 1.0, 2.0]], [[0, 1]], [],
        ):
            with self.subTest(triangles=triangles), self.assertRaises(ValueError):
                validate_topology(triangles)

    def test_face_validity_does_not_fill_partial_detections(self):
        self.assertTrue(valid_face(np.zeros((468, 3))))
        self.assertTrue(valid_face(np.zeros((478, 3))))
        for points in (None, [], np.zeros((467, 3)), np.zeros((479, 3)),
                       np.zeros((468, 2)), np.full((478, 3), np.nan), np.full((478, 3), np.inf)):
            self.assertFalse(valid_face(points))


class NormalTests(unittest.TestCase):
    def test_area_weighted_smooth_normals_on_planar_quad_and_rotated_surface(self):
        points = np.array(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)), dtype=np.float32)
        triangles = np.array(((0, 1, 2), (0, 2, 3)))
        normals = vertex_normals(points, triangles)
        np.testing.assert_allclose(normals, np.tile((0, 0, 1), (4, 1)))
        rotated = rotate_points(points, 35, 20)
        np.testing.assert_allclose(vertex_normals(rotated, triangles), rotate_points(normals, 35, 20), atol=1e-6)

    def test_collapsed_blink_triangles_and_unused_iris_vertices_stay_finite(self):
        points = np.zeros((478, 3), dtype=np.float32)
        normals = vertex_normals(points)
        self.assertTrue(np.isfinite(normals).all())
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1)
        self.assertLess(int(TRIANGLES.max()), 468)

    def test_invalid_mesh_vertices_are_rejected(self):
        for points, faces in ((np.full((3, 3), np.nan), [[0, 1, 2]]),
                              (np.zeros((3, 2)), [[0, 1, 2]]),
                              (np.zeros((3, 3)), [[0, 1, 3]])):
            with self.assertRaises(ValueError):
                vertex_normals(points, faces)

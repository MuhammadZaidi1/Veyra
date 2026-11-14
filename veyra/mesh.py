"""Validated, fixed MediaPipe surface connectivity and smooth vertex normals.

Only the 468 face vertices are triangulated. The ten iris landmarks remain
points/lines. Mouth, eye openings and the outer edge remain open; no back of
head, eyeballs, teeth, or otherwise unobserved anatomy is added.
"""

import numpy as np

from .topology_data import TRIANGLE_INDICES

FACE_VERTEX_COUNT = 468
TRIANGLES = np.asarray(TRIANGLE_INDICES, dtype=np.uint32)
TRIANGLES.setflags(write=False)


def validate_topology(triangles, vertex_count=FACE_VERTEX_COUNT):
    """Reject invalid indices, duplicates, nonmanifold or inconsistent winding."""
    faces = np.asarray(triangles)
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer)):
        raise ValueError("Triangles must be a nonempty integer F x 3 array")
    if faces.min() < 0 or faces.max() >= vertex_count:
        raise ValueError("Triangle vertex is outside the landmark array")
    canonical = np.sort(faces, axis=1)
    if np.any(np.diff(canonical, axis=1) == 0):
        raise ValueError("Triangle has repeated vertices")
    if len(np.unique(canonical, axis=0)) != len(faces):
        raise ValueError("Duplicate triangle")
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    undirected = np.sort(directed, axis=1)
    edges, inverse, counts = np.unique(undirected, axis=0, return_inverse=True, return_counts=True)
    if counts.max() > 2:
        raise ValueError("Nonmanifold edge")
    signs = np.where(directed[:, 0] < directed[:, 1], 1, -1)
    winding = np.bincount(inverse, weights=signs)
    if np.any(winding[counts == 2] != 0):
        raise ValueError("Inconsistent triangle winding")
    return edges.astype(np.uint32), edges[counts == 1].astype(np.uint32)


SURFACE_EDGES, BOUNDARY_EDGES = validate_topology(TRIANGLES)
SURFACE_EDGES.setflags(write=False)
BOUNDARY_EDGES.setflags(write=False)


def valid_face(points):
    """Validate model output without inferring missing vertices or depth."""
    if points is None:
        return False
    try:
        points = np.asarray(points)
        return (points.shape in ((468, 3), (478, 3)) and np.isfinite(points).all()
                and np.max(np.abs(points)) < 100)
    except (TypeError, ValueError):
        return False


def vertex_normals(points, triangles=TRIANGLES):
    """Area-weighted vertex normals, safely ignoring collapsed blink triangles."""
    points = np.asarray(points, dtype=np.float32)
    faces = np.asarray(triangles)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or faces.ndim != 2 or faces.shape[1] != 3
            or not np.issubdtype(faces.dtype, np.integer)
            or (faces.size and (faces.min() < 0 or faces.max() >= len(points)))):
        raise ValueError("Invalid mesh vertices or triangle indices")
    vertices = points[faces]
    face_normals = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    corners = faces.ravel()
    normals = np.empty_like(points)
    for axis in range(3):
        normals[:, axis] = np.bincount(corners, np.repeat(face_normals[:, axis], 3), minlength=len(points))
    lengths = np.linalg.norm(normals, axis=1)
    good = lengths > 1e-12
    normals[good] /= lengths[good, None]
    normals[~good] = (0, 0, 1)
    return normals

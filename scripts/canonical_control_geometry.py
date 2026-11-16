"""Exact reference-vertex transforms for the canonical-only research control.

No MakeHuman input, correspondence transfer, fitted labels, or deformations.
Blender can import this module with its bundled NumPy; PyTorch is not required.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "third_party" / "mediapipe-reference"
CANONICAL_SHA256 = "8bac80443397e113f41a8b565ea72c59390bc031d9defab289dba7bc0c54e618"
REVISION = "02ca022d573c004c463d8c61eb3f8ce17e95a7fc"
SCHEMA = "veyra-canonical-render-control-v1"
FRAME_WIDTH = 0.23
RESOLUTION = 128
CONDITIONS = [(split, yaw, energy)
              for split, yaws in (("train", (-12, 0, 12)), ("val", (-6,)), ("test", (6,)))
              for yaw in yaws for energy in (3, 5)]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_canonical(path):
    """Read the pinned OBJ without welding, sorting, remeshing or reindexing."""
    if sha256(path) != CANONICAL_SHA256:
        raise ValueError("Pinned canonical OBJ hash mismatch")
    vertices, triangles = [], []
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == "v":
            vertices.append([float(x) for x in fields[1:4]])
        elif fields and fields[0] == "f":
            if len(fields) != 4:
                raise ValueError("Expected triangular canonical faces")
            triangles.append([int(x.split("/")[0]) - 1 for x in fields[1:]])
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int32)
    if vertices.shape != (468, 3) or triangles.shape != (898, 3):
        raise ValueError("Unexpected pinned source geometry")
    return vertices, triangles


def verify_reference():
    provenance = json.loads((REFERENCE / "provenance.json").read_text())
    if provenance["revision"] != REVISION:
        raise ValueError("Pinned reference revision changed")
    for name, record in provenance["files"].items():
        if sha256(REFERENCE / name) != record["sha256"]:
            raise ValueError(f"Pinned reference changed: {name}")
    read_canonical(REFERENCE / "canonical_face_model.obj")
    return provenance


def source_to_world(vertices):
    """Canonical cm, x right/y up/z forward -> Blender m, x right/y back/z up."""
    matrix = np.eye(4)
    matrix[:3, :3] = [[.01, 0, 0], [0, 0, -.01], [0, .01, 0]]
    matrix[:3, 3] = -matrix[:3, :3] @ vertices.mean(axis=0)
    return matrix


def transform(vertices, matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Expected finite 4x4 matrix")
    return (np.c_[vertices, np.ones(len(vertices))] @ matrix.T)[:, :3]


def camera_geometry(vertices, canonical_to_world, camera_to_world, frame_width=FRAME_WIDTH,
                    resolution=RESOLUTION):
    """Return absolute camera XYZ, centered Veyra XYZ, and PNG pixel centers.

    Blender camera axes already match +x right, +y up, +z toward camera.
    Absolute depth is negative for visible points. Center all 468 points and
    divide ALL three axes by orthographic image width. Save the removed origin
    to make the pixel transform reversible. PNG (0,0) is the top-left pixel center.
    """
    if not np.isfinite(frame_width) or frame_width <= 0:
        raise ValueError("Expected positive finite frame width")
    world = transform(vertices, canonical_to_world)
    camera = transform(world, np.linalg.inv(np.asarray(camera_to_world, dtype=np.float64)))
    targets = (camera - camera.mean(axis=0)) / frame_width
    pixel_xy = np.c_[(camera[:, 0] / frame_width + .5) * resolution - .5,
                     (.5 - camera[:, 1] / frame_width) * resolution - .5]
    return camera, targets, pixel_xy

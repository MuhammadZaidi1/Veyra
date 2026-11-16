"""Canonical-control unit tests; opt-in Blender/CUDA integration uses local renders.

Set VEYRA_CANONICAL_CONTROL to a fresh, checked renderer output to exercise the
real PNGs and eight total CUDA optimizer steps (whole vs checkpoint/resume).
Unit fixtures are explicitly artificial, never a training dataset.
"""

import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest

import numpy as np
from PIL import Image, ImageDraw
import torch

from scripts.canonical_control import validate
from scripts.canonical_control_geometry import (
    CONDITIONS, FRAME_WIDTH, REFERENCE, RESOLUTION, SCHEMA,
    camera_geometry, read_canonical, sha256, source_to_world, transform, verify_reference,
)
from scripts.canonical_training import load_config, run
from veyra3d.checkpoint import load_checkpoint


def camera_at_yaw(yaw):
    theta = np.deg2rad(yaw)
    c, s = np.cos(theta), np.sin(theta)
    camera = np.eye(4)
    camera[:3, :3] = [[c, 0, s], [s, 0, -c], [0, 1, 0]]
    camera[:3, 3] = [s, -c, 0]
    return camera


def fixture(root):
    vertices, _ = read_canonical(REFERENCE / "canonical_face_model.obj")
    matrix = source_to_world(vertices)
    for name in ("canonical_face_model.obj", "LICENSE"):
        shutil.copyfile(REFERENCE / name, root / name)
    rows = []
    for index, (split, yaw, energy) in enumerate(CONDITIONS):
        camera = camera_at_yaw(yaw)
        xyz, target, pixels = camera_geometry(vertices, matrix, camera)
        # Unique non-blank random PNGs exercise integrity only; not render tests.
        rgb = np.random.default_rng(index).integers(0, 256, (128, 128, 3), dtype=np.uint8)
        Image.fromarray(rgb).save(root / f"{index}.png")
        np.savez(root / f"{index}.npz", vertices=target.astype(np.float32),
                 camera_vertices=xyz, vertex_ids=np.arange(468, dtype=np.int32),
                 valid=np.ones(468, dtype=bool), pixel_xy=pixels)
        rows.append({"split": split, "condition": {"yaw_degrees": yaw, "key_energy": energy},
                     "camera_to_world": camera.tolist(), "image": f"{index}.png", "geometry": f"{index}.npz",
                     "image_sha256": sha256(root / f"{index}.png"),
                     "geometry_sha256": sha256(root / f"{index}.npz"), "projection_error_px": 0.})
    indices = [1, 4, 33, 133, 263, 362, 61, 291, 0, 17, 10, 152]
    _, _, pixels = camera_geometry(vertices, matrix, camera, resolution=512)
    probe = Image.new("RGB", (512, 512))
    draw = ImageDraw.Draw(probe)
    for x, y in pixels[indices]:
        draw.ellipse((round(x) - 2, round(y) - 2, round(x) + 2, round(y) + 2), fill="cyan")
    probe.save(root / "probes.png")
    meta = {"schema": SCHEMA, "source": verify_reference(), "source_file": "canonical_face_model.obj",
            "canonical_to_world": matrix.tolist(), "frame_width_m": FRAME_WIDTH, "resolution": RESOLUTION,
            "rows": rows, "deformed": False, "synthetic_only": True,
            "identity_count": 1, "identity_separated": False,
            "probe": {"image": "probes.png", "indices": indices, "pixel_xy": pixels[indices].tolist(),
                      "camera_to_world": camera.tolist(), "image_sha256": sha256(root / "probes.png")}}
    write_manifest(root, meta)
    return meta


def write_manifest(root, meta):
    (root / "control.json").write_text(json.dumps(meta))


class GeometryTests(unittest.TestCase):
    def test_exact_source_order_rigid_transform_and_camera_signs(self):
        vertices, triangles = read_canonical(REFERENCE / "canonical_face_model.obj")
        self.assertEqual(triangles.shape, (898, 3))
        matrix = source_to_world(vertices)
        np.testing.assert_allclose(transform(transform(vertices, matrix), np.linalg.inv(matrix)),
                                   vertices, rtol=0, atol=1e-13)
        camera, target, pixels = camera_geometry(vertices, matrix, camera_at_yaw(0))
        np.testing.assert_allclose(target, (vertices - vertices.mean(0)) * .01 / FRAME_WIDTH,
                                   rtol=0, atol=2e-14)
        self.assertTrue((camera[:, 2] < 0).all())
        self.assertLess(pixels[10, 1], pixels[152, 1])  # forehead above chin in PNG
        self.assertGreater(target[1, 2], target[234, 2])  # nose toward camera

    def test_reference_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.obj"
            path.write_bytes((REFERENCE / "canonical_face_model.obj").read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "hash"):
                read_canonical(path)

    def test_tiny_configuration_preserves_468_architecture(self):
        _, config = load_config()
        self.assertTrue(config.deterministic)
        self.assertFalse(config.amp)
        self.assertLessEqual(config.epochs, 2)
        self.assertEqual((config.loss.edge, config.loss.normal), (0., 0.))


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="veyra-canonical-fixture-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.meta = fixture(self.root)

    def test_complete_control_and_all_vertex_labels(self):
        result = validate(self.root)
        self.assertEqual(result["splits"], {"train": 6, "val": 2, "test": 2})
        self.assertEqual(result["exact_ordered_vertices"], 468)
        self.assertFalse(result["identity_separated"])
        self.assertFalse(result["human_anatomical_validation"])

    def test_file_tampering_is_rejected(self):
        with (self.root / "0.png").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "hash"):
            validate(self.root)

    def test_split_leakage_and_missing_conditions_are_rejected(self):
        self.meta["rows"][6]["condition"] = self.meta["rows"][0]["condition"]
        write_manifest(self.root, self.meta)
        with self.assertRaisesRegex(ValueError, "condition"):
            validate(self.root)

    def test_changed_order_or_target_is_rejected_even_after_rehash(self):
        path = self.root / "0.npz"
        with np.load(path) as source:
            arrays = {key: source[key].copy() for key in source.files}
        arrays["vertices"][[13, 14]] = arrays["vertices"][[14, 13]]
        np.savez(path, **arrays)
        self.meta["rows"][0]["geometry_sha256"] = sha256(path)
        write_manifest(self.root, self.meta)
        with self.assertRaisesRegex(ValueError, "label or projection"):
            validate(self.root)

    def test_deformation_and_path_escape_are_rejected(self):
        self.meta["deformed"] = True
        write_manifest(self.root, self.meta)
        with self.assertRaisesRegex(ValueError, "undeformed"):
            validate(self.root)
        self.meta["deformed"] = False
        self.meta["rows"][0]["image"] = "../escape.png"
        write_manifest(self.root, self.meta)
        with self.assertRaisesRegex(ValueError, "escapes"):
            validate(self.root)

    def test_inverted_raster_coordinates_are_rejected(self):
        self.meta["probe"]["pixel_xy"][0][1] = 511 - self.meta["probe"]["pixel_xy"][0][1]
        write_manifest(self.root, self.meta)
        with self.assertRaisesRegex(ValueError, "probe coordinates"):
            validate(self.root)


CONTROL = os.environ.get("VEYRA_CANONICAL_CONTROL")


@unittest.skipUnless(CONTROL, "Set VEYRA_CANONICAL_CONTROL to opt into local Blender output checks")
class RenderIntegrationTests(unittest.TestCase):
    def test_real_blender_outputs_and_overlay_exist(self):
        root = Path(CONTROL)
        result = validate(root)
        self.assertLess(result["independent_blender_projection_max_error_px"], .001)
        self.assertLessEqual(result["independent_raster_max_error_px"], 2.)
        with Image.open(root / "condition-contact-sheet.png") as image:
            self.assertEqual(image.size, (2560, 1024))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_deterministic_cuda_and_exact_checkpoint_resume(self):
        with tempfile.TemporaryDirectory(prefix="veyra-canonical-cuda-") as temporary:
            root = Path(temporary)
            whole = run(CONTROL, root / "whole", epochs=2, device="cuda")
            run(CONTROL, root / "resumed", epochs=1, device="cuda")
            resumed = run(CONTROL, root / "resumed", epochs=2,
                          resume=root / "resumed/last.pt", device="cuda")
            a = load_checkpoint(root / "whole/last.pt")
            b = load_checkpoint(root / "resumed/last.pt")

            def exact(left, right):
                if isinstance(left, torch.Tensor):
                    self.assertTrue(torch.equal(left, right))
                elif isinstance(left, dict):
                    self.assertEqual(left.keys(), right.keys())
                    for key in left:
                        exact(left[key], right[key])
                elif isinstance(left, (list, tuple)):
                    self.assertEqual(len(left), len(right))
                    for x, y in zip(left, right):
                        exact(x, y)
                else:
                    self.assertEqual(left, right)

            for key in ("model", "optimizer", "scaler", "rng", "global_step", "epoch", "best_validation"):
                exact(a[key], b[key])
            self.assertEqual(whole["global_step"], 4)
            self.assertEqual(whole["history"][-1]["synthetic_validation"]["relative_vertex_error"],
                             resumed["history"][-1]["synthetic_validation"]["relative_vertex_error"])
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
            self.assertFalse(whole["test_evaluated"])
            # Save small, inspectable metrics; temporary verification weights are removed.
            report_path = Path(CONTROL) / "cuda-verification.json"
            if report_path.exists():
                report_path = Path(CONTROL) / f"cuda-verification-{time.time_ns()}.json"
            with report_path.open("x") as stream:
                stream.write(json.dumps({"purpose": "eight CUDA steps total, not a convergence study",
                    "bitwise_checkpoint_resume": True, "deterministic_warn_only": False,
                    "whole": whole, "resumed": resumed}, indent=2) + "\n")

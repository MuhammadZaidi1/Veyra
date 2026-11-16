"""Offline canonical rendering control: launch Blender, verify data, draw overlays.

This separate loader deliberately bypasses the real-dataset identity split API:
all splits here share ONE template and measure held-out rendering conditions only.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from scripts.canonical_control_geometry import (
    CANONICAL_SHA256, CONDITIONS, FRAME_WIDTH, RESOLUTION, ROOT, SCHEMA,
    camera_geometry, read_canonical, sha256, source_to_world, verify_reference,
)
from scripts.run_blender_prototype import locate_blender, verify_raster


def local_file(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("Control paths must be relative")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Control file missing or escapes root")
    return path


def validate(root):
    """Recompute every label from source vertices + transforms; fail closed."""
    root = Path(root).resolve()
    manifest = root / "control.json"
    meta = json.loads(manifest.read_text())
    if (meta.get("schema") != SCHEMA or meta.get("deformed") is not False
            or meta.get("synthetic_only") is not True or meta.get("identity_count") != 1
            or meta.get("identity_separated") is not False):
        raise ValueError("Not an undeformed, single-template canonical control")
    if meta["resolution"] != RESOLUTION or meta["frame_width_m"] != FRAME_WIDTH:
        raise ValueError("Unexpected orthographic image frame")
    if meta["source"] != verify_reference():
        raise ValueError("Reference provenance mismatch")
    vertices, triangles = read_canonical(local_file(root, meta["source_file"]))
    if sha256(local_file(root, "LICENSE")) != meta["source"]["files"]["LICENSE"]["sha256"]:
        raise ValueError("Reference license missing or changed")
    matrix = np.asarray(meta["canonical_to_world"], dtype=np.float64)
    if matrix.shape != (4, 4) or not np.allclose(matrix, source_to_world(vertices), rtol=0, atol=1e-14):
        raise ValueError("Canonical geometry transform changed")
    rows = meta["rows"]
    expected = set(CONDITIONS)
    seen, paths, image_hashes = set(), set(), set()
    digest = hashlib.sha256(manifest.read_bytes())
    maximum_label_error = maximum_pixel_error = maximum_blender_error = 0.
    for row in rows:
        condition = (row["split"], row["condition"]["yaw_degrees"], row["condition"]["key_energy"])
        if condition not in expected or condition in seen:
            raise ValueError("Invalid, duplicate or leaking split condition")
        seen.add(condition)
        camera = np.asarray(row["camera_to_world"], dtype=np.float64)
        if camera.shape != (4, 4) or not np.isfinite(camera).all():
            raise ValueError("Invalid camera matrix")
        if not np.allclose(camera[3], [0, 0, 0, 1], rtol=0, atol=1e-7):
            raise ValueError("Non-affine camera matrix")
        if not np.allclose(camera[:3, :3].T @ camera[:3, :3], np.eye(3), rtol=0, atol=1e-6):
            raise ValueError("Non-rigid camera matrix")
        if np.linalg.det(camera[:3, :3]) < 0:
            raise ValueError("Reflected camera frame")
        theta = np.deg2rad(condition[1])
        if not np.allclose(camera[:3, 3], [np.sin(theta), -np.cos(theta), 0], atol=1e-6, rtol=0):
            raise ValueError("Camera does not match recorded condition")
        camera_xyz, target, pixel_xy = camera_geometry(vertices, matrix, camera)
        if not (camera_xyz[:, 2] < 0).all() or not ((pixel_xy >= 0) & (pixel_xy < RESOLUTION)).all():
            raise ValueError("Geometry outside the camera frustum")
        for name in ("image", "geometry"):
            path = local_file(root, row[name])
            if path in paths:
                raise ValueError("Duplicate file across records/splits")
            paths.add(path)
            content_hash = sha256(path)
            if content_hash != row[name + "_sha256"]:
                raise ValueError(f"Content hash mismatch: {row[name]}")
            digest.update(content_hash.encode())
            if name == "image":
                if content_hash in image_hashes:
                    raise ValueError("Duplicate image content across conditions")
                image_hashes.add(content_hash)
                with Image.open(path) as image:
                    if image.size != (RESOLUTION, RESOLUTION) or image.mode != "RGB":
                        raise ValueError("Expected square RGB control image")
                    if np.asarray(image).std() < 1:
                        raise ValueError("Blank control image")
            else:
                with np.load(path, allow_pickle=False) as source:
                    if set(source.files) != {"vertices", "camera_vertices", "valid", "vertex_ids", "pixel_xy"}:
                        raise ValueError("Unexpected canonical geometry arrays")
                    labels = source["vertices"]
                    if labels.shape != (468, 3) or labels.dtype != np.float32 or not np.isfinite(labels).all():
                        raise ValueError("Expected finite float32 [468,3] labels")
                    if (source["valid"].dtype != np.bool_ or source["valid"].shape != (468,)
                            or not source["valid"].all()):
                        raise ValueError("Canonical source needs all 468 known template vertices")
                    if source["vertex_ids"].dtype != np.int32 or not np.array_equal(source["vertex_ids"], np.arange(468)):
                        raise ValueError("Canonical vertex ordering changed")
                    if source["camera_vertices"].shape != (468, 3) or source["pixel_xy"].shape != (468, 2):
                        raise ValueError("Invalid absolute coordinates/projections")
                    label_error = float(np.max(np.abs(labels - target)))
                    pixel_error = float(np.max(np.abs(source["pixel_xy"] - pixel_xy)))
                    if (label_error > 5e-8 or not np.isfinite(pixel_error) or pixel_error > 1e-8
                            or not np.allclose(source["camera_vertices"], camera_xyz, atol=1e-12, rtol=0)
                            or not np.allclose(labels.mean(0), 0, atol=1e-7, rtol=0)):
                        raise ValueError("Source-to-camera label or projection mismatch")
                    # Labels are centered, so add the saved source origin for image overlays.
                    restored = labels.astype(np.float64) * FRAME_WIDTH + camera_xyz.mean(0)
                    restored_xy = np.c_[(restored[:, 0] / FRAME_WIDTH + .5) * RESOLUTION - .5,
                                       (.5 - restored[:, 1] / FRAME_WIDTH) * RESOLUTION - .5]
                    if np.max(np.abs(restored_xy - pixel_xy)) > 1e-5:
                        raise ValueError("Centered labels do not reconstruct PNG locations")
                    maximum_label_error = max(maximum_label_error, label_error)
                    maximum_pixel_error = max(maximum_pixel_error, pixel_error)
        error = row["projection_error_px"]
        if not np.isfinite(error) or error < 0 or error > .001:
            raise ValueError("Blender projection check failed")
        maximum_blender_error = max(maximum_blender_error, error)
    if seen != expected:
        raise ValueError("Incomplete train/val/test rendering conditions")
    probe = meta["probe"]
    probe_path = local_file(root, probe["image"])
    if sha256(probe_path) != probe["image_sha256"]:
        raise ValueError("Raster probe hash mismatch")
    _, _, probe_pixels = camera_geometry(vertices, matrix, probe["camera_to_world"], resolution=512)
    expected_probe_pixels = probe_pixels[probe["indices"]]
    if len(probe["indices"]) != 12 or not np.allclose(expected_probe_pixels, probe["pixel_xy"], atol=1e-8, rtol=0):
        raise ValueError("Raster probe coordinates changed")
    with Image.open(probe_path) as image:
        errors = verify_raster(image, expected_probe_pixels)
    digest.update(probe["image_sha256"].encode())
    return {"status": "executable_canonical_control_only", "schema": SCHEMA,
            "signature": digest.hexdigest(), "frames": len(rows),
            "splits": dict(Counter(r["split"] for r in rows)),
            "source_sha256": CANONICAL_SHA256, "exact_ordered_vertices": 468,
            "render_triangles": len(triangles), "model_loss_triangles_unchanged": 852,
            "float32_label_max_abs_error": maximum_label_error,
            "saved_projection_max_error_px": maximum_pixel_error,
            "independent_blender_projection_max_error_px": maximum_blender_error,
            "independent_raster_max_error_px": max(errors),
            "deformations": 0, "makehuman_labels_used": 0, "identity_count": 1,
            "identity_separated": False, "human_anatomical_validation": False,
            "real_world_accuracy_established": False, "identity_accuracy_established": False}


def overlays(root):
    root = Path(root)
    meta = json.loads((root / "control.json").read_text())
    selected = {0, 1, 4, 10, 13, 14, 17, 33, 61, 133, 152, 234, 263, 291, 362, 454}
    contact = Image.new("RGB", (5 * 512, 2 * 512))
    for number, row in enumerate(meta["rows"]):
        with Image.open(root / row["image"]) as image:
            result = image.resize((512, 512)).convert("RGB")
        with np.load(root / row["geometry"], allow_pickle=False) as source:
            # Reproject actual float32 training targets, rather than drawing stored pixel_xy.
            camera = source["vertices"].astype(np.float64) * FRAME_WIDTH + source["camera_vertices"].mean(0)
        pixels = np.c_[(camera[:, 0] / FRAME_WIDTH + .5) * 512 - .5,
                       (.5 - camera[:, 1] / FRAME_WIDTH) * 512 - .5]
        draw = ImageDraw.Draw(result)
        for index, (x, y) in enumerate(pixels):
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(255, 190, 40))
            if index in selected:
                draw.text((x + 2, y - 6), str(index), fill="cyan", stroke_width=1, stroke_fill="black")
        draw.rectangle((0, 0, 511, 34), fill="black")
        draw.text((4, 3), f"{row['split']} | {row['condition']}", fill="white")
        draw.text((4, 18), "468 exact TEMPLATE vertices; not verified human anatomy", fill="white")
        result.save(root / (Path(row["image"]).stem + "-overlay.png"))
        contact.paste(result, ((number % 5) * 512, (number // 5) * 512))
    contact.save(root / "condition-contact-sheet.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("render", "check"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--blender")
    args = parser.parse_args()
    if args.action == "render":
        if args.output.exists():
            parser.error("Use a fresh output folder")
        command = [locate_blender(args.blender), "--background", "--factory-startup",
                   "--python-exit-code", "1", "--python", str(ROOT / "scripts" / "render_dataset.py"),
                   "--", "--output", str(args.output.resolve())]
        subprocess.run(command, cwd=ROOT, check=True, timeout=300)
    report = validate(args.output)
    if args.action == "render":
        overlays(args.output)
        (args.output / "integrity-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

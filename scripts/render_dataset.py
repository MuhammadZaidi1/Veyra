# SPDX-License-Identifier: GPL-3.0-or-later
"""Blender background entry: ten canonical control frames, never a full dataset.

Use `python -m scripts.canonical_control render --output ...` for checks/overlays.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector
import numpy as np

from scripts.canonical_control_geometry import (
    CONDITIONS, FRAME_WIDTH, REFERENCE, RESOLUTION, SCHEMA, camera_geometry,
    read_canonical, sha256, source_to_world, transform, verify_reference,
)


def material(name, color, emission=False):
    result = bpy.data.materials.new(name)
    result.use_nodes = True
    shader = result.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = .7
    if emission:
        shader.inputs["Emission Color"].default_value = (*color, 1)
        shader.inputs["Emission Strength"].default_value = 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    out = args.output.resolve()
    if out.exists():
        raise ValueError("Use a fresh output folder; existing results are preserved")
    provenance = verify_reference()
    vertices, triangles = read_canonical(REFERENCE / "canonical_face_model.obj")
    source_matrix = source_to_world(vertices)
    world = transform(vertices, source_matrix)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    mesh = bpy.data.meshes.new("Canonical_exact_468_original_898_faces")
    mesh.from_pydata(world.tolist(), [], triangles.tolist())
    mesh.update()
    face = bpy.data.objects.new("Canonical_reference", mesh)
    bpy.context.collection.objects.link(face)
    mesh.materials.append(material("Own_clay", (.5, .3, .2)))
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 8
    scene.cycles.seed = 42
    scene.cycles.use_denoising = False
    scene.render.resolution_x = scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "Standard"
    scene.world = bpy.data.worlds.new("Own_background")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (.04, .05, .07, 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = .4
    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    bpy.context.collection.objects.link(camera)
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = FRAME_WIDTH
    scene.camera = camera
    lights = []
    for name, location, energy in (("Key", (-.25, -.35, .35), 3), ("Fill", (.25, -.3, .15), 2)):
        light = bpy.data.objects.new(name, bpy.data.lights.new(name, "AREA"))
        bpy.context.collection.objects.link(light)
        light.location = location
        light.rotation_euler = (-light.location).to_track_quat("-Z", "Y").to_euler()
        light.data.energy = energy
        light.data.size = .35
        lights.append(light)

    out.mkdir(parents=True)
    (out / ".gitignore").write_text("*\n")
    for name in ("canonical_face_model.obj", "LICENSE", "README.md", "provenance.json"):
        shutil.copyfile(REFERENCE / name, out / name)
    (out / "SOURCE_NOTICE.txt").write_text(
        "MediaPipe canonical geometry, The MediaPipe Authors, Apache-2.0.\n"
        "Original OBJ bytes and ordering preserved; see LICENSE and provenance.json.\n"
        "Derived images/labels apply documented rigid transforms, centering and unit conversion.\n"
        "Procedural clay/lights/cameras are project-created. No MakeHuman inputs.\n"
        "One synthetic template; not verified human anatomy, identity data or production weights.\n")
    rows = []
    for split, yaw, energy in CONDITIONS:
        theta = np.deg2rad(yaw)
        camera.location = (float(np.sin(theta)), -float(np.cos(theta)), 0)
        camera.rotation_euler = (-camera.location).to_track_quat("-Z", "Y").to_euler()
        lights[0].data.energy = energy
        bpy.context.view_layer.update()
        camera_matrix = np.asarray(camera.matrix_world, dtype=np.float64)
        camera_xyz, target, pixel_xy = camera_geometry(vertices, source_matrix, camera_matrix)
        actual_world = np.asarray([tuple(face.matrix_world @ q.co) for q in mesh.vertices])
        uv = np.asarray([tuple(world_to_camera_view(scene, camera, Vector(p))) for p in actual_world])
        blender_xy = np.c_[uv[:, 0] * RESOLUTION - .5, (1 - uv[:, 1]) * RESOLUTION - .5]
        error = float(np.abs(blender_xy - pixel_xy).max())
        if error > .001 or not (camera_xyz[:, 2] < 0).all():
            raise ValueError("Independent Blender projection mismatch or vertices behind camera")
        key = f"{split}-yaw{yaw:+d}-key{energy}"
        scene.render.filepath = str(out / f"{key}.png")
        bpy.ops.render.render(write_still=True)
        np.savez(out / f"{key}.npz", vertices=target.astype(np.float32),
                 camera_vertices=camera_xyz, valid=np.ones(468, dtype=np.bool_),
                 vertex_ids=np.arange(468, dtype=np.int32), pixel_xy=pixel_xy)
        rows.append({"split": split, "image": f"{key}.png", "geometry": f"{key}.npz",
                     "condition": {"yaw_degrees": yaw, "key_energy": energy},
                     "camera_to_world": camera_matrix.tolist(), "projection_error_px": error,
                     "image_sha256": sha256(out / f"{key}.png"),
                     "geometry_sha256": sha256(out / f"{key}.npz")})

    # Independent raster probe at the last camera pose; not a training image.
    face.hide_render = True
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0
    scene.render.resolution_x = scene.render.resolution_y = 512
    marker = material("Probe_cyan", (0, 1, 1), emission=True)
    probe_indices = [1, 4, 33, 133, 263, 362, 61, 291, 0, 17, 10, 152]
    for index in probe_indices:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=.0016,
                                           location=world[index])
        bpy.context.object.data.materials.append(marker)
    scene.render.filepath = str(out / "coordinate-probes.png")
    bpy.ops.render.render(write_still=True)
    _, _, probe_pixels = camera_geometry(vertices, source_matrix, camera_matrix, resolution=512)
    meta = {"schema": SCHEMA, "source": provenance, "source_file": "canonical_face_model.obj",
            "canonical_to_world": source_matrix.tolist(), "frame_width_m": FRAME_WIDTH,
            "resolution": RESOLUTION, "rows": rows, "deformed": False,
            "identity_count": 1, "identity_separated": False, "synthetic_only": True,
            "split_scope": "held-out camera yaw on the SAME template; not held-out identities",
            "source_triangles": 898, "model_loss_triangles_unchanged": 852,
            "label_contract": "centered-crop-width-xyz-468-v1",
            "blender_version": bpy.app.version_string, "cycles_device": "CPU",
            "cycles_samples": 8, "seed": 42,
            "renderer_sha256": sha256(__file__),
            "probe": {"image": "coordinate-probes.png", "indices": probe_indices,
                      "pixel_xy": probe_pixels[probe_indices].tolist(),
                      "camera_to_world": camera_matrix.tolist(),
                      "image_sha256": sha256(out / "coordinate-probes.png")}}
    (out / "control.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")
    print("CANONICAL_CONTROL_RENDERED", json.dumps({"frames": len(rows), "output": str(out)}))


if __name__ == "__main__":
    main()

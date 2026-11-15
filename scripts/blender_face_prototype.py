# SPDX-License-Identifier: GPL-3.0-or-later
"""Run inside Blender only. One CC0 base face; no dataset or training labels."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import addon_utils
import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector
import numpy as np

LICENSE_URL = 'https://github.com/makehumancommunity/mpfb2/blob/master/LICENSE.md'


def find_mpfb():
    modules = [module for module in addon_utils.modules() if module.__name__.split('.')[-1] == 'mpfb']
    if len(modules) != 1:
        raise RuntimeError('MPFB missing or ambiguous. Install/enable MPFB in Blender Preferences > Get Extensions. See docs/veyra3d-blender-prototype.md')
    module = modules[0]
    if not addon_utils.check(module.__name__)[1]:
        addon_utils.enable(module.__name__, default_set=False, persistent=False)
    root = Path(module.__file__).parent
    asset = root / 'data/3dobjs/base.obj'
    if not asset.is_file() or 'explicitly released as CC0' not in asset.read_text()[:2000]:
        raise RuntimeError('Bundled base.obj missing or its explicit CC0 declaration was not found; refusing asset')
    return module, root, asset


def material(name, color, emission=False):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (*color, 1)
    shader.inputs['Roughness'].default_value = .65
    if emission:
        shader.inputs['Emission Color'].default_value = (*color, 1)
        shader.inputs['Emission Strength'].default_value = 1
    return mat


def point_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()


def sphere(name, center, radius, mat):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=radius, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    module, root, asset = find_mpfb()
    report = {'blender':bpy.app.version_string, 'mpfb_module':module.__name__,
              'mpfb_path':str(root), 'base_asset':str(asset),
              'base_sha256':hashlib.sha256(asset.read_bytes()).hexdigest(),
              'asset_license':'CC0-1.0',
              'mpfb_manifest':(root/'blender_manifest.toml').read_text(),
              'mesh_metadata_sha256':hashlib.sha256((root/'data/mesh_metadata/basemesh_vertex_groups.json').read_bytes()).hexdigest(), 'license_source':LICENSE_URL,
              'external_assets_used':False, 'anatomical_mapping_validated':False}
    if args.check_only:
        print('VEYRA_CHECK ' + json.dumps(report))
        return
    if args.output is None:
        parser.error('--output required unless --check-only')
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Output already exists; choose a fresh directory')
    # This process never opens/saves a user scene or preferences.
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    service = importlib.import_module(module.__name__ + '.services.objectservice').ObjectService
    source = service.load_base_mesh(scale_factor=.1)
    groups = service.get_base_mesh_vertex_group_definition()
    body_ids = set(groups['body'])
    # Copy only body polygons; strip helper cubes, clothes cages, etc. Preserve source IDs.
    faces = [tuple(poly.vertices) for poly in source.data.polygons if all(i in body_ids for i in poly.vertices)]
    used = sorted({i for face in faces for i in face})
    reindex = {old:new for new, old in enumerate(used)}
    coordinates = [source.matrix_world @ source.data.vertices[i].co for i in used]
    mesh = bpy.data.meshes.new('CC0_body_surface')
    mesh.from_pydata(coordinates, [], [tuple(reindex[i] for i in face) for face in faces])
    mesh.update()
    source_ids = mesh.attributes.new('makehuman_source_vertex_id', 'INT', 'POINT')
    source_ids.data.foreach_set('value', used)
    face = bpy.data.objects.new('CC0_MakeHuman_base', mesh)
    bpy.context.collection.objects.link(face)
    face['anatomical_mapping_status'] = 'UNMAPPED: no Veyra468 training labels'
    eye_centers = []
    for key in ('joint-l-eye', 'joint-r-eye'):
        pts = [source.matrix_world @ source.data.vertices[i].co for i in groups[key]]
        eye_centers.append(sum(pts, Vector()) / len(pts))
    bpy.data.objects.remove(source, do_unlink=True)
    skin = material('Original_procedural_clay', (.55, .30, .20))
    mesh.materials.append(skin)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    white = material('Original_eye_white', (.8,.83,.8))
    dark = material('Original_iris', (.025,.045,.035))
    eyes = []
    for number, center in enumerate(eye_centers):
        eyes.append(sphere(f'Procedural_eye_{number}', center, .0115, white))
        eyes.append(sphere(f'Procedural_iris_{number}', center + Vector((0,-.0105,0)), .0045, dark))
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    # CPU Cycles is a bounded portable reference; GPU rendering is a later optimization.
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 16
    scene.cycles.use_denoising = False
    scene.cycles.seed = 42
    scene.render.resolution_x = scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.view_settings.view_transform = 'Standard'
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new('Original_constant_world')
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs[0].default_value = (.035,.045,.065,1)
    scene.world.node_tree.nodes['Background'].inputs[1].default_value = .4
    cam_data = bpy.data.cameras.new('Inspection_camera')
    camera = bpy.data.objects.new('Inspection_camera', cam_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (0,-1.1,.725)
    point_at(camera, (0,0,.725))
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = .33
    cam_data.clip_start, cam_data.clip_end = .01, 10
    scene.camera = camera
    for name, location, energy, size in [('Key',(-.35,-.5,1.1),3.5,.5),('Fill',(.35,-.2,.8),1.2,.4)]:
        data = bpy.data.lights.new(name, 'AREA')
        data.energy, data.shape, data.size = energy, 'DISK', size
        light = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(light)
        light.location = location
        point_at(light, (0,0,.725))
    bpy.context.view_layer.update()
    # Evaluated geometry, with no topology-changing modifiers in this prototype.
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = face.evaluated_get(depsgraph)
    surface = evaluated.to_mesh()
    try:
        world = np.array([tuple(evaluated.matrix_world @ vertex.co) for vertex in surface.vertices])
        surface.calc_loop_triangles()
        triangles = np.array([tuple(t.vertices) for t in surface.loop_triangles], dtype=np.int32)
    finally:
        evaluated.to_mesh_clear()
    inv = camera.matrix_world.inverted()
    camera_xyz = np.array([tuple(inv @ Vector(p)) for p in world])
    # Front surface points for coordinate diagnostics, NOT anatomical landmark guesses.
    candidate_ids = []
    for index, p in enumerate(world):
        if not (-.085 < p[0] < .085 and .625 < p[2] < .82 and p[1] < -.06):
            continue
        origin = Vector((p[0], -1.1, p[2]))
        hit, location, normal, poly, obj, matrix = scene.ray_cast(depsgraph, origin, Vector((0,1,0)))
        if hit and obj.original == face and (location-Vector(p)).length < 0.00015:
            candidate_ids.append(index)
    if len(candidate_ids) < 12:
        raise RuntimeError('Too few visible diagnostic points; inspect framing')
    selected = np.array(candidate_ids)[np.linspace(0,len(candidate_ids)-1,12,dtype=int)]
    points = world[selected]
    cam_points = camera_xyz[selected]
    center = cam_points.mean(axis=0)
    normalized = (cam_points-center)/cam_data.ortho_scale
    # An independent Blender projection API is compared with the analytical transform.
    projected = np.array([tuple(world_to_camera_view(scene,camera,Vector(p)))[:2] for p in points])
    pixels = np.column_stack((projected[:,0]*512-.5, (1-projected[:,1])*512-.5))
    restored = normalized*cam_data.ortho_scale+center
    analytic = np.column_stack(((restored[:,0]/cam_data.ortho_scale+.5)*512-.5,
                                (.5-restored[:,1]/cam_data.ortho_scale)*512-.5))
    error = float(np.max(np.abs(pixels-analytic)))
    if error > .001 or np.any(pixels < 3) or np.any(pixels > 508):
        raise RuntimeError(f'Coordinate check failed: {error} pixels')
    output.mkdir(parents=True)
    scene.render.filepath = str(output/'face.png')
    bpy.ops.render.render(write_still=True)
    # Independent raster check: render isolated emissive markers through the SAME camera.
    face.hide_render = True
    for eye in eyes:
        eye.hide_render = True
    scene.world.node_tree.nodes['Background'].inputs[0].default_value = (0,0,0,1)
    scene.world.node_tree.nodes['Background'].inputs[1].default_value = 0
    markers=[]
    cyan=material('Diagnostic_cyan_emission',(0,1,1),emission=True)
    for number,p in enumerate(points):
        markers.append(sphere(f'DIAGNOSTIC_NOT_LANDMARK_{number}',p,.0016,cyan))
    scene.render.filepath = str(output/'coordinate-probes.png')
    bpy.ops.render.render(write_still=True)
    for marker in markers:
        bpy.data.objects.remove(marker,do_unlink=True)
    face.hide_render=False
    for eye in eyes:
        eye.hide_render=False
    scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.035,.045,.065,1)
    scene.world.node_tree.nodes['Background'].inputs[1].default_value=.4
    scene.render.filepath=str(output/'face.png')
    bpy.ops.wm.save_as_mainfile(filepath=str(output/'inspection.blend'))
    # Intentionally not a trainable vertices/valid NPZ or a dataset manifest.
    np.savez(output/'surface-review.npz', world_xyz=world, source_vertex_ids=np.array(used), triangles=triangles)
    mapping = {'schema':'veyra468-surface-anchors-v1', 'status':'unmapped',
        'asset_sha256':report['base_sha256'], 'topology_sha256':hashlib.sha256(triangles.tobytes()).hexdigest(),
        'reviewer':None, 'slots':[{'veyra_index':i,'source_triangle':None,'barycentric':None} for i in range(468)],
        'warning':'All slots intentionally empty. No anatomical correspondence validated. No training labels exported.'}
    (output/'mapping-draft.json').write_text(json.dumps(mapping,indent=2)+'\n')
    report.update({'status':'rendered_awaiting_coordinate_raster_check', 'training_ready':False,
        'renderer':'Cycles CPU, 16 samples (bounded single-face prototype)',
        'camera_matrix_world':[list(row) for row in camera.matrix_world],
        'ortho_width':cam_data.ortho_scale,'resolution':[512,512],
        'diagnostic_source_vertex_ids':np.array(used)[selected].tolist(),
        'diagnostic_world_xyz':points.tolist(), 'diagnostic_camera_center':center.tolist(),
        'diagnostic_centered_xyz':normalized.tolist(),'diagnostic_pixel_xy':pixels.tolist(),
        'projection_max_abs_error_px':error,'surface_vertex_count':len(world),
        'warning':'12 diagnostic mesh vertices are NOT Veyra468 anatomical labels.'})
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('VEYRA_RENDER '+str(output))


if __name__ == '__main__':
    main()

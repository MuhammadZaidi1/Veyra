# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only canonical/source 3D inspection renders; never changes the map."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bpy
import numpy as np
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
from blender_face_prototype import material,point_at


def curve(name,points,mat,radius=.00022):
    data=bpy.data.curves.new(name,'CURVE');data.dimensions='3D';data.bevel_depth=radius;data.bevel_resolution=1
    spline=data.splines.new('POLY');spline.points.add(len(points)-1)
    for item,p in zip(spline.points,points):item.co=(*p,1)
    obj=bpy.data.objects.new(name,data);bpy.context.collection.objects.link(obj);data.materials.append(mat)
    return obj


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--prototype',type=Path,required=True)
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:]);out=args.audit.resolve()
    if (out/'render-views.json').exists():raise ValueError('Audit render outputs already exist')
    g=np.load(out/'audit-geometry.npz',allow_pickle=False);regions=json.loads((out/'display-regions.json').read_text())
    bpy.ops.wm.open_mainfile(filepath=str((args.prototype/'inspection.blend').resolve()))
    scene=bpy.context.scene;source=bpy.data.objects['CC0_MakeHuman_base'];source_objects=[source]+[o for o in scene.objects if o.name.startswith('Procedural_')]
    actual=np.array([tuple(source.matrix_world@v.co) for v in source.data.vertices])
    if not np.allclose(actual,g['source_vertices'],atol=1e-8):raise ValueError('Source scene differs from audit')
    gray=material('Audit_neutral_clay',(.45,.43,.40));source.data.materials.clear();source.data.materials.append(gray)
    mesh=bpy.data.meshes.new('Canonical_unwarped_face_patch');mesh.from_pydata(g['canonical_aligned'].tolist(),[],g['canonical_triangles'].tolist());mesh.update()
    canonical=bpy.data.objects.new('Canonical_similarity_only',mesh);bpy.context.collection.objects.link(canonical)
    mesh.materials.append(gray);cap_mat=material('Canonical_mouth_cap_not_MH_skin',(.9,.38,.08));mesh.materials.append(cap_mat)
    for face in mesh.polygons:
        face.use_smooth=True
        if face.index in regions['mouth_caps']:face.material_index=1
    colors={'lips':material('Audit_lip_contours',(.05,.8,.7),True),'contour':material('Audit_face_oval',(.15,.5,1),True)}
    displays={}
    for name,key in [('canonical','canonical_aligned'),('makehuman','mapped_points')]:
        objects=[];points=g[key]
        for region in ['lips','contour']:
            for a,b in regions['region_edges'][region]:objects.append(curve(f'{name}_{region}_{a}_{b}',[points[a],points[b]],colors[region]))
        displays[name]=objects
    scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=16;scene.cycles.use_denoising=False
    scene.render.resolution_x=scene.render.resolution_y=1024;scene.render.resolution_percentage=100
    scene.view_settings.view_transform='Standard';scene.camera.data.ortho_scale=.33
    target=(0,-.04,.725);views=[]
    for view,position in {'front':(0,-1.1,.725),'oblique':(.9,-.56,.725),'right-profile':(1.1,-.04,.725),'left-profile':(-1.1,-.04,.725)}.items():
        scene.camera.location=position;point_at(scene.camera,target);bpy.context.view_layer.update()
        for model,key in [('canonical','canonical_aligned'),('makehuman','mapped_points')]:
            canonical.hide_render=model!='canonical'
            for o in source_objects:o.hide_render=model!='makehuman'
            for name,objects in displays.items():
                for o in objects:o.hide_render=name!=model
            filename=f'{model}-{view}.png';scene.render.filepath=str(out/filename);bpy.ops.render.render(write_still=True)
            uv=np.array([tuple(world_to_camera_view(scene,scene.camera,Vector(v))) for v in g[key]])
            views.append({'model':model,'view':view,'file':filename,'pixel_xy':np.column_stack((uv[:,0]*1024-.5,(1-uv[:,1])*1024-.5)).tolist(),
                'camera_matrix_world':[list(row) for row in scene.camera.matrix_world],'resolution':1024})
    (out/'render-views.json').write_text(json.dumps({'status':'unvalidated','canonical_display':'same similarity transform, no residual warp',
        'orange_faces':'canonical mouth-cap triangles, not source skin','lines':'straight 3D landmark chords, not continuous source-surface curves','views':views},indent=2)+'\n')


if __name__=='__main__':main()

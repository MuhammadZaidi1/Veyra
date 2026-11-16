# SPDX-License-Identifier: GPL-3.0-or-later
"""Six canonical-468 projection/glasses-frame controls. No dataset or training labels exported."""
import argparse,json,sys,math
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1])]
import bpy,numpy as np
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
from scripts.correspondence_math import parse_obj
from scripts.correspondence_gate import structural_checks
from scripts.makehuman_semantics import sha
from scripts.mediapipe_evidence_policy import centered_camera_geometry
from veyra.topology_data import TRIANGLE_INDICES


def material(name,color,emission=False):
    m=bpy.data.materials.new(name);m.use_nodes=True;s=m.node_tree.nodes.get('Principled BSDF');s.inputs['Base Color'].default_value=(*color,1);s.inputs['Roughness'].default_value=.7
    if emission:s.inputs['Emission Color'].default_value=(*color,1);s.inputs['Emission Strength'].default_value=1
    return m


def curve(name,points,mat,parent):
    c=bpy.data.curves.new(name,'CURVE');c.dimensions='3D';c.bevel_depth=.0012;c.bevel_resolution=2;line=c.splines.new('POLY');line.points.add(len(points)-1)
    for p,xyz in zip(line.points,points):p.co=(*xyz,1)
    o=bpy.data.objects.new(name,c);bpy.context.collection.objects.link(o);o.parent=parent;c.materials.append(mat);return o


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);out=a.output.resolve()
    if out.exists():raise ValueError('Use a fresh diagnostic folder')
    provenance=json.loads((a.reference/'provenance.json').read_text())
    for name,entry in provenance['files'].items():
        if sha(a.reference/name)!=entry['sha256']:raise ValueError('Pinned reference changed')
    canonical,_=parse_obj((a.reference/'canonical_face_model.obj').read_text());v=.01*canonical[:,[0,2,1]]*[1,-1,1];v-=v.mean(0);triangles=np.array(TRIANGLE_INDICES)
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False);root=bpy.data.objects.new('Canonical_pose',None);bpy.context.collection.objects.link(root)
    mesh=bpy.data.meshes.new('Exact_order_468_existing_852_faces');mesh.from_pydata(v.tolist(),[],triangles.tolist());mesh.update();face=bpy.data.objects.new('Canonical_reference_468',mesh);bpy.context.collection.objects.link(face);face.parent=root;mesh.materials.append(material('Own_clay',(.5,.35,.27)))
    for polygon in mesh.polygons:polygon.use_smooth=True
    frames=[];frame_mat=material('Own_frame_material',(.02,.04,.08));centers=[];sizes=[]
    for corners in [[33,133],[362,263]]:
        points=v[corners];center=points.mean(0);half_width=abs(points[0,0]-points[1,0])*.62;half_height=.011;front_y=float(points[:,1].min()-.006);centers.append((center[0],front_y,center[2]));sizes.append(half_width)
        theta=np.linspace(0,2*np.pi,65);ring=np.c_[center[0]+half_width*np.cos(theta),np.full(len(theta),front_y),center[2]+half_height*np.sin(theta)];frames.append(curve('Own_glasses_frame_'+str(len(frames)),ring,frame_mat,root))
    bridge_y=min(x[1] for x in centers)-.003;bridge_z=max(x[2] for x in centers)+.003
    frames.append(curve('Own_glasses_bridge',[(centers[0][0]+sizes[0],centers[0][1],centers[0][2]),(0,bridge_y,bridge_z),(centers[1][0]-sizes[1],centers[1][1],centers[1][2])],frame_mat,root))
    for center,width in zip(centers,sizes):
        sign=1 if center[0]>0 else -1;frames.append(curve('Own_glasses_arm_'+str(sign),[(center[0]+sign*width,center[1],center[2]),(center[0]+sign*(width+.014),center[1]+.045,center[2]-.004)],frame_mat,root))
    scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=16;scene.cycles.use_denoising=False;scene.render.resolution_x=scene.render.resolution_y=1024;scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG';scene.view_settings.view_transform='Standard'
    scene.world=bpy.data.worlds.new('Own_background');scene.world.use_nodes=True;scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.04,.05,.07,1);scene.world.node_tree.nodes['Background'].inputs[1].default_value=.4
    camera_data=bpy.data.cameras.new('Diagnostic_camera');camera=bpy.data.objects.new('Diagnostic_camera',camera_data);bpy.context.collection.objects.link(camera);camera.location=(0,-1,0);camera.rotation_euler=(Vector((0,0,0))-camera.location).to_track_quat('-Z','Y').to_euler();camera_data.type='ORTHO';camera_data.ortho_scale=.23;scene.camera=camera
    for name,loc,energy in [('Key',(-.25,-.35,.35),5),('Fill',(.25,-.3,.15),2)]:
        data=bpy.data.lights.new(name,'AREA');data.energy=energy;data.shape='DISK';data.size=.35;o=bpy.data.objects.new(name,data);bpy.context.collection.objects.link(o);o.location=loc;o.rotation_euler=(-o.location).to_track_quat('-Z','Y').to_euler()
    out.mkdir(parents=True);records=[];pairs={};max_projection=0
    for pose,yaw in [('left',-20),('front',0),('right',20)]:
        root.rotation_euler[2]=math.radians(yaw);bpy.context.view_layer.update();world=np.array([tuple(face.matrix_world@q.co) for q in mesh.vertices]);check=structural_checks(v,world,triangles)
        if not check['passed']:raise ValueError('Canonical rigid-pose structural failure')
        targets=centered_camera_geometry(world,np.array(camera.matrix_world),camera_data.ortho_scale)
        for glasses in [False,True]:
            for obj in frames:obj.hide_render=not glasses;obj.hide_set(not glasses)
            bpy.context.view_layer.update();updated=np.array([tuple(face.matrix_world@q.co) for q in mesh.vertices]);np.testing.assert_array_equal(world,updated)
            uv=np.array([tuple(world_to_camera_view(scene,camera,Vector(q))) for q in world]);cam=np.c_[world,np.ones(468)]@np.linalg.inv(np.array(camera.matrix_world)).T;analytic=np.c_[cam[:,0]/.23+.5,.5+cam[:,1]/.23];error=float(abs(analytic-uv[:,:2]).max()*1024);max_projection=max(max_projection,error)
            if error>.001:raise ValueError('Projection mismatch')
            name=pose+('-frames' if glasses else '-clear');scene.render.filepath=str(out/(name+'.png'));bpy.ops.render.render(write_still=True)
            depsgraph=bpy.context.evaluated_depsgraph_get();direction=-camera.matrix_world.col[2].xyz;occluded=[];self_hidden=[];unknown=[]
            for index,point in enumerate(world):
                hit,location,normal,polygon,obj,matrix=scene.ray_cast(depsgraph,Vector(point)-direction*2,direction)
                if not hit:unknown.append(index)
                elif obj.original!=face:occluded.append(index)
                elif (location-Vector(point)).length>.0002:self_hidden.append(index)
            xy=np.c_[uv[:,0]*1024-.5,(1-uv[:,1])*1024-.5]
            records.append({'name':name,'yaw_degrees':yaw,'glasses_frames':glasses,'pixel_xy':xy.tolist(),'known_template_indices':468,'camera_matrix':[list(row) for row in camera.matrix_world],'pose_matrix':[list(row) for row in face.matrix_world],'orthographic_width':.23,'projection_error_px':error,'target_centroid_max_abs':float(abs(targets.mean(0)).max()),'frame_occluded_indices':occluded,'self_hidden_indices':self_hidden,'visibility_unknown_indices':unknown,'visibility_is_separate_from_label_evidence':True,'structural_check':check})
            if not glasses:pairs[pose]=targets.copy()
            else:np.testing.assert_array_equal(targets,pairs[pose])
    # Independent raster probe: emissive points through the same camera, face/frames hidden.
    root.rotation_euler[2]=0;face.hide_render=True
    for obj in frames:obj.hide_render=True
    scene.world.node_tree.nodes['Background'].inputs[1].default_value=0;scene.render.resolution_x=scene.render.resolution_y=512;bpy.context.view_layer.update();marker_mat=material('Probe_cyan',(0,1,1),True);probe_indices=[1,4,33,133,263,362,61,291,0,17,10,152];probe_world=[]
    for i in probe_indices:
        point=face.matrix_world@Vector(v[i]);probe_world.append(point);bpy.ops.mesh.primitive_uv_sphere_add(segments=16,ring_count=8,radius=.0016,location=point);bpy.context.object.data.materials.append(marker_mat)
    scene.render.filepath=str(out/'coordinate-probes.png');bpy.ops.render.render(write_still=True);uv=np.array([tuple(world_to_camera_view(scene,camera,q)) for q in probe_world]);pixel=np.c_[uv[:,0]*512-.5,(1-uv[:,1])*512-.5]
    report={'status':'canonical_468_control_only_no_training','reference_sha256':provenance['files']['canonical_face_model.obj']['sha256'],'landmark_count':468,'render_triangle_count':len(triangles),'source_semantics':'Exact canonical vertex identity; not a MakeHuman transfer or measured human anatomy','canonical_geometry_changed':False,'poses':records,'paired_clear_frame_target_max_difference':0.,'projection_max_error_px':max_projection,'probe_pixel_xy':pixel.tolist(),'dataset_exported':False,'training_started':False,'identity_count':1,'identity_recognition_evaluation_possible':False,'glasses_scope':'Procedural frames only; no refractive lenses or real eyewear robustness claim','centering_scope':'Complete known canonical geometry, never partial/unverified MakeHuman anchors'}
    (out/'geometry-report.json').write_text(json.dumps(report,indent=2));print('CANONICAL_CONTROL',json.dumps({k:report[k] for k in ['landmark_count','render_triangle_count','paired_clear_frame_target_max_difference','projection_max_error_px','training_started']}))

if __name__=='__main__':main()

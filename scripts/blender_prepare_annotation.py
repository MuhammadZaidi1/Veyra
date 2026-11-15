# SPDX-License-Identifier: GPL-3.0-or-later
"""Prepare an isolated frozen-surface annotation project in Blender background mode."""
import argparse,json,sys,shutil,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bpy,numpy as np
from blender_face_prototype import point_at
from mathutils import Vector
import blender_annotation_ui as ui

def main():
 p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--mapping',type=Path);p.add_argument('--prototype',type=Path);a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);out=a.project.resolve();out.mkdir(parents=True,exist_ok=False)
 repo=Path(__file__).resolve().parents[1];base=a.mapping or repo/'runs/veyra3d/correspondence-anatomical-final/mapping-candidates.json';prototype=a.prototype or repo/'runs/veyra3d/blender-prototype-002'
 shutil.copy2(base,out/'base-map.json');shutil.copy2(prototype/'surface-review.npz',out/'surface.npz');m=json.loads(base.read_text());g=np.load(out/'surface.npz');pts=np.array([np.array(s['barycentric'])@g['world_xyz'][g['triangles'][s['source_triangle']]] for s in m['slots']])
 config={k:m[k] for k in ['asset_sha256','topology_sha256']};config.update(base_mapping_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),original_mapping=str(base),repo=str(repo),surface_sha256=hashlib.sha256((out/'surface.npz').read_bytes()).hexdigest(),status='unreviewed_annotation_project')
 (out/'project.json').write_text(json.dumps(config,indent=2));bpy.ops.wm.open_mainfile(filepath=str(prototype/'inspection.blend'));scene=bpy.context.scene;scene['veyra_annotation_root']=str(out)
 upper=[61,40,39,37,0,267,269,270,409,291,185];lower=[146,91,181,84,17,314,405,321,375];outer=set(upper+lower)
 # Explicit official outer-border IDs; classifications are review queues, not anatomical claims.
 outer={0,17,37,39,40,61,84,91,146,181,185,267,269,270,291,314,321,375,405,409}
 oval=set(m['regions']['contour']);assert len(outer)==20 and len(oval)==36
 forehead={10,109,67,103,54,338,297,332,284};temples={21,162,127,234,93,251,389,356,454,323};chin={152,148,176,377,400}
 for i in sorted(outer|oval):
  group=('lip_corners' if i in {61,291} else 'lip_upper' if i in upper else 'lip_lower') if i in outer else ('forehead' if i in forehead else 'temples' if i in temples else 'chin' if i in chin else 'jaw')
  obj=bpy.data.objects.new(f'MP_{i:03d}_{group}',None);scene.collection.objects.link(obj);obj.location=pts[i];obj.empty_display_type='SPHERE';obj.empty_display_size=.0007;obj.show_name=True;obj.show_in_front=True;obj.color=(1,.25,.1,1)
  obj['veyra_editable']=True;obj['veyra_index']=i;obj['annotation_group']=group;obj['source_triangle']=m['slots'][i]['source_triangle'];obj['barycentric']=m['slots'][i]['barycentric'];obj['decision']='unreviewed';obj['reviewer']='';obj['note']='Original candidate position; actual anatomical border must be inspected'
 source=bpy.data.objects['CC0_MakeHuman_base'];source.hide_select=True
 ui.register();(out/'annotations').mkdir();(out/'annotations/draft-0000.json').write_text(json.dumps(ui.snapshot(),indent=2))
 for area in bpy.context.screen.areas:
  if area.type=='VIEW_3D':
   area.spaces.active.region_3d.view_location=(0,-.04,.725);area.spaces.active.region_3d.view_distance=.4;area.spaces.active.overlay.show_extras=True;area.spaces.active.region_3d.view_rotation=(Vector((0,-.04,.725))-Vector((0,-1.1,.725))).to_track_quat('-Z','Y');area.spaces.active.region_3d.view_perspective='ORTHO'
 text=bpy.data.texts.new('START_HERE.py');text.write("import runpy\nrunpy.run_path("+repr(str(repo/'scripts/blender_annotation_ui.py'))+", run_name='__main__')\n")
 bpy.context.preferences.filepaths.file_preview_type='NONE'
 scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=8;scene.render.resolution_x=scene.render.resolution_y=1024;scene.render.resolution_percentage=100;scene.camera.data.type='ORTHO';scene.camera.data.ortho_scale=.33
 views=[]
 for name,pos in [('front',(0,-1.1,.725)),('profile',(1.1,-.04,.725)),('oblique',(.9,-.56,.725))]:
  scene.camera.location=pos;point_at(scene.camera,(0,-.04,.725));bpy.context.view_layer.update();scene.render.filepath=str(out/(name+'.png'));bpy.ops.render.render(write_still=True);views.append({'name':name,'matrix':[list(r) for r in scene.camera.matrix_world],'width':.33,'resolution':1024})
 (out/'views.json').write_text(json.dumps(views,indent=2));bpy.ops.wm.save_as_mainfile(filepath=str(out/'annotation.blend'))
 print('ANNOTATION_PROJECT_READY',out,'markers',len(ui.snapshot()['annotations']));ui.unregister()
if __name__=='__main__':main()

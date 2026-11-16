# SPDX-License-Identifier: GPL-3.0-or-later
"""Nine bounded native/retained/rejected geometry inspection renders, never a dataset."""
import argparse,json,sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1])]
import bpy,numpy as np
from mathutils import Matrix,Vector
from bpy_extras.object_utils import world_to_camera_view
from scripts.makehuman_semantics import CoreTargets,sha
from scripts.correspondence_math import evaluate_anchors

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['result','project','prototype']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);folder=a.result/'inspection-renders';folder.mkdir(exist_ok=False)
    report=json.loads((a.result/'report.json').read_text());g=np.load(a.project/'surface.npz');v,t,ids=g['world_xyz'],g['triangles'],g['source_vertex_ids'];base=json.loads((a.project/'base-map.json').read_text());candidates=json.loads((a.result/'tested-candidates.json').read_text());root=Path(json.loads((a.prototype/'report.json').read_text())['mpfb_path']);targets=CoreTargets(root,ids);cases={c['name']:c for c in targets.cases(v)};views={x['name']:x for x in json.loads((a.project/'views.json').read_text())}
    for rel,meta in report['targets'].items():
        if sha(root/'data/targets'/rel)!=meta['sha256']:raise ValueError('Target changed after evaluation')
    bpy.ops.wm.open_mainfile(filepath=str((a.prototype/'inspection.blend').resolve()));scene=bpy.context.scene;obj=bpy.data.objects['CC0_MakeHuman_base'];scene.cycles.samples=16;scene.render.resolution_x=scene.render.resolution_y=1024;scene.render.resolution_percentage=100;records=[]
    for name,view in [('base/neutral','front'),('base/neutral','profile'),('base/neutral','oblique'),('base/mouth_open','front'),('base/eye_close','front'),('narrow/open_pull','front'),('narrow/open_pull','profile'),('tall/open_pull','front'),('tall/open_pull','profile')]:
        varied=cases[name]['vertices'];obj.data.vertices.foreach_set('co',varied.astype(np.float32).ravel());obj.data.update();scene.camera.matrix_world=Matrix(views[view]['matrix']);bpy.context.view_layer.update();filename=name.replace('/','-')+'-'+view+'.png';scene.render.filepath=str((folder/filename).resolve());bpy.ops.render.render(write_still=True)
        record={'case':name,'view':view,'file':filename,'pixels':{},'source_pixel_xy':[],'note':'Unvalidated raw core deltas, procedural eyeballs unchanged'}
        for label,mapping in [('baseline',base),('rig_combined',candidates['rig_combined'])]:
            points=evaluate_anchors(varied,t,mapping['slots']);uv=np.array([tuple(world_to_camera_view(scene,scene.camera,Vector(q))) for q in points]);record['pixels'][label]=np.c_[uv[:,0]*1024-.5,(1-uv[:,1])*1024-.5].tolist()
        uv=np.array([tuple(world_to_camera_view(scene,scene.camera,Vector(q))) for q in varied]);record['source_pixel_xy']=np.c_[uv[:,0]*1024-.5,(1-uv[:,1])*1024-.5].tolist();records.append(record)
    # Serialize independent neutral-view checks for every tested family, including singletons.
    evidence=json.loads((a.result/'point-evidence.json').read_text());maps={name:{'slots':[dict(slot) for slot in base['slots']]} for name in report['evaluations']}
    editable=[row['veyra_index'] for row in evidence]
    for row in evidence:
        i=row['veyra_index']
        for entry in row['candidates']:
            maps[entry['family']]['slots'][i].update(source_triangle=entry['source_triangle'],barycentric=entry['barycentric'])
    camera_checks=[]
    for view in views.values():
        scene.camera.matrix_world=Matrix(view['matrix']);bpy.context.view_layer.update()
        for name,mapping in maps.items():
            points=evaluate_anchors(v,t,mapping['slots'])[editable];uv=np.array([tuple(world_to_camera_view(scene,scene.camera,Vector(q))) for q in points]);cam=np.c_[points,np.ones(len(points))]@np.linalg.inv(np.array(view['matrix'])).T;analytical=np.c_[cam[:,0]/view['width']+.5,.5+cam[:,1]/view['width']];error=float(abs(analytical-uv[:,:2]).max()*1024)
            if error>.001:raise ValueError('Independent candidate camera disagreement')
            camera_checks.append({'family':name,'view':view['name'],'max_error_px':error})
    (folder/'candidate-view-checks.json').write_text(json.dumps(camera_checks,indent=2))
    (folder/'renders.json').write_text(json.dumps(records,indent=2));print('STRICT_INSPECTION_RENDERS',len(records))
if __name__=='__main__':main()

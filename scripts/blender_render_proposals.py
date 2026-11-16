# SPDX-License-Identifier: GPL-3.0-or-later
"""Render nine bounded morph/expression inspection views; no dataset export."""
import argparse,gzip,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bpy,numpy as np
from mathutils import Matrix,Vector
from bpy_extras.object_utils import world_to_camera_view
from correspondence_math import evaluate_anchors

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--prototype',type=Path,required=True);a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);root=a.project.resolve();folder=root/'stability-renders';folder.mkdir(exist_ok=False)
    report=json.loads((root/'proposal-report.json').read_text());base=json.loads((root/'base-map.json').read_text());proposed=json.loads((root/'mapping-proposals.json').read_text());surface=np.load(root/'surface.npz');v,t,ids=surface['world_xyz'],surface['triangles'],surface['source_vertex_ids'];lookup={int(x):i for i,x in enumerate(ids)};asset_root=Path(json.loads((a.prototype/'report.json').read_text())['mpfb_path'])
    def target(relative):
        path=asset_root/'data/targets'/relative
        if hashlib.sha256(path.read_bytes()).hexdigest()!=report['targets'][relative]['sha256']:raise ValueError('Target changed')
        d=np.zeros_like(v)
        for line in gzip.decompress(path.read_bytes()).decode().splitlines():
            f=line.split()
            if len(f)==4 and not line.lstrip().startswith('#') and int(f[0]) in lookup:d[lookup[int(f[0])]]=.1*np.array(list(map(float,f[1:])))[[0,2,1]]*[1,-1,1]
        return d
    zero=np.zeros_like(v);identities={'base':zero,'round':.5*target('head/head-round.target.gz'),'narrow':.5*target('head/head-scale-horiz-decr.target.gz'),'chin':.5*target('chin/chin-height-incr.target.gz'),'temple':.5*target('forehead/forehead-temple-incr.target.gz')};ex='expression/units/caucasian/';expressions={'neutral':zero,'mouth_open':.35*target(ex+'mouth-open.target.gz'),'corner_pull':.35*target(ex+'mouth-corner-puller.target.gz'),'eye_close':.35*(target(ex+'eye-left-closure.target.gz')+target(ex+'eye-right-closure.target.gz'))}
    bpy.ops.wm.open_mainfile(filepath=str(root/'annotation.blend'));scene=bpy.context.scene;obj=bpy.data.objects['CC0_MakeHuman_base'];views={x['name']:x for x in json.loads((root/'views.json').read_text())};scene.cycles.samples=16;scene.render.resolution_x=scene.render.resolution_y=1024;scene.render.resolution_percentage=100;renders=[]
    for identity,expression,view_name in [('base','mouth_open','front'),('base','corner_pull','front'),('base','eye_close','front'),('round','neutral','front'),('round','neutral','profile'),('narrow','mouth_open','front'),('narrow','mouth_open','oblique'),('chin','neutral','profile'),('temple','neutral','oblique')]:
        varied=v+identities[identity]+expressions[expression];obj.data.vertices.foreach_set('co',varied.astype(np.float32).ravel());obj.data.update();scene.camera.matrix_world=Matrix(views[view_name]['matrix']);bpy.context.view_layer.update();name=f'{identity}-{expression}-{view_name}';scene.render.filepath=str(folder/(name+'.png'));bpy.ops.render.render(write_still=True)
        item={'name':name,'identity':identity,'expression':expression,'view':view_name,'note':'Raw core deltas, not a validated expression rig; procedural eyeballs unchanged','pixel_xy':{}}
        for label,mapping in [('before',base),('after',proposed)]:
            points=evaluate_anchors(varied,t,mapping['slots']);uv=np.array([tuple(world_to_camera_view(scene,scene.camera,Vector(q))) for q in points]);item['pixel_xy'][label]=np.c_[uv[:,0]*1024-.5,(1-uv[:,1])*1024-.5].tolist()
        renders.append(item)
    (folder/'views.json').write_text(json.dumps(renders,indent=2));print('INSPECTION_RENDER_COUNT',len(renders))
if __name__=='__main__':main()

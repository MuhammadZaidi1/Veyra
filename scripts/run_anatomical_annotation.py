"""Windows annotation launcher and independently checked candidate rebuild."""
import argparse,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw
from annotation_core import rebuild,digest
from correspondence_math import evaluate_anchors,deformation_metrics
from run_blender_prototype import locate_blender

def project_points(points,view):
 cam=np.c_[points,np.ones(len(points))]@np.linalg.inv(np.array(view['matrix'])).T
 return np.c_[(cam[:,0]/view['width']+.5)*1024-.5,(.5-cam[:,1]/view['width'])*1024-.5]

def generate(project,annotation,output,draft=False,reference_geometry=None):
 config=json.loads((project/'project.json').read_text())
 if digest((project/'surface.npz').read_bytes())!=config['surface_sha256']:raise ValueError('Frozen surface archive changed')
 base=json.loads((project/'base-map.json').read_text());session=json.loads(annotation.read_text());g=np.load(project/'surface.npz');v,t,ids=g['world_xyz'],g['triangles'],g['source_vertex_ids']
 result=rebuild(base,session,v,t,ids,digest((project/'base-map.json').read_bytes()),draft)
 before=evaluate_anchors(v,t,base['slots']);after=evaluate_anchors(v,t,result['slots'])
 for slot in result['slots']:
  if slot['source_vertex_ids']!=ids[t[slot['source_triangle']]].tolist():raise ValueError('Source vertex IDs mismatch')
 from veyra.topology_data import TRIANGLE_INDICES
 tri=np.array(TRIANGLE_INDICES).reshape(-1,3)
 report={'status':result['annotation_status'],'applied_indices':result['annotation_applied_indices'],'point_count':468,'surface_attachment_failures':0,'before_to_after':deformation_metrics(before,after,tri),'warning':'Numerical checks are not anatomical approval; no correction is made by draft preview.'}
 for name,points in [('before',before),('after',after)]:
  distance=np.linalg.norm(points[:,None]-points[None,:],axis=2);report[name+'_duplicate_pairs']=int(np.count_nonzero(np.triu(distance<1e-8,1)))
 audit=np.load(reference_geometry or project.parents[0]/'anatomy-audit-001/audit-geometry.npz',allow_pickle=False)
 reference=audit['canonical_aligned']
 for name,points in [('before',before),('after',after)]:
  report[name+'_fixed_reference_skin']=deformation_metrics(reference,points,tri)
  report[name+'_canonical_including_caps']=deformation_metrics(reference,points,audit['canonical_triangles'])
 output.mkdir(parents=True,exist_ok=False);(output/'mapping-candidates.json').write_text(json.dumps(result,indent=2));(output/'annotations.json').write_text(json.dumps(session,indent=2));(output/'checks.json').write_text(json.dumps(report,indent=2))
 editable={e['veyra_index'] for e in session['annotations']}
 for view in json.loads((project/'views.json').read_text()):
  panels=[]
  for name,points in [('BEFORE',before),('AFTER',after)]:
   image=Image.open(project/(view['name']+'.png')).convert('RGB');draw=ImageDraw.Draw(image);xy=project_points(points,view)
   for region in ['lips','contour']:
    for a,b in base['region_edges'][region]:draw.line([tuple(xy[a]),tuple(xy[b])],fill=(30,230,220),width=1)
   for i,(x,y) in enumerate(xy):
    if i in editable:draw.ellipse((x-2,y-2,x+2,y+2),fill=(255,170,30));draw.text((x+3,y-3),str(i),fill=(255,255,130))
   draw.rectangle((0,0,1024,42),fill='black');draw.text((12,10),name+' — UNVALIDATED; chords are not surface curves',fill='white');panels.append(image)
  sheet=Image.new('RGB',(2048,1024));sheet.paste(panels[0],(0,0));sheet.paste(panels[1],(1024,0));sheet.save(output/('before-after-'+view['name']+'.png'))
 print(json.dumps(report,indent=2))

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','open','rebuild']);p.add_argument('--project',type=Path,default=Path('runs/veyra3d/annotation-project-001'));p.add_argument('--annotations',type=Path);p.add_argument('--output',type=Path);p.add_argument('--draft-preview',action='store_true');p.add_argument('--blender');p.add_argument('--mapping',type=Path);p.add_argument('--prototype',type=Path);p.add_argument('--reference-geometry',type=Path);a=p.parse_args();scripts=Path(__file__).resolve().parent
 if a.action=='prepare':
  command=[locate_blender(a.blender),'--background','--factory-startup','--python-exit-code','1','--python',str(scripts/'blender_prepare_annotation.py'),'--','--project',str(a.project.resolve())]
  for flag,value in [('--mapping',a.mapping),('--prototype',a.prototype)]:
   if value:command.extend([flag,str(value.resolve())])
  subprocess.run(command,check=True,timeout=300)
 elif a.action=='open':subprocess.Popen([locate_blender(a.blender),'--factory-startup',str((a.project/'annotation.blend').resolve()),'--python',str(scripts/'blender_annotation_ui.py')])
 else:
  if not a.annotations or not a.output:p.error('rebuild requires --annotations and --output')
  generate(a.project.resolve(),a.annotations.resolve(),a.output.resolve(),a.draft_preview,a.reference_geometry)
if __name__=='__main__':main()

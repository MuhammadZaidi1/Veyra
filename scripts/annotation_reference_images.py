"""Create labeled source-geometry reference sheets (not anatomical truth)."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from run_anatomical_annotation import project_points
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('project',type=Path);parser.add_argument('--audit',type=Path);args=parser.parse_args();p=args.project;g=np.load(p/'surface.npz');m=json.loads((p/'base-map.json').read_text());entries=json.loads((p/'annotations/draft-0000.json').read_text())['annotations'];views=json.loads((p/'views.json').read_text())
for view in views:
 image=Image.open(p/(view['name']+'.png')).convert('RGB');draw=ImageDraw.Draw(image);xy=project_points(g['world_xyz'],view)
 # An inspectable mouth patch with actual source vertex and triangulated edge IDs.
 mask=(g['world_xyz'][:,2]>.635)&(g['world_xyz'][:,2]<.695)&(abs(g['world_xyz'][:,0])<.042)&(g['world_xyz'][:,1]<-.1)
 selected=set(np.flatnonzero(mask));edges=set()
 for tri in g['triangles']:
  for a,b in [(tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])]:
   if a in selected and b in selected:edges.add(tuple(sorted((int(a),int(b)))))
 for a,b in edges:draw.line([tuple(xy[a]),tuple(xy[b])],fill=(100,140,160))
 for i in selected:
  x,y=xy[i];draw.text((x,y),'V'+str(g['source_vertex_ids'][i]),fill=(255,220,80))
 for e in entries:
  if not e['annotation_group'].startswith('lip'):continue
  point=np.array(e['barycentric'])@g['world_xyz'][g['triangles'][e['source_triangle']]];x,y=project_points(point[None],view)[0];draw.ellipse((x-2,y-2,x+2,y+2),fill='red');draw.text((x,y-12),'MP'+str(e['veyra_index']),fill='cyan')
 box=xy[list(selected)];bounds=(int(box[:,0].min())-30,int(box[:,1].min())-30,int(box[:,0].max())+60,int(box[:,1].max())+30);crop=image.crop(bounds);crop=crop.resize((crop.width*3,crop.height*3));crop.save(p/('mouth-source-'+view['name']+'.png'))
# Existing comparison is explicitly a canonical intended topology reference, not a verified MH border.
audit=args.audit or p.parent/'anatomy-audit-001';sheet=Image.new('RGB',(2048,1100),'black');d=ImageDraw.Draw(sheet);d.text((20,10),'REFERENCE ONLY: canonical intended landmark semantics vs UNVERIFIED MakeHuman candidate borders',fill='white');sheet.paste(Image.open(audit/'comparison-front.png').resize((1024,1024)),(0,60));sheet.paste(Image.open(audit/'comparison-right-profile.png').resize((1024,1024)),(1024,60));sheet.save(p/'reference-board.png')
print('Reference board and three numbered source mouth close-ups saved')
# Local numbered vertex/edge cards avoid hiding anatomy under whole-mouth labels.
for e in entries:
 point=np.array(e['barycentric'])@g['world_xyz'][g['triangles'][e['source_triangle']]];view=views[0];xy=project_points(g['world_xyz'],view);center=project_points(point[None],view)[0];radius=32;scale=8
 image=Image.open(p/'front.png').convert('RGB').crop((int(center[0])-radius,int(center[1])-radius,int(center[0])+radius,int(center[1])+radius)).resize((512,512));draw=ImageDraw.Draw(image)
 selected=np.argsort(np.linalg.norm(g['world_xyz']-point,axis=1))[:8];chosen=set(selected.tolist());local=(xy-center+radius)*scale;edges=set()
 for tri in g['triangles']:
  for a,b in [(tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])]:
   if int(a) in chosen and int(b) in chosen:edges.add(tuple(sorted((int(a),int(b)))))
 for a,b in edges:draw.line([tuple(local[a]),tuple(local[b])],fill='cyan',width=1)
 for j,i in enumerate(selected):
  x,y=local[i];label=(8,45+j*24);draw.line([(x,y),(label[0]+80,label[1])],fill=(200,180,60));draw.text(label,'V'+str(g['source_vertex_ids'][i]),fill='yellow');draw.ellipse((x-2,y-2,x+2,y+2),fill='yellow')
 draw.rectangle((0,0,512,30),fill='black');draw.text((8,8),f'MP {e["veyra_index"]} {e["annotation_group"]}: UNREVIEWED',fill='white');draw.ellipse((253,253,259,259),fill='red')
 draw.rectangle((0,425,512,512),fill='black');draw.text((8,430),'Source triangle edges (includes quad diagonals):',fill='white')
 for j,(a,b) in enumerate(sorted(edges)[:8]):draw.text((8+(j%2)*245,445+(j//2)*15),f'E{g["source_vertex_ids"][a]}-{g["source_vertex_ids"][b]}',fill='cyan')
 folder=p/'point-cards';folder.mkdir(exist_ok=True);image.save(folder/f'MP-{e["veyra_index"]:03d}.png')

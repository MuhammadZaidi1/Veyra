"""Visual/machine evidence of retained baselines, rejected regions and native topology."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scripts.makehuman_semantics import sha

def project(points,view):
    cam=np.c_[points,np.ones(len(points))]@np.linalg.inv(np.array(view['matrix'])).T;n=view['resolution']
    return np.c_[(cam[:,0]/view['width']+.5)*n-.5,(.5-cam[:,1]/view['width'])*n-.5]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--result',type=Path,required=True);p.add_argument('--project',type=Path,required=True);a=p.parse_args();root=a.result
    if (root/'regional-sheet.png').exists():raise ValueError('Visual review already exists; preserve it and use a fresh result')
    report=json.loads((root/'report.json').read_text());records=json.loads((root/'point-evidence.json').read_text());inventory=json.loads((root/'semantic-inventory.json').read_text());native=json.loads((root/'native-topology-evaluation.json').read_text());g=np.load(root/'inspection-geometry.npz');baseline=g['baseline'];v=g['source_vertices'];ids=g['source_vertex_ids'];lookup={int(x):i for i,x in enumerate(ids)};base=json.loads((root/'mapping-retained.json').read_text());views=json.loads((a.project/'views.json').read_text());editable=[r['veyra_index'] for r in records]
    try:font=ImageFont.truetype('DejaVuSans.ttf',20);small=ImageFont.truetype('DejaVuSans.ttf',14)
    except OSError:font=ImageFont.load_default(size=20);small=ImageFont.load_default(size=14)
    def overlay(image,xy,indices,color,edges=True):
        image=image.copy();d=ImageDraw.Draw(image)
        if edges:
            for region in ['lips','contour']:
                for i,j in base['region_edges'][region]:
                    if i in indices or j in indices:d.line([tuple(xy[i]),tuple(xy[j])],fill=color,width=1)
        for i in indices:
            x,y=xy[i];d.ellipse((x-2,y-2,x+2,y+2),fill=color);d.text((x+3,y-5),str(i),font=small,fill=color,stroke_width=1,stroke_fill='black')
        return image
    before_failed=np.load(a.project.parent/'annotation-proposals-005/proposal-geometry.npz')['after'] if (a.project.parent/'annotation-proposals-005/proposal-geometry.npz').is_file() else None
    for view in views:
        source=Image.open(a.project/(view['name']+'.png')).convert('RGB');panels=[]
        for title,key,color in [('Original candidate baseline','baseline',(80,210,250)),('Rig-constrained trial - REJECTED','candidate_rig_combined',(255,85,80)),('Retained result - UNCHANGED / UNRESOLVED','retained',(255,200,70))]:
            image=overlay(source,project(g[key],view),editable,color);d=ImageDraw.Draw(image);d.rectangle((0,0,1024,60),fill='black');d.text((10,6),title,font=font,fill='white');d.text((10,34),'No anatomical acceptance. Lines are landmark chords, not continuous skin curves.',font=small,fill='white');panels.append(image)
        sheet=Image.new('RGB',(3072,1024));[sheet.paste(image,(i*1024,0)) for i,image in enumerate(panels)];sheet.save(root/('comparison-'+view['name']+'.png'))
    folder=root/'regions';folder.mkdir(exist_ok=True);groups=sorted({r['region'] for r in records});cards=[];front=views[0]
    for group in groups:
        indices=[r['veyra_index'] for r in records if r['region']==group];before=project(baseline,front);trial=project(g['candidate_rig_combined'],front);coords=np.r_[before[indices],trial[indices]];lo=np.floor(coords.min(0)-25).astype(int);hi=np.ceil(coords.max(0)+25).astype(int);width,height=np.maximum(hi-lo,[100,80]);scale=min(4,700/width,550/height);card=Image.new('RGB',(2250,710),'black');draw=ImageDraw.Draw(card)
        for col,(name,points,color) in enumerate([('BASELINE',baseline,(80,210,250)),('RIG TRIAL - REJECTED',g['candidate_rig_combined'],(255,85,80)),('RETAINED - UNCHANGED',baseline,(255,200,70))]):
            image=Image.open(a.project/'front.png').convert('RGB').crop((lo[0],lo[1],lo[0]+width,lo[1]+height)).resize((int(width*scale),int(height*scale)));xy=(project(points,front)-lo)*scale;image=overlay(image,xy,indices,color);card.paste(image,(col*750+10,65));draw.text((col*750+15,10),name,font=font,fill=color)
        draw.text((12,630),group+': '+', '.join(map(str,indices)),font=font,fill='white');draw.text((12,665),'Rig endpoints identify an editing region, not an exact MediaPipe anatomical boundary.',font=small,fill='white');card.save(folder/(group+'.png'));cards.append(card)
    board=Image.new('RGB',(2250,710*len(cards)));[board.paste(image,(0,k*710)) for k,image in enumerate(cards)];board.save(root/'regional-sheet.png')
    # Actual original source connectivity, no MediaPipe triangles and no sparse Delaunay surrogate.
    tri=np.array(native['triangles']);source=np.array(native['source_vertex_ids']);native_vertices=np.array([lookup[int(i)] for i in source]);edges=np.unique(np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1),axis=0)
    for view in views:
        image=Image.open(a.project/(view['name']+'.png')).convert('RGB');xy=project(v[native_vertices],view);d=ImageDraw.Draw(image)
        for i,j in edges:d.line([tuple(xy[i]),tuple(xy[j])],fill=(40,190,180),width=1)
        d.rectangle((0,0,1024,65),fill='black');d.text((10,6),f'Native source patch: {len(source)} vertices / {len(tri)} triangles',font=font,fill='white');d.text((10,36),'Exact source IDs; not MediaPipe semantics; no training contract change.',font=small,fill='white');image.save(root/('native-topology-'+view['name']+'.png'))
    # Rig endpoints deliberately include off-surface means, with their support spans visible.
    image=Image.open(a.project/'front.png').convert('RGB');d=ImageDraw.Draw(image)
    unique={}
    for entry in inventory['rig_endpoints']:
        key=tuple(entry['source_vertex_ids']);unique.setdefault(key,entry)
    for entry in unique.values():
        pts=np.array([v[lookup[i]] for i in entry['source_vertex_ids']]);xy=project(pts,front);center=project(np.array(entry['position'])[None],front)[0]
        if len(xy)>1:d.line([tuple(x) for x in xy],fill=(255,100,240),width=2)
        x,y=center;d.ellipse((x-2,y-2,x+2,y+2),fill=(255,100,240));d.text((x+4,y-5),entry['name'],font=small,fill=(255,180,240),stroke_width=1,stroke_fill='black')
    d.rectangle((0,0,1024,65),fill='black');d.text((10,6),'Author rig references - NOT canonical anatomical labels',font=font,fill='white');d.text((10,36),'Magenta spans connect contributors to MEAN endpoints; means can leave the skin.',font=small,fill='white');image.save(root/'rig-semantic-references.png')
    render_folder=root/'inspection-renders'
    if (render_folder/'renders.json').is_file():
        for item in json.loads((render_folder/'renders.json').read_text()):
            source=Image.open(render_folder/item['file']).convert('RGB');sheet=Image.new('RGB',(3072,1024))
            for col,(label,key,color) in enumerate([('BASELINE','baseline',(80,210,250)),('REJECTED RIG TRIAL','rig_combined',(255,85,80)),('RETAINED / UNVALIDATED','baseline',(255,200,70))]):
                image=overlay(source,np.array(item['pixels'][key]),editable,color);d=ImageDraw.Draw(image);d.rectangle((0,0,1024,65),fill='black');d.text((10,5),label+' '+item['case'],font=font,fill='white');d.text((10,35),'Raw core target combination; inspect recorded geometry failures.',font=small,fill='white');sheet.paste(image,(col*1024,0))
            sheet.save(render_folder/(Path(item['file']).stem+'-comparison.png'))
    camera_path=root/'inspection-renders/candidate-view-checks.json';candidate_checks=json.loads(camera_path.read_text()) if camera_path.exists() else []
    failing_base=[c for c in report['baseline_cases'] if not c['passed']];safe=[c['case'] for c in native['case_checks'] if c['passed']]
    rows=['# Strict automated correspondence results','', '**No reliable anatomical correction was established. All 56 disputed slots are retained unchanged.**','',f"Baseline SHA256: `{report['baseline_mapping_sha256']}`. Operational checkpoint: `{report['github_checkpoint']}`; baseline is not anatomically accepted.",'', f"Tested {len(report['baseline_cases'])} cases, {len(report['evaluations'])} candidate families/singletons, {len(report['view_checks'])} baseline camera checks and {len(candidate_checks)} neutral candidate camera checks.",'',f'Baseline stress-case failures: {len(failing_base)}; native patch structurally screened configurations: {len(safe)}/{len(native["case_checks"])}. No unsafe requests are approved for training.','', '| Candidate | No regression vs posed baseline | Absolute structural pass | Failing cases | Decision |','|---|---|---|---:|---|']
    for name,e in report['evaluations'].items():
        if name.startswith('rig_point_'):continue
        rows.append(f"| {name} | {e.get('no_regression_passed',False)} | {e['structurally_admissible']} | {sum(not c['passed'] for c in e['case_checks'])} | {e['decision']} |")
    rows+=['','## Per-point machine-evidence index','','Every point remains unresolved. Each JSON entry includes all source triangles/weights, regional context and failure cases.','', '| MP | Region | Geometrically nonregressing trials | Outcome |','|---|---|---|---|']
    for record in records:
        names=[c['family'] for c in record['candidates'] if c.get('no_regression_passed')];rows.append(f"| {record['veyra_index']} | {record['region']} | {', '.join(names) or 'None'} | Unchanged: exact anatomical semantics unresolved |")
    (root/'summary.md').write_text('\n'.join(rows)+'\n');print('STRICT_VISUALS: regional panels, native topology, rig references, deformation comparisons, summary')

if __name__=='__main__':main()

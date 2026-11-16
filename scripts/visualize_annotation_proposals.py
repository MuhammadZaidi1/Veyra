"""Readable proposal evidence: 3D-projected before/after, ensembles and regional review cards."""
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.correspondence_math import evaluate_anchors,deformation_metrics
from veyra.topology_data import TRIANGLE_INDICES


def xy(points,view):
    cam=np.c_[points,np.ones(len(points))]@np.linalg.inv(np.array(view['matrix'])).T;n=view['resolution']
    return np.c_[(cam[:,0]/view['width']+.5)*n-.5,(.5-cam[:,1]/view['width'])*n-.5]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);args=p.parse_args();root=args.project
    data=np.load(root/'proposal-geometry.npz');surface=np.load(root/'surface.npz');before,after,ref,ensemble=data['before'],data['after'],data['reference'],data['ensemble'];editable=data['editable_indices'].tolist()
    records=json.loads((root/'point-evidence.json').read_text());report=json.loads((root/'proposal-report.json').read_text());base=json.loads((root/'base-map.json').read_text());proposed=json.loads((root/'mapping-proposals.json').read_text());views=json.loads((root/'views.json').read_text());record={r['veyra_index']:r for r in records}
    try:font=ImageFont.truetype('DejaVuSans.ttf',20);small=ImageFont.truetype('DejaVuSans.ttf',14)
    except OSError:font=ImageFont.load_default(size=20);small=ImageFont.load_default(size=14)
    skin=np.array(TRIANGLE_INDICES);a=before[skin];b=after[skin];flipped=np.flatnonzero((np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0])*np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0])).sum(1)<0)
    supported=set(report['supported_geometric_indices']);conservative=copy.deepcopy(base)
    for i in supported:conservative['slots'][i]=copy.deepcopy(proposed['slots'][i])
    conservative.update(status='candidate_unvalidated',reviewer=None,proposal_only=True,proposal_editable_indices=sorted(supported),warning='Only locally supported geometric proposals; NOT anatomical approval. All other slots unchanged.')
    cp=evaluate_anchors(surface['world_xyz'],surface['triangles'],conservative['slots']);checks=deformation_metrics(before,cp,np.array(TRIANGLE_INDICES))
    (root/'mapping-supported-only-proposals.json').write_text(json.dumps(conservative,indent=2));(root/'supported-only-checks.json').write_text(json.dumps(checks,indent=2))
    sheets={}
    for view in views:
        panels=[];original=Image.open(root/(view['name']+'.png')).convert('RGB')
        for title,points in [('ORIGINAL - unvalidated',before),('FULL PROPOSAL - unsafe pending review',after),('CANONICAL-DERIVED PRIOR - not surface attachment',ref)]:
            image=original.copy();draw=ImageDraw.Draw(image);proj=xy(points,view)
            for region in ['lips','contour']:
                for a,b in base['region_edges'][region]:draw.line([tuple(proj[a]),tuple(proj[b])],fill=(40,180,200),width=1)
            if title.startswith('FULL'):
                for triangle in skin[flipped]:
                    line=[tuple(proj[i]) for i in triangle];draw.line(line+[line[0]],fill=(255,60,65),width=2)
                for positions in ensemble:
                    for i,(x,y) in zip(editable,xy(positions,view)):draw.ellipse((x-1,y-1,x+1,y+1),fill=(180,130,200))
            for i in editable:
                x,y=proj[i];color=(60,255,120) if i in supported else (255,160,40);draw.ellipse((x-2,y-2,x+2,y+2),fill=color);draw.text((x+4,y-7),str(i),font=small,fill=color,stroke_width=1,stroke_fill='black')
            draw.rectangle((0,0,1024,60),fill='black');draw.text((10,4),title,font=font,fill='white');draw.text((10,33),'Green: local numerical support only. Orange: ambiguous. No accepted points.',font=small,fill='white');panels.append(image)
        sheet=Image.new('RGB',(3072,1024));[sheet.paste(image,(k*1024,0)) for k,image in enumerate(panels)];sheet.save(root/('comparison-'+view['name']+'.png'));sheets[view['name']]=panels
    # Seven region decisions: compare geometry before examining every point separately.
    questions=[('upper-lip','lip_upper','Is the outer upper border near the proposed ridge, distinct from the inner rim?',0),('lower-lip','lip_lower','Where is the vermilion border, distinct from the labiomental crease?',17),('mouth-corners','lip_corners','Do 61/291 mark the commissures, rather than merely a narrower canonical mouth?',61),('forehead','forehead','Which forehead extent represents the canonical face patch? It is not a hairline.',10),('temples-cheeks','temples','Review temporal and cheek patch boundaries in oblique/profile, not the silhouette.',234),('jaw','jaw','Does the curve follow mandibular anatomy rather than the camera silhouette?',172),('chin','chin','Which chin extent/depth matches canonical landmark semantics?',152)]
    review_cards=[]
    for name,group,question,control in questions:
        selected=[r['veyra_index'] for r in records if r['group']==group];view=views[0];proj=xy(np.r_[before[selected],after[selected]],view);lo=np.floor(proj.min(0)-22).astype(int);hi=np.ceil(proj.max(0)+22).astype(int)
        # Do not use mirrored-vertex equality as a confidence cue; paired cards remain distinct.
        width=max(hi[0]-lo[0],100);height=max(hi[1]-lo[1],80);scale=min(6,700/width,500/height)
        card=Image.new('RGB',(1500,730),'black');d=ImageDraw.Draw(card);d.text((15,8),name.upper()+' - HUMAN REGION DECISION REQUIRED',font=font,fill='white');d.text((15,40),question,font=small,fill='white')
        for column,(title,points) in enumerate([('Before',before),('Proposed',after)]):
            crop=Image.open(root/'front.png').convert('RGB').crop((lo[0],lo[1],lo[0]+width,lo[1]+height));crop=crop.resize((round(width*scale),round(height*scale)));draw=ImageDraw.Draw(crop)
            local=(xy(points,view)-lo)*scale
            for region in ['lips','contour']:
                for a,b in base['region_edges'][region]:
                    if a in selected or b in selected:draw.line([tuple(local[a]),tuple(local[b])],fill=(40,200,220),width=2)
            for i in selected:
                x,y=local[i];color=(60,255,120) if i in supported else (255,160,40);draw.ellipse((x-3,y-3,x+3,y+3),fill=color);draw.text((x+5,y-8),str(i),font=font,fill=color,stroke_width=1,stroke_fill='black')
            card.paste(crop,(column*750+10,90));d.text((column*750+15,65),title,font=font,fill='white')
        d.text((15,620),'Indices: '+', '.join(map(str,selected)),font=small,fill='white');d.text((15,650),'Seed control '+str(control)+': inspect source-ID card and profile. All decisions remain uncertain.',font=small,fill=(255,200,80));folder=root/'review-cards';folder.mkdir(exist_ok=True);card.save(folder/(name+'.png'));review_cards.append(card)
    board=Image.new('RGB',(1500,730*7));[board.paste(image,(0,k*730)) for k,image in enumerate(review_cards)];board.save(root/'review-board.png')
    # Detailed source IDs around each proposed marker, plus numbered reference ID.
    folder=root/'point-cards';folder.mkdir(exist_ok=True)
    st=surface['triangles'];source_edges=np.unique(np.sort(np.concatenate([st[:,[0,1]],st[:,[1,2]],st[:,[2,0]]]),axis=1),axis=0)
    for i in editable:
        position=after[i];view=views[0];center=xy(position[None],view)[0];radius=32;scale=8;image=Image.open(root/'front.png').convert('RGB').crop((int(center[0])-radius,int(center[1])-radius,int(center[0])+radius,int(center[1])+radius)).resize((512,512));d=ImageDraw.Draw(image)
        nearest=np.argsort(np.linalg.norm(surface['world_xyz']-position,axis=1))[:8];local=(xy(surface['world_xyz'][nearest],view)-center+radius)*scale
        positions={int(vid):tuple(q) for vid,q in zip(nearest,local)};edges=source_edges[np.isin(source_edges,nearest).all(1)]
        for a,b in edges:d.line([positions[int(a)],positions[int(b)]],fill='cyan',width=1)
        for j,(vid,(x,y)) in enumerate(zip(nearest,local)):
            label=(8,45+j*24);d.line([(x,y),(95,label[1])],fill=(190,170,40));d.text(label,'V'+str(surface['source_vertex_ids'][vid]),fill='yellow')
        d.rectangle((0,0,512,30),fill='black');d.text((8,8),f'MP {i}: {record[i]["classification"]}',fill='white');d.ellipse((253,253,259,259),fill='red');d.rectangle((0,425,512,512),fill='black');d.text((8,433),'Source triangle '+str(record[i]['source_triangle'])+'; anatomy UNVERIFIED',fill='white');d.text((8,455),'Hypothesis spread '+str(round(record[i]['ensemble_max_spread']*1000,2))+' nominal scene mm',fill='white');
        for j,(a,b) in enumerate(edges[:4]):d.text((8+(j%2)*245,475+(j//2)*15),f'E{surface["source_vertex_ids"][a]}-{surface["source_vertex_ids"][b]}',fill='cyan')
        image.save(folder/f'MP-{i:03d}.png')
    # Diagnostic renders include actual varied geometry, not just shifted 2D labels.
    stability=root/'stability-renders'
    if (stability/'views.json').exists():
        for item in json.loads((stability/'views.json').read_text()):
            source=Image.open(stability/(item['name']+'.png')).convert('RGB');sheet=Image.new('RGB',(2048,1024))
            for column,label in enumerate(['before','after']):
                image=source.copy();d=ImageDraw.Draw(image);proj=np.array(item['pixel_xy'][label])
                for region in ['lips','contour']:
                    for a,b in base['region_edges'][region]:d.line([tuple(proj[a]),tuple(proj[b])],fill=(40,190,210),width=1)
                for i in editable:
                    x,y=proj[i];d.ellipse((x-2,y-2,x+2,y+2),fill=(255,170,40));d.text((x+3,y-5),str(i),font=small,fill='yellow',stroke_width=1,stroke_fill='black')
                d.rectangle((0,0,1024,60),fill='black');d.text((10,5),label.upper()+' '+item['name'],font=font,fill='white');d.text((10,33),'UNVALIDATED raw core deltas; fixed procedural eyeballs',font=small,fill='white');sheet.paste(image,(column*1024,0))
            sheet.save(stability/(item['name']+'-comparison.png'))
    rows=['# Automated annotation proposal review','', '**No anatomical approval. No accepted points. Full proposal is not safe to apply as a complete map.**','',f'56 proposals; {len(supported)} locally supported geometric proposals; {56-len(supported)} ambiguous.','', '## Candidate checks','',f"- Full 56-point change: {report['new_skin_metrics']['normal_flip_triangles']} dense-skin normal reversals and {report['new_skin_metrics']['collapsed_or_stretched_edges']} extreme edge ratios versus original mapping.",f"- Supported-only alternative: {checks['normal_flip_triangles']} reversals and {checks['collapsed_or_stretched_edges']} extreme edge ratios; still anatomically unvalidated.",f"- {len(report['cases'])} core identity/expression combinations; {sum(len(c['support_failures']) for c in report['cases'])} support-triangle failures.",'', '## Seven grouped review questions','']
    rows.extend(f'- **{name}:** {question}' for name,group,question,control in questions)
    rows+=['','These seven visual region checks reduce navigation and adjustment work; they do not prove only seven points are ambiguous. All 54 ambiguous points remain explicitly listed below.','', '| MP | Group | Status | Move (scene mm) | Ensemble spread | Morph reselect drift | Curvature rank | Morph-gradient rank | Flags |','|---|---|---|---:|---:|---:|---:|---:|---|']
    for r in records:rows.append(f"| {r['veyra_index']} | {r['group']} | {r['classification']} | {r['displacement']*1000:.2f} | {r['ensemble_max_spread']*1000:.2f} | {r['fixed_pool_reselection_max_drift']*1000:.2f} | {r['curvature_regional_percentile']:.2f} | {r['morph_gradient_regional_percentile']:.2f} | {', '.join(r['flags'])} |")
    (root/'review-report.md').write_text('\n'.join(rows)+'\n');print('Comparison overlays, seven region cards, 56 source-ID cards and evidence report written; supported-only checks:',checks)

if __name__=='__main__':main()

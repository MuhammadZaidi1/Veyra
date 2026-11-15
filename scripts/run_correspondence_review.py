"""Candidate fitting launcher + large numbered review overlays. Never exports training data."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scripts.run_blender_prototype import locate_blender

REVISION='02ca022d573c004c463d8c61eb3f8ce17e95a7fc'
REFERENCE_FILES={
    'canonical_face_model.obj':('mediapipe/modules/face_geometry/data/canonical_face_model.obj','8bac80443397e113f41a8b565ea72c59390bc031d9defab289dba7bc0c54e618'),
    'face_mesh_connections.py':('mediapipe/python/solutions/face_mesh_connections.py','cc7171e44d0612db38720cb2fd2516c395992cdd7457efe341543be6f3c22d87'),
    'LICENSE':('LICENSE','8707eef0533987efc5b155d64761eeb6e20793f50b9bd1a68dad1cf4719d0ed8')}


def fetch_reference(root):
    root=Path(root)
    # All downloads verified in memory before writing. Only this explicit operation uses network.
    data={};meta={'revision':REVISION,'files':{}}
    for name,(remote,digest) in REFERENCE_FILES.items():
        url=f'https://raw.githubusercontent.com/google-ai-edge/mediapipe/{REVISION}/{remote}'
        payload=urllib.request.urlopen(url,timeout=30).read()
        if hashlib.sha256(payload).hexdigest()!=digest:raise ValueError('Reference hash mismatch: '+name)
        data[name]=payload
        meta['files'][name]={'url':url,'sha256':digest}
        if (root/name).exists() and (root/name).read_bytes()!=payload:raise ValueError('Refusing changed reference overwrite')
    root.mkdir(parents=True,exist_ok=True)
    for name,payload in data.items():(root/name).write_bytes(payload)
    (root/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')


def overlays(output):
    output=Path(output)
    mapping=json.loads((output/'mapping-candidates.json').read_text())
    report=json.loads((output/'review-report.json').read_text())
    flags={s['veyra_index'] for s in mapping['slots'] if s['review_flags']}
    edges=mapping['region_edges']
    try:font=ImageFont.truetype('DejaVuSans.ttf',16)
    except OSError:font=ImageFont.load_default()
    def draw_points(source,xy,selected,title,scale=3):
        image=source.resize((source.width*scale,source.height*scale))
        draw=ImageDraw.Draw(image)
        for region_edges in edges.values():
            for a,b in region_edges:
                if a in selected and b in selected:
                    draw.line([tuple(xy[a]*scale),tuple(xy[b]*scale)],fill=(80,130,255),width=2)
        for i in sorted(selected):
            x,y=xy[i]*scale
            color=(255,90,60) if i in flags else (60,255,180)
            draw.ellipse((x-3,y-3,x+3,y+3),fill=color)
            draw.text((x+4,y-10),str(i),font=font,fill=(255,255,80),stroke_width=1,stroke_fill=(0,0,0))
        draw.rectangle((0,0,image.width,52),fill=(0,0,0))
        draw.text((8,4),'UNVALIDATED CANDIDATES - '+title,font=font,fill=(255,255,0))
        draw.text((8,28),'Red: heuristic review flag. Green: no flag, NOT anatomical approval.',font=font,fill='white')
        return image
    sheet=Image.new('RGB',(512*3,560*3),(20,20,20))
    for number,render in enumerate(report['renders']):
        with Image.open(output/(render['name']+'.png')) as opened:source=opened.convert('RGB')
        xy=np.array(render['pixel_xy'])
        image=draw_points(source,xy,set(range(468)),render['name'],scale=max(1,1536//source.width))
        image.save(output/(render['name']+'-numbered.png'))
        thumb=image.resize((512,512))
        x=(number%3)*512;y=(number//3)*560
        sheet.paste(thumb,(x,y));ImageDraw.Draw(sheet).text((x+5,y+518),render['name'],fill='white',font=font)
        if render['name']=='base-neutral-front':
            for region,ids in mapping['regions'].items():
                # Crop original face and convert to local pixels for readable region IDs.
                ids=set(ids);p=xy[list(ids)]
                margin=20*source.width/512
                xmin,ymin=np.maximum(np.floor(p.min(0)-margin),0).astype(int)
                xmax,ymax=np.minimum(np.ceil(p.max(0)+margin),source.width).astype(int)
                # Keep enough vertical room for the warning banner above the region.
                ymin=max(0,ymin-int(25*source.width/512))
                crop=source.crop((xmin,ymin,xmax,ymax))
                closeup=draw_points(crop,xy-[xmin,ymin],ids,region,scale=max(2,7680//source.width))
                closeup.save(output/('region-'+region+'.png'))
    sheet.save(output/'inspection-sheet.png')
    rows=['# Candidate review queue','', 'All 468 slots require human anatomical review. Flags are heuristics, not probabilities.','',
        '| Index | Triangle | Reasons |','|---|---|---|']
    rows.extend(f"| {s['veyra_index']} | {s['source_triangle']} | {', '.join(s['review_flags']) or 'No heuristic flag; still unreviewed'} |" for s in mapping['slots'])
    (output/'review-queue.md').write_text('\n'.join(rows)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',choices=['biharmonic','polyharmonic'],default='polyharmonic')
    parser.add_argument('--projection',choices=['closest','parameter-ray'],default='closest')
    parser.add_argument('--geometry-only',action='store_true')
    parser.add_argument('--fitting',choices=['legacy','anatomical'],default='anatomical')
    parser.add_argument('--resolution',type=int,choices=[512,1024,1536],default=1536)
    parser.add_argument('--blender',type=Path)
    parser.add_argument('--prototype',type=Path,default=Path('runs/veyra3d/blender-prototype-002'))
    parser.add_argument('--reference',type=Path,default=Path('third_party/mediapipe-reference'))
    parser.add_argument('--output',type=Path,default=Path('runs/veyra3d/correspondence-anatomical-001'))
    parser.add_argument('--controls',type=Path)
    parser.add_argument('--anchors',type=Path)
    parser.add_argument('--fetch-reference',action='store_true',help='Explicitly fetch pinned mesh/connectivity/license only, then exit')
    parser.add_argument('--overlays-only',action='store_true')
    args=parser.parse_args()
    if args.fetch_reference:
        fetch_reference(args.reference);return
    if not args.overlays_only:
        if args.output.exists():parser.error('Output exists; choose a fresh directory')
        command=[locate_blender(args.blender),'--background','--python-exit-code','1','--python',str(Path(__file__).with_name('blender_correspondence.py').resolve()),'--',
            '--prototype',str(args.prototype.resolve()),'--reference',str(args.reference.resolve()),'--output',str(args.output.resolve())]
        if args.geometry_only:command.append('--geometry-only')
        command.extend(['--fitting',args.fitting,'--resolution',str(args.resolution),'--registration',args.registration,'--projection',args.projection])
        if args.anchors:command.extend(['--anchors',str(args.anchors.resolve())])
        if args.controls:command.extend(['--controls',str(args.controls.resolve())])
        subprocess.run(command,check=True,timeout=300)
    report=json.loads((args.output/'review-report.json').read_text()) if args.geometry_only else overlays(args.output)
    print(json.dumps({k:report[k] for k in ['candidate_anchors','numerical_anchor_failures','flagged_candidates','flag_counts']},indent=2))


if __name__=='__main__':main()

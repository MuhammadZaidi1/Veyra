"""Read-only anatomical audit; never rewrites correspondence or exports labels."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scripts.correspondence_math import parse_obj,evaluate_anchors,deformation_metrics
from scripts.anatomical_constraints import body_quads
from veyra.topology_data import TRIANGLE_INDICES


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(mapping_root,prototype,reference,baseline,asset_root,output):
    mapping_root,prototype,reference,baseline,asset_root,output=map(Path,(mapping_root,prototype,reference,baseline,asset_root,output))
    if output.exists():raise ValueError('Audit output must be fresh')
    mapping_path=mapping_root/'mapping-candidates.json';before=sha(mapping_path)
    mapping=json.loads(mapping_path.read_text());curves=json.loads((mapping_root/'anatomical-curves.json').read_text())['curves']
    surface=np.load(prototype/'surface-review.npz',allow_pickle=False)
    canonical,ct=parse_obj((reference/'canonical_face_model.obj').read_text())
    source_vertices=surface['world_xyz'];triangles=surface['triangles'];source_ids=surface['source_vertex_ids']
    points=evaluate_anchors(source_vertices,triangles,mapping['slots'])
    aligned=np.load(mapping_root/'alignment-review.npz',allow_pickle=False)['aligned_world']
    fixed=np.load(baseline/'alignment-review.npz',allow_pickle=False)['aligned_world']
    actual=np.asarray(TRIANGLE_INDICES,dtype=np.int32)
    actual_faces={tuple(sorted(t)) for t in actual}
    actual_edges={tuple(sorted((int(a),int(b)))) for t in actual for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0]))}
    mouth={c['name']:c for c in curves if c.get('name','').startswith('mouth')}
    inner=set(mouth['mouth_inner']['canonical_loop']);outer=set(mouth['mouth_outer']['canonical_loop'])
    caps=[i for i,t in enumerate(ct) if set(t)<=inner]
    p0=fixed[ct];p1=points[ct];n0=np.cross(p0[:,1]-p0[:,0],p0[:,2]-p0[:,0]);n1=np.cross(p1[:,1]-p1[:,0],p1[:,2]-p1[:,0])
    cos=(n0*n1).sum(1)/np.maximum(np.linalg.norm(n0,axis=1)*np.linalg.norm(n1,axis=1),1e-30)
    warnings=[]
    for i in np.flatnonzero(cos<0):
        warnings.append({'canonical_triangle_id':int(i),'landmarks':ct[i].tolist(),
            'normal_angle_degrees':float(np.degrees(np.arccos(np.clip(cos[i],-1,1)))),
            'reference_area':float(np.linalg.norm(n0[i])/2),'mapped_area':float(np.linalg.norm(n1[i])/2),
            'canonical_mouth_cap':int(i) in caps,'exists_in_veyra_skin':tuple(sorted(ct[i])) in actual_faces,
            'source_support_triangles':[mapping['slots'][int(j)]['source_triangle'] for j in ct[i]]})
    edges=np.unique(np.sort(np.concatenate((ct[:,[0,1]],ct[:,[1,2]],ct[:,[2,0]])),axis=1),axis=0)
    old=np.linalg.norm(fixed[edges[:,0]]-fixed[edges[:,1]],axis=1);new=np.linalg.norm(points[edges[:,0]]-points[edges[:,1]],axis=1)
    ratio=new/old;edge_warnings=[]
    for i in np.flatnonzero((ratio<.25)|(ratio>4)):
        edge_warnings.append({'landmarks':edges[i].tolist(),'reference_length':float(old[i]),'mapped_length':float(new[i]),
            'ratio':float(ratio[i]),'exists_in_veyra_skin':tuple(edges[i]) in actual_edges,
            'both_inner_lip':set(edges[i])<=inner,'same_source_point':bool(new[i]<1e-8)})
    # The canonical face oval is an actual boundary of the canonical OBJ face patch.
    all_edges=np.sort(np.concatenate((ct[:,[0,1]],ct[:,[1,2]],ct[:,[2,0]])),axis=1)
    unique,counts=np.unique(all_edges,axis=0,return_counts=True);boundary={tuple(t) for t in unique[counts==1]}
    oval_edges={tuple(sorted(e)) for e in mapping['region_edges']['contour']}
    actual_edge_array=np.sort(np.concatenate((actual[:,[0,1]],actual[:,[1,2]],actual[:,[2,0]])),axis=1)
    ae,ac=np.unique(actual_edge_array,axis=0,return_counts=True)
    actual_boundary={tuple(e) for e in ae[ac==1]}
    def oval_proportions(p,up):
        height=float(p[10,up]-p[152,up]);eye_height=float(p[[33,133,263,362],up].mean())
        return {'width_over_height':float((p[454,0]-p[234,0])/height),'forehead_fraction_above_outer_inner_canthi':float((p[10,up]-eye_height)/height)}
    oval=mapping['regions']['contour'];oval_distance=np.linalg.norm(points[oval]-aligned[oval],axis=1)
    def shape_metrics(p,up,forward):
        width=float(abs(p[291,0]-p[61,0]));eye_width=float(abs(p[263,0]-p[33,0]))
        return {'mouth_width':width,'width_over_outer_canthi_span':width/eye_width,
            'outer_height_over_width':float((p[0,up]-p[17,up])/width),
            'upper_band_over_width':float((p[0,up]-p[13,up])/width),
            'lower_band_over_width':float((p[14,up]-p[17,up])/width),
            'aperture_vertical_gap_over_width':float((p[13,up]-p[14,up])/width),
            'lower_border_forward_minus_inner_over_width':float((forward(p[17])-forward(p[14]))/width)}
    # Audit the supplied material UV region; it is NOT an anatomical vermilion annotation.
    asset=asset_root/'data/3dobjs/base.obj'
    if sha(asset)!=mapping['asset_sha256']:raise ValueError('Source asset hash mismatch')
    quads=body_quads(asset.read_text());uv_path=asset_root/'data/uv_layers/lips_solid.json.gz'
    uv=json.loads(gzip.decompress(uv_path.read_bytes()));q=[quads[int(i)] for i in uv]
    qe=np.array([sorted((a,b)) for f in q for a,b in zip(f,f[1:]+f[:1])]);ue,uc=np.unique(qe,axis=0,return_counts=True)
    uv_boundary=set(ue[uc==1].ravel().tolist())
    profile_seeds=[(362,455,466,467,509,468,474),(5337,492,496,495,513,494,493)]
    lookup={int(i):j for j,i in enumerate(source_ids)}
    profiles={name:[{'source_vertex_id':int(i),'world_xyz':source_vertices[lookup[i]].tolist()} for i in chain]
              for name,chain in zip(('upper','lower'),profile_seeds)}
    report={'status':'focused_audit_unvalidated','mapping_unchanged':True,'mapping_sha256':before,
        'normal_warnings':warnings,'edge_warnings':edge_warnings,
        'veyra_skin_metrics':deformation_metrics(fixed,points,actual),
        'canonical_cap_triangle_count':len(caps),'canonical_face_count':len(ct),'veyra_face_count':len(actual),
        'shared_unordered_faces':len({tuple(sorted(t)) for t in ct}&actual_faces),
        'face_oval':{'vertices':len(oval),'canonical_boundary_matches_official_oval':boundary==oval_edges,
            'oval_is_boundary_of_veyra_skin':oval_edges<=actual_boundary,'veyra_total_boundary_edges':len(actual_boundary),
            'canonical_proportions':oval_proportions(canonical,1),'mapped_proportions':oval_proportions(points,2),
            'coarse_source_controls':{str(i):{'world_xyz':points[i].tolist(),'source_vertex_ids':mapping['slots'][i]['source_vertex_ids'],'barycentric':mapping['slots'][i]['barycentric']} for i in [10,152]},
            'maximum_similarity_reference_distance':float(oval_distance.max()),
            'largest_distance_indices':[int(oval[i]) for i in np.argsort(oval_distance)[-6:][::-1]],
            'fixed_anatomical_curve_not_view_dependent_silhouette':True},
        'lip_shape_ratios':{'canonical_native':shape_metrics(canonical,1,lambda x:x[2]),
            'makehuman_mapped':shape_metrics(points,2,lambda x:-x[1])},
        'source_material_region':{'uv_path':str(uv_path),'sha256':sha(uv_path),'faces':len(uv),
            'boundary_vertex_count':len(uv_boundary),'is_verified_vermilion_boundary':False,
            'current_outer_loop_on_uv_boundary':sum(i in uv_boundary for i in mouth['mouth_outer']['source_loop']),
            'warning':'UV region is a material mask spanning perioral and oral areas; not ground-truth anatomy.'},
        'source_meridian_profiles':profiles,
        'recommendation':'Leave mapping unapproved. Classify canonical mouth-cap diagnostics separately; obtain reviewed vermilion-border annotation before relocating outer lip anchors.'}
    output.mkdir(parents=True)
    (output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez(output/'audit-geometry.npz',canonical_aligned=aligned,canonical_original=canonical,canonical_triangles=ct,
        mapped_points=points,source_vertices=source_vertices,source_triangles=triangles,source_vertex_ids=source_ids)
    (output/'display-regions.json').write_text(json.dumps({'region_edges':mapping['region_edges'],'warning_triangles':[w['canonical_triangle_id'] for w in warnings],
        'mouth_caps':caps,'source_outer_loop':mouth['mouth_outer']['source_loop'],'source_inner_loop':mouth['mouth_inner']['source_loop']},indent=2)+'\n')
    if sha(mapping_path)!=before:raise RuntimeError('Input mapping changed during audit')
    return report


def profiles(output):
    output=Path(output);r=json.loads((output/'audit.json').read_text());g=np.load(output/'audit-geometry.npz')
    try:font=ImageFont.truetype('DejaVuSans.ttf',20)
    except OSError:font=ImageFont.load_default()
    image=Image.new('RGB',(1600,1050),'white');d=ImageDraw.Draw(image)
    # One physical scale and common frame; x is anterior, y is superior. Not pixel fitting.
    center=(g['mapped_points'][13]+g['mapped_points'][14])/2
    mid=[0,11,12,13,14,15,16,17,18]
    canon=g['canonical_aligned'][mid];upper=np.array([p['world_xyz'] for p in r['source_meridian_profiles']['upper']]);lower=np.array([p['world_xyz'] for p in r['source_meridian_profiles']['lower']])
    arrays=[canon,upper,lower];data=np.vstack(arrays);xx=-data[:,1];yy=data[:,2]
    span=max(np.ptp(xx),np.ptp(yy))+.012;cx=float((-center[1]));cy=float(center[2]);scale=800/span
    def pixel(p):return (800+(-p[1]-cx)*scale,540-(p[2]-cy)*scale)
    d.text((30,18),'3D sagittal mouth profile: depth right = forward; height up; equal axis scale',fill='black',font=font)
    d.text((30,48),'Canonical: similarity only, no local shape warp. MakeHuman: actual surface meridian vertices.',fill='black',font=font)
    d.line((100,540,1500,540),fill=(210,210,210),width=1);d.line((800,110,800,970),fill=(210,210,210),width=1)
    for points,color,labels in [(canon,(20,80,200),[f'MP {i}' for i in mid]),(upper,(180,40,150),[str(p['source_vertex_id']) for p in r['source_meridian_profiles']['upper']]),(lower,(0,130,60),[str(p['source_vertex_id']) for p in r['source_meridian_profiles']['lower']])]:
        xy=[pixel(p) for p in points];d.line(xy,fill=color,width=3)
        for (x,y),label in zip(xy,labels):d.ellipse((x-4,y-4,x+4,y+4),fill=color);d.text((x+7,y-10),label,font=font,fill=color)
    d.text((30,990),'Blue: canonical centerline | Purple: source upper lip | Green: source lower lip | Not anatomical approval',font=font,fill='black')
    image.save(output/'sagittal-profile.png')


def comparison_figures(output):
    output=Path(output);meta=json.loads((output/'render-views.json').read_text())
    try:font=ImageFont.truetype('DejaVuSans.ttf',24)
    except OSError:font=ImageFont.load_default()
    sheet=Image.new('RGB',(1600,4*850),(245,245,245));warnings=json.loads((output/'audit.json').read_text())['normal_warnings']
    for row,view in enumerate(('front','oblique','right-profile','left-profile')):
        pair=Image.new('RGB',(2048,1100),'white')
        for col,model in enumerate(('canonical','makehuman')):
            entry=next(v for v in meta['views'] if v['model']==model and v['view']==view)
            with Image.open(output/entry['file']) as opened:im=opened.convert('RGB')
            draw=ImageDraw.Draw(im);xy=np.array(entry['pixel_xy'])
            ids=[0,13,14,17,61,291,10,152,234,454] if view in ('front','oblique') else [0,13,14,17,10,152]
            for i in ids:
                x,y=xy[i];draw.ellipse((x-3,y-3,x+3,y+3),fill='yellow')
                draw.text((x+6,y-14),str(i),font=font,fill='yellow',stroke_width=1,stroke_fill='black')
            pair.paste(im,(col*1024,76))
            title='CANONICAL: similarity only; orange = mouth cap' if model=='canonical' else 'MAKEHUMAN: unchanged surface anchors'
            ImageDraw.Draw(pair).text((col*1024+12,8),title,font=font,fill='black')
            ImageDraw.Draw(pair).text((col*1024+12,40),view+' | same camera/scale | unvalidated',font=font,fill='black')
        pair.save(output/('comparison-'+view+'.png'))
        sheet.paste(pair.resize((1600,850)),(0,row*850))
    sheet.save(output/'comparison-sheet.png')
    # Show each warned cap triangle with ALL its original indices; lines are diagnostics, not source faces.
    panels=[]
    for model in ('canonical','makehuman'):
        entry=next(v for v in meta['views'] if v['model']==model and v['view']=='front')
        with Image.open(output/entry['file']) as opened:im=opened.convert('RGB')
        xy=np.array(entry['pixel_xy']);ids=sorted({i for w in warnings for i in w['landmarks']}|{13,14})
        p=xy[ids];left,top=np.floor(p.min(0)-35).astype(int);right,bottom=np.ceil(p.max(0)+35).astype(int)
        crop=im.crop((left,top,right,bottom)).resize(((right-left)*4,(bottom-top)*4))
        draw=ImageDraw.Draw(crop)
        for w in warnings:
            poly=[tuple((xy[i]-[left,top])*4) for i in w['landmarks']]
            draw.line(poly+[poly[0]],fill=(255,160,20),width=2)
        for i in ids:
            x,y=(xy[i]-[left,top])*4;draw.text((x+3,y-12),str(i),font=font,fill='yellow',stroke_width=1,stroke_fill='black')
        panels.append(crop)
    width=max(im.width for im in panels);height=max(im.height for im in panels)
    figure=Image.new('RGB',(width*2,height+80),'white');draw=ImageDraw.Draw(figure)
    for col,(model,im) in enumerate(zip(('Canonical cap faces','MakeHuman cap CHORDS: no source faces'),panels)):
        draw.text((col*width+8,8),model,font=font,fill='black');figure.paste(im,(col*width,70))
    figure.save(output/'six-cap-warnings.png')


def source_topology_figure(output,asset_root):
    from scripts.anatomical_constraints import trace_quad_loop
    output=Path(output);g=np.load(output/'audit-geometry.npz');v=g['source_vertices'];ids=g['source_vertex_ids'];lookup={int(i):j for j,i in enumerate(ids)}
    quads=body_quads((Path(asset_root)/'data/3dobjs/base.obj').read_text())
    loops=[trace_quad_loop(quads,*seed) for seed in [(362,357),(455,454),(466,463),(467,464),(509,508),(468,465),(474,473)]]
    palette=[(180,0,180),(50,100,180),(0,140,160),(180,140,0),(110,80,160),(20,130,50),(220,30,30)]
    try:font=ImageFont.truetype('DejaVuSans.ttf',20)
    except OSError:font=ImageFont.load_default()
    image=Image.new('RGB',(1800,1000),'white');draw=ImageDraw.Draw(image)
    draw.text((20,10),'Actual MakeHuman quad rings: selected outer = purple; inner rim = red. No shading/pixel fitting.',font=font,fill='black')
    center_z=.6605;scale=11000
    for column,axes in enumerate(((0,2),(1,2))):
        title='Frontal source geometry (x vs height)' if column==0 else 'Sagittal geometry (forward depth vs height)'
        draw.text((column*900+30,50),title,font=font,fill='black')
        def pixel(p):
            x=p[0] if column==0 else -p[1]-.15
            return (column*900+450+x*scale,530-(p[2]-center_z)*scale)
        # Actual source mesh edges near the mouth, not generated canonical edges.
        for q in quads:
            points=v[[lookup[i] for i in q]]
            if np.all(np.abs(points[:,0])<.04) and np.all((points[:,2]>.636)&(points[:,2]<.684)) and np.all(points[:,1]<-.13):
                poly=[pixel(p) for p in points];draw.line(poly+[poly[0]],fill=(220,220,220),width=1)
        for number,loop in enumerate(loops):
            points=v[[lookup[i] for i in loop]];poly=[pixel(p) for p in points]
            draw.line(poly+[poly[0]],fill=palette[number],width=3)
            for i in loop:
                if abs(v[lookup[i],0])<1e-7:
                    x,y=pixel(v[lookup[i]]);draw.ellipse((x-3,y-3,x+3,y+3),fill=palette[number]);draw.text((x+7,y-10),str(i),font=font,fill=palette[number])
    draw.text((20,940),'Source vertex IDs label meridian points. A regular quad ring is not proof of a vermilion boundary.',font=font,fill='black')
    image.save(output/'source-mouth-topology.png')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--figures-only',action='store_true')
    p.add_argument('--mapping',type=Path,default=Path('runs/veyra3d/correspondence-anatomical-final'))
    p.add_argument('--prototype',type=Path,default=Path('runs/veyra3d/blender-prototype-002'))
    p.add_argument('--reference',type=Path,default=Path('third_party/mediapipe-reference'))
    p.add_argument('--baseline',type=Path,default=Path('runs/veyra3d/correspondence-002'))
    p.add_argument('--asset-root',type=Path,default=None,help='MPFB installation root (required outside the Windows default location)')
    p.add_argument('--output',type=Path,default=Path('runs/veyra3d/anatomy-audit-001'))
    a=p.parse_args()
    if a.asset_root is None:
        appdata=os.environ.get('APPDATA')
        if not appdata:p.error('--asset-root is required on this platform')
        a.asset_root=Path(appdata)/'Blender Foundation/Blender/5.3/extensions/blender_org/mpfb'
    if a.figures_only:
        profiles(a.output);comparison_figures(a.output);source_topology_figure(a.output,a.asset_root);raise SystemExit(0)
    r=audit(a.mapping,a.prototype,a.reference,a.baseline,a.asset_root,a.output);profiles(a.output)
    print(json.dumps({'warnings_investigated':len(r['normal_warnings']),'veyra_skin_metrics':r['veyra_skin_metrics'],'mapping_unchanged':r['mapping_unchanged'],'lip_shape_ratios':r['lip_shape_ratios']},indent=2))

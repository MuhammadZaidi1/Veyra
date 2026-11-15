# SPDX-License-Identifier: GPL-3.0-or-later
"""Blender-only, bounded candidate mapping and deformation REVIEW; no training export."""
import argparse
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from bpy_extras.object_utils import world_to_camera_view
from blender_face_prototype import find_mpfb,point_at
from anatomical_constraints import body_quads,feature_constraints,harmonic_warp,preserve_parameter_orientation,lip_ribbon_constraints,enclosed_quad_patch
from correspondence_math import parse_obj,fit_similarity,radial_warp,barycentric,evaluate_anchors,deformation_metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--registration',choices=['biharmonic','polyharmonic'],default='polyharmonic')
    parser.add_argument('--projection',choices=['closest','parameter-ray'],default='closest')
    parser.add_argument('--geometry-only',action='store_true')
    parser.add_argument('--fitting',choices=['legacy','anatomical'],default='anatomical')
    parser.add_argument('--resolution',type=int,choices=[512,1024,1536],default=1536)
    parser.add_argument('--prototype',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--anchors',type=Path,help='Optional manually edited candidate map; always rechecked/unvalidated')
    parser.add_argument('--controls',type=Path,help='Optional edited coarse-control JSON, never auto-approved')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    if args.output.exists():raise ValueError('Choose a fresh output directory')
    source=json.loads((args.prototype/'report.json').read_text())
    draft=json.loads((args.prototype/'mapping-draft.json').read_text())
    archive=np.load(args.prototype/'surface-review.npz',allow_pickle=False)
    vertices=archive['world_xyz'];triangles=archive['triangles'];source_ids=archive['source_vertex_ids']
    topology_hash=hashlib.sha256(triangles.tobytes()).hexdigest()
    if draft['topology_sha256']!=topology_hash:raise ValueError('Prototype topology hash mismatch')
    provenance=json.loads((args.reference/'provenance.json').read_text())
    for filename,meta in provenance['files'].items():
        if hashlib.sha256((args.reference/filename).read_bytes()).hexdigest()!=meta['sha256']:
            raise ValueError('Reference checksum mismatch: '+filename)
    canonical,canonical_triangles=parse_obj((args.reference/'canonical_face_model.obj').read_text())
    # Fixed axis conversion: canonical x right,y up,z forward -> MH x right,y back,z up.
    oriented=canonical[:,[0,2,1]]*np.array([1,-1,1])
    bpy.ops.wm.open_mainfile(filepath=str((args.prototype/'inspection.blend').resolve()))
    module,root,asset=find_mpfb()
    if hashlib.sha256(asset.read_bytes()).hexdigest()!=draft['asset_sha256']:raise ValueError('MPFB asset changed')
    obj=bpy.data.objects['CC0_MakeHuman_base']
    actual=np.array([tuple(obj.matrix_world@v.co) for v in obj.data.vertices])
    if not np.allclose(actual,vertices,atol=1e-8):raise ValueError('Scene/review geometry mismatch')
    stable_ids=np.empty(len(vertices),dtype=np.int32)
    obj.data.attributes['makehuman_source_vertex_id'].data.foreach_get('value',stable_ids)
    if not np.array_equal(stable_ids,source_ids):raise ValueError('Source IDs changed')
    # Thirteen coarse anatomical guesses grounded in local skin locations; all require review.
    seeds={1:(0,-.168,.691),152:(0,-.105,.608),10:(0,-.105,.819),
        33:(-.045,-.14,.731),133:(-.016,-.14,.73),263:(.045,-.14,.731),362:(.016,-.14,.73),
        61:(-.026,-.15,.667),291:(.026,-.15,.667),
        0:(0,-.15973,.6696),13:(0,-.16012,.66712),14:(0,-.15887,.66491),17:(0,-.1553,.66081)}
    head=(vertices[:,2]>.59)&(vertices[:,2]<.86)&(np.abs(vertices[:,0])<.12)&(vertices[:,1]<.01)
    eligible=np.flatnonzero(head)
    if args.controls:
        controls=json.loads(args.controls.read_text())
        if controls['asset_sha256']!=draft['asset_sha256']:raise ValueError('Control asset mismatch')
    else:
        controls={'asset_sha256':draft['asset_sha256'],'status':'heuristic_unreviewed','controls':[
            {'canonical_index':i,'source_vertex_id':int(source_ids[eligible[np.linalg.norm(vertices[eligible]-p,axis=1).argmin()]] ),'human_reviewed':False}
            for i,p in seeds.items()]}
    id_lookup={int(value):i for i,value in enumerate(source_ids)}
    indices=[c['canonical_index'] for c in controls['controls']]
    if len(set(indices))!=len(indices) or len(indices)<5 or min(indices)<0 or max(indices)>=468:raise ValueError('Invalid alignment controls')
    target=np.array([vertices[id_lookup[c['source_vertex_id']]] for c in controls['controls']])
    scale,rotation,translation=fit_similarity(oriented[indices],target)
    aligned=scale*oriented@rotation+translation
    warped=radial_warp(aligned,aligned[indices],target)
    # Restrict nearest surface to head, not eyeball objects, body helpers or neck.
    allowed=np.flatnonzero(head[triangles].all(1))
    bvh=BVHTree.FromPolygons(vertices.tolist(),triangles[allowed].tolist(),all_triangles=True)
    slots=[];points=[]
    connections=importlib.util.spec_from_file_location('official_connections',args.reference/'face_mesh_connections.py')
    regions_module=importlib.util.module_from_spec(connections);connections.loader.exec_module(regions_module)
    region_edges={name:list(getattr(regions_module,attribute)) for name,attribute in {
        'eyes':'FACEMESH_LEFT_EYE','other_eye':'FACEMESH_RIGHT_EYE','lips':'FACEMESH_LIPS','contour':'FACEMESH_FACE_OVAL'}.items()}
    regions={name:sorted({i for e in edges for i in e}) for name,edges in region_edges.items()}
    regions['nose']=sorted({i for i in range(468) if abs(canonical[i,0])<1.8 and -2.8<canonical[i,1]<3})
    regions['jaw']=[i for i in regions['contour'] if canonical[i,1]<-2]
    sensitive=set(sum(regions.values(),[]))
    feature_anchors={};feature_records=[];orientation_report={}
    if args.fitting=='anatomical':
        feature_anchors,feature_records=feature_constraints(canonical,region_edges,vertices,triangles,source_ids,body_quads(asset.read_text()))
        ribbon,ribbon_triangles,ribbon_record=lip_ribbon_constraints(canonical,canonical_triangles,vertices,triangles,source_ids,body_quads(asset.read_text()),feature_records)
        ribbon_bvh=BVHTree.FromPolygons(vertices.tolist(),triangles[ribbon_triangles].tolist(),all_triangles=True)
        for i,entry in ribbon.items():
            location,normal,local,distance=ribbon_bvh.find_nearest(Vector(entry['point']))
            support=int(ribbon_triangles[local]);weights=barycentric(np.array(location),vertices[triangles[support]])
            feature_anchors[i]={**entry,'source_triangle':support,'barycentric':weights.tolist(),'point':weights@vertices[triangles[support]]}
        feature_records.append(ribbon_record)
        regions['lip_ribbon']=sorted(set(regions['lips'])|set(ribbon))
        sensitive.update(ribbon)
        fixed=sorted(set(feature_anchors)|{1,152,10})
        targets=np.array([feature_anchors[i]['point'] if i in feature_anchors else target[indices.index(i)] for i in fixed])
        scale,rotation,translation=fit_similarity(oriented[fixed],targets)
        aligned=scale*oriented@rotation+translation
        warped=harmonic_warp(aligned,canonical_triangles,fixed,targets) if args.registration=='biharmonic' else radial_warp(aligned,aligned[fixed],targets)
        warped,orientation_report=preserve_parameter_orientation(aligned,warped,canonical_triangles,fixed)
        controls['effective_coarse_indices_in_anatomical_mode']=[1,152,10]
        controls['feature_boundary_constraints']=72
        controls['lip_ribbon_constraints']=len(ribbon)
        # Exclude cavity-facing/back-facing triangles from dense external-skin fitting.
        support=vertices[triangles[allowed]]
        normals=np.cross(support[:,1]-support[:,0],support[:,2]-support[:,0])
        allowed=allowed[normals[:,1] < .05*np.linalg.norm(normals,axis=1)]
        quads=body_quads(asset.read_text())
        mouth_loop=next(r['source_loop'] for r in feature_records if r.get('name')=='mouth_inner')
        cavity_quads=enclosed_quad_patch(quads,mouth_loop,554)
        cavity_vertices=set(sum([quads[k] for k in cavity_quads],[]))
        allowed=np.array([k for k in allowed if not all(int(source_ids[j]) in cavity_vertices for j in triangles[k])])
        orientation_report['excluded_oral_cavity_quads']=len(cavity_quads)
        bvh=BVHTree.FromPolygons(vertices.tolist(),triangles[allowed].tolist(),all_triangles=True)
    for i,p in enumerate(warped):
        if args.fitting=='anatomical' and args.projection=='parameter-ray' and i not in feature_anchors:
            location,normal,local,ray_distance=bvh.ray_cast(Vector((p[0],-1,p[2])),Vector((0,1,0)))
            if location is None:
                location,normal,local,distance=bvh.find_nearest(Vector(p))
            else:distance=float(np.linalg.norm(np.array(location)-p))
        else:location,normal,local,distance=bvh.find_nearest(Vector(p))
        if location is None:raise ValueError('Surface projection failed')
        triangle=int(allowed[local]);weights=barycentric(np.array(location),vertices[triangles[triangle]])
        if i in feature_anchors:
            anchor=feature_anchors[i];triangle=anchor['source_triangle'];weights=np.array(anchor['barycentric'])
            location=Vector(anchor['point']);distance=float(np.linalg.norm(np.array(location)-p))
            a,b,c=vertices[triangles[triangle]];normal=Vector(np.cross(b-a,c-a))
        flags=[]
        if i in sensitive:flags.append('sensitive_region_manual_review')
        if distance>.008:flags.append('projection_distance_gt_8mm_scene_units')
        if weights.min()<.001:flags.append('triangle_boundary_anchor')
        if normal.y>0:flags.append('positive_y_winding_normal_review')
        slots.append({'veyra_index':i,'source_triangle':triangle,'source_vertex_ids':source_ids[triangles[triangle]].tolist(),
            'barycentric':weights.tolist(),'projection_distance':float(distance),'review_flags':flags,'human_reviewed':False,'feature_curve':feature_anchors.get(i,{}).get('feature_curve')})
        points.append(tuple(location))
    if args.anchors:
        edited=json.loads(args.anchors.read_text())
        if edited['asset_sha256']!=draft['asset_sha256'] or edited['topology_sha256']!=topology_hash:
            raise ValueError('Edited anchors do not match prototype asset/topology')
        slots=edited['slots']
        points=evaluate_anchors(vertices,triangles,slots).tolist()
        for i,slot in enumerate(slots):
            tri=triangles[slot['source_triangle']]
            slot['source_vertex_ids']=source_ids[tri].tolist()
            slot['human_reviewed']=False  # This tool never certifies anatomical validation.
            slot['review_flags']=['sensitive_region_manual_review'] if i in sensitive else []
            distance=float(np.linalg.norm(np.array(points[i])-warped[i]))
            slot['projection_distance']=distance
            if distance>.008:slot['review_flags'].append('projection_distance_gt_8mm_scene_units')
            if min(slot['barycentric'])<.001:slot['review_flags'].append('triangle_boundary_anchor')
            a,b,c=vertices[tri]
            if np.cross(b-a,c-a)[1]>0:slot['review_flags'].append('positive_y_winding_normal_review')
    points=np.array(points)
    reconstructed=evaluate_anchors(vertices,triangles,slots)
    reconstruction_error=float(np.abs(points-reconstructed).max())
    if reconstruction_error>2e-7:raise ValueError(f'Barycentric reconstruction failed: {reconstruction_error}')
    points=reconstructed.copy()  # Use exact stored barycentric attachment, not BVH float32 roundoff.
    initial_quality=deformation_metrics(warped,points,canonical_triangles)
    for triangle_id in orientation_report.get('unresolved_parameter_triangles',[]):
        for i in canonical_triangles[triangle_id]:
            if 'parameter_orientation_requires_review' not in slots[int(i)]['review_flags']:
                slots[int(i)]['review_flags'].append('parameter_orientation_requires_review')
    for i in initial_quality['flagged_vertices']:slots[i]['review_flags'].append('projection_connectivity_distortion')
    # Distinct canonical points collapsing to the same skin point is suspicious.
    pairdist=np.linalg.norm(points[:,None]-points[None,:],axis=2);np.fill_diagonal(pairdist,np.inf)
    for i in np.flatnonzero(pairdist.min(1)<.0002):slots[int(i)]['review_flags'].append('near_duplicate_anchor_0.2mm')
    mapping={**draft,'status':'candidate_unvalidated','reviewer':None,'slots':slots,
        'canonical_reference':provenance,'regions':regions,'region_edges':region_edges,
        'alignment':{'scale':scale,'rotation':rotation.tolist(),'translation':translation.tolist(),
            'algorithm':('72 quad-loop boundaries + 40 intrinsic lip-ribbon anchors + smooth 3D warp + external skin projection' if args.fitting=='anatomical' else 'legacy polyharmonic warp + nearest head triangle'),
            'fitting_mode':args.fitting,'registration':args.registration,'projection':args.projection,
            'control_residual_max':float(np.linalg.norm(warped[fixed]-targets,axis=1).max()) if args.fitting=='anatomical' else float(np.linalg.norm(warped[indices]-target,axis=1).max())},
        'warning':'Automatic candidates only. All 468 require anatomical review. Not training ready.'}
    mapping['manual_anchor_input']=str(args.anchors.resolve()) if args.anchors else None
    args.output.mkdir(parents=True)
    (args.output/'mapping-candidates.json').write_text(json.dumps(mapping,indent=2)+'\n')
    (args.output/'anatomical-curves.json').write_text(json.dumps({'status':'unvalidated','curves':feature_records},indent=2)+'\n')
    (args.output/'alignment-controls.json').write_text(json.dumps(controls,indent=2)+'\n')
    np.savez(args.output/'alignment-review.npz',canonical_original=canonical,aligned_world=aligned,warped_world=warped,projected_world=points)
    # Read only named bundled core targets. No user expression library is scanned.
    identity_specs={'base':[], 'round':[('head/head-round.target.gz',.6)],
        'narrow':[('head/head-scale-horiz-decr.target.gz',.5)],'nose':[('nose/nose-scale-vert-incr.target.gz',.5)]}
    deformation_inputs={}
    # Targets are source-index deltas in OBJ x,y,z; MPFB import rotates (x,y,z)->(x,-z,y), scale .1.
    for name,specs in identity_specs.items():
        delta=np.zeros_like(vertices)
        for relative,weight in specs:
            path=root/'data/targets'/relative
            if not path.is_file():raise ValueError('Required core morph missing: '+str(path))
            text=gzip.decompress(path.read_bytes()).decode()
            provenance_entry={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'weight':weight,'license':'CC0 bundled core target'}
            deformation_inputs.setdefault(name,[]).append(provenance_entry)
            for line in text.splitlines():
                fields=line.split()
                if len(fields)==4 and not line.lstrip().startswith('#'):
                    source_id=int(fields[0])
                    if source_id in id_lookup:
                        d=np.array([float(x) for x in fields[1:]])
                        delta[id_lookup[source_id]]+=weight*.1*d[[0,2,1]]*np.array([1,-1,1])
        identity_specs[name]=vertices+delta
    expressions=['neutral','smile_proxy','jaw_open_proxy','blink_proxy']
    scene=bpy.context.scene;camera=scene.camera
    resolution=args.resolution
    scene.render.resolution_x=scene.render.resolution_y=resolution
    scene.cycles.samples=32
    cases=[];renders=[]
    for identity,base in identity_specs.items():
        for expression in expressions:
            varied=base.copy()
            # Authored smooth displacement tests, NOT validated FACS/MPFB expression rigs.
            x,y,z=vertices.T
            front=np.exp(-((y+.13)/.055)**2)
            if expression=='smile_proxy':
                influence=np.exp(-((z-.667)/.017)**2)*np.exp(-((np.abs(x)-.024)/.024)**2)*front
                varied[:,2]+=.006*influence;varied[:,0]+=.003*np.sign(x)*influence
            elif expression=='jaw_open_proxy':
                influence=np.exp(-(x/.055)**4)*np.clip((.667-z)/.045,0,1)*front
                varied[:,2]-=.012*influence;varied[:,1]-=.002*influence
            elif expression=='blink_proxy':
                for cx in (-.031,.031):
                    influence=np.exp(-((x-cx)/.022)**4)*np.exp(-((z-.73)/.012)**2)*front
                    varied[:,2]-=.65*(z-.73)*influence
            obj.data.vertices.foreach_set('co',varied.astype(np.float32).ravel());obj.data.update();bpy.context.view_layer.update()
            mesh_points=np.array([tuple(v.co) for v in obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices])
            carried=evaluate_anchors(mesh_points,triangles,slots)
            metrics=deformation_metrics(points,carried,canonical_triangles)
            # Barycentrics are fixed; test that support triangles remain nondegenerate and oriented.
            supports=triangles[[s['source_triangle'] for s in slots]]
            old=vertices[supports];new=mesh_points[supports]
            n0=np.cross(old[:,1]-old[:,0],old[:,2]-old[:,0]);n1=np.cross(new[:,1]-new[:,0],new[:,2]-new[:,0])
            bad=np.flatnonzero((np.linalg.norm(n1,axis=1)<1e-10)|((n0*n1).sum(1)<=0)).tolist()
            metrics.update(support_triangle_failures=len(bad),support_failure_indices=bad,
                identity=identity,expression=expression,topology_preserved=len(mesh_points)==len(vertices),
                evaluated_vertex_max_change=float(np.linalg.norm(mesh_points-vertices,axis=1).max()))
            metrics['lip_seam_vertical_gap']=float(carried[13,2]-carried[14,2])
            metrics['feature_ordering_failures']=[]
            for record in feature_records:
                if 'canonical_upper_center' not in record:continue
                upper=record['canonical_upper_center'];lower=record['canonical_lower_center']
                if carried[upper,2]<=carried[lower,2]:metrics['feature_ordering_failures'].append(record['name'])
            metrics['view_checks']=[]
            for view,position in {'front':(0,-1.1,.725),'left':(-.55,-.95,.725),'right':(.55,-.95,.725)}.items():
                camera.location=position;point_at(camera,(0,-.04,.725));bpy.context.view_layer.update()
                uvcheck=np.array([tuple(world_to_camera_view(scene,camera,Vector(p))) for p in carried])
                invcheck=camera.matrix_world.inverted()
                camcheck=np.array([tuple(invcheck@Vector(p)) for p in carried])
                pixelcheck=np.column_stack((uvcheck[:,0]*resolution-.5,(1-uvcheck[:,1])*resolution-.5))
                analyticcheck=np.column_stack(((camcheck[:,0]/camera.data.ortho_scale+.5)*resolution-.5,(.5-camcheck[:,1]/camera.data.ortho_scale)*resolution-.5))
                viewerror=float(np.abs(pixelcheck-analyticcheck).max())
                if viewerror>.001:raise ValueError('Deformation/view projection disagreement')
                metrics['view_checks'].append({'view':view,'projection_error_px':viewerror,'out_of_frame_points':int(((uvcheck[:,:2]<0)|(uvcheck[:,:2]>1)).any(1).sum())})
            cases.append(metrics)
            # One render each for base expression cases and each identity neutral (7 total).
            if identity=='base' or expression=='neutral':
                views=['front','left','right'] if identity=='base' and expression=='neutral' else ['front']
                for view in views:
                    camera.location={'front':(0,-1.1,.725),'left':(-.55,-.95,.725),'right':(.55,-.95,.725)}[view]
                    point_at(camera,(0,-.04,.725));bpy.context.view_layer.update()
                    name=f'{identity}-{expression}-{view}'
                    scene.render.filepath=str((args.output/(name+'.png')).resolve())
                    if not args.geometry_only:bpy.ops.render.render(write_still=True)
                    uv=np.array([tuple(world_to_camera_view(scene,camera,Vector(p))) for p in carried])
                    pixels=np.column_stack((uv[:,0]*resolution-.5,(1-uv[:,1])*resolution-.5))
                    inv=camera.matrix_world.inverted()
                    camera_points=np.array([tuple(inv@Vector(p)) for p in carried])
                    analytic=np.column_stack(((camera_points[:,0]/camera.data.ortho_scale+.5)*resolution-.5,
                        (.5-camera_points[:,1]/camera.data.ortho_scale)*resolution-.5))
                    projection_error=float(np.abs(pixels-analytic).max())
                    if projection_error>.001:raise ValueError('Camera projection check failed')
                    renders.append({'name':name,'pixel_xy':pixels.tolist(),'projection_error_px':projection_error,
                        'identity':identity,'expression':expression,'view':view,'resolution':resolution})
    for case in cases:
        for i in case['flagged_vertices']+case['support_failure_indices']:
            if 'deformation_instability' not in slots[i]['review_flags']:slots[i]['review_flags'].append('deformation_instability')
    (args.output/'mapping-candidates.json').write_text(json.dumps(mapping,indent=2)+'\n')
    feature_residuals={str(i):float(np.linalg.norm(points[i]-feature_anchors[i]['point'])) for i in feature_anchors}
    report={'parameter_orientation':orientation_report,'geometry_only':args.geometry_only,'feature_constraint_residuals':feature_residuals,'feature_assignment_status':'unvalidated','review_scope':'All 468 remain anatomically unvalidated; 40 additional lip-ribbon points now receive mandatory focused review', 'status':'candidate_unvalidated','training_ready':False,'candidate_anchors':468,
        'numerical_anchor_failures':0,'barycentric_max_error':reconstruction_error,
        'projection_quality':initial_quality,'flagged_candidates':sum(bool(s['review_flags']) for s in slots),
        'flag_counts':{flag:sum(flag in s['review_flags'] for s in slots) for flag in sorted({f for s in slots for f in s['review_flags']})},
        'cases':cases,'renders':renders,'deformation_sources':deformation_inputs,
        'expression_scope':'Authored smooth expression proxies, not validated facial rig expressions',
        'anatomical_validation':'NOT PERFORMED; all 468 need human review',
        'reference_revision':provenance['revision']}
    (args.output/'review-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('VEYRA_CANDIDATES',json.dumps({k:report[k] for k in ['candidate_anchors','numerical_anchor_failures','flagged_candidates','flag_counts']}))


if __name__=='__main__':main()

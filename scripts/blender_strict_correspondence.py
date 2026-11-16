# SPDX-License-Identifier: GPL-3.0-or-later
"""Semantic regional proposals with hard rejection and a native-topology feasibility study.

Everything is diagnostic. No anatomy acceptance, training arrays or model changes.
"""
import argparse,copy,gzip,json,shutil,sys,subprocess
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1])]
import bpy,numpy as np
from mathutils import Vector,Matrix
from mathutils.bvhtree import BVHTree
from bpy_extras.object_utils import world_to_camera_view
from scripts.makehuman_semantics import CoreTargets,rig_inventory,sha
from scripts.correspondence_gate import structural_checks,support_checks,quad_graph,shortest_mesh_path,semantic_decision
from scripts.correspondence_math import evaluate_anchors,barycentric,parse_obj
from scripts.anatomical_constraints import body_quads,trace_quad_loop,enclosed_quad_patch,paths_between,segment_sample
from scripts.surface_proposal_math import ordered_loop
from veyra.topology_data import TRIANGLE_INDICES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['project','prototype','proposal','reference','audit','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);out=args.output.resolve()
    if out.exists():raise ValueError('Output must be fresh')
    repo=Path(__file__).resolve().parents[1];project=args.project.resolve();prototype=args.prototype.resolve();previous=args.proposal.resolve()
    roots=[project,prototype,previous,args.audit.resolve(),Path(json.loads((project/'project.json').read_text())['original_mapping']).parent]
    preserved={str(p.resolve()):sha(p) for root in roots for p in root.rglob('*') if p.is_file()}
    preserved.update({str(p.resolve()):sha(p) for folder in ['scripts','docs','veyra3d','tests_research','third_party'] for p in (repo/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts})
    baseline_bytes=(project/'base-map.json').read_bytes();base=json.loads(baseline_bytes);config=json.loads((project/'project.json').read_text());original_hash=sha(project/'base-map.json')
    if config['base_mapping_sha256']!=original_hash or config['surface_sha256']!=sha(project/'surface.npz'):raise ValueError('Frozen project hashes changed')
    if json.loads((previous/'proposal-report.json').read_text())['original_mapping_sha256']!=original_hash:raise ValueError('Failed proposal does not use this original baseline')
    old_proposal=json.loads((previous/'mapping-proposals.json').read_text());records=json.loads((project/'annotations/draft-0000.json').read_text())['annotations'];editable=[r['veyra_index'] for r in records];groups={r['veyra_index']:r['annotation_group'] for r in records}
    if len(set(editable))!=56:raise ValueError('Expected 56 distinct disputed indices')
    g=np.load(project/'surface.npz',allow_pickle=False);vertices,triangles,source_ids=g['world_xyz'],g['triangles'],g['source_vertex_ids'];baseline=evaluate_anchors(vertices,triangles,base['slots']);lookup={int(i):j for j,i in enumerate(source_ids)}
    import hashlib
    if hashlib.sha256(triangles.astype(np.int32).tobytes()).hexdigest()!=base['topology_sha256']:raise ValueError('Topology hash mismatch')
    root=Path(json.loads((prototype/'report.json').read_text())['mpfb_path']);asset=root/'data/3dobjs/base.obj'
    if sha(asset)!=base['asset_sha256'] or 'explicitly released as CC0' not in asset.read_text()[:2000]:raise ValueError('Expected verified CC0 core asset')
    provenance=json.loads((args.reference/'provenance.json').read_text())
    if any(sha(args.reference/name)!=entry['sha256'] for name,entry in provenance['files'].items()):raise ValueError('Pinned reference changed')
    canonical,_=parse_obj((args.reference/'canonical_face_model.obj').read_text());inventory=rig_inventory(root,vertices,source_ids);hints={r['name']:np.array(r['position']) for r in inventory['rig_endpoints']}
    quads=body_quads(asset.read_text());inner=trace_quad_loop(quads,474,473);cavity=enclosed_quad_patch(quads,inner,554);cavity_ids=set(sum([quads[i] for i in cavity],[]));interior_ids=cavity_ids-set(inner)
    head=(vertices[:,2]>.60)&(vertices[:,2]<.84)&(abs(vertices[:,0])<.10)&(vertices[:,1]<.005)
    f=vertices[triangles];normals=np.cross(f[:,1]-f[:,0],f[:,2]-f[:,0]);eligible=np.flatnonzero(head[triangles].all(1)&(normals[:,1]<.05*np.linalg.norm(normals,axis=1)))
    eligible=np.array([i for i in eligible if not all(int(source_ids[j]) in cavity_ids for j in triangles[i])])
    bvh=BVHTree.FromPolygons(vertices.tolist(),triangles[eligible].tolist(),all_triangles=True)
    graph_mask=head&np.array([int(i) not in interior_ids for i in source_ids]);graph=quad_graph(vertices,quads,source_ids,graph_mask);nodes=np.array(sorted(graph))
    endpoints={}
    for name,hint in hints.items():
        p,n,local,distance=bvh.find_nearest(Vector(hint));index=int(eligible[local]);weights=barycentric(np.array(p),vertices[triangles[index]])
        endpoints[name]={'source_triangle':index,'barycentric':weights.tolist(),'source_vertex_ids':source_ids[triangles[index]].tolist(),'projection_distance':float(distance),'strategy':'projected rig reference, not a certified facial landmark'}
    lip_ids=[i for i in editable if groups[i].startswith('lip')];outer=ordered_loop([e for e in base['region_edges']['lips'] if set(e)<=set(lip_ids)],0);upper,lower=paths_between(outer,61,291)
    if 0 not in upper:upper,lower=lower,upper
    oval=ordered_loop(base['region_edges']['contour'],10);skin=np.array(TRIANGLE_INDICES);edge_support={}
    for index,tri in enumerate(triangles):
        for a,b in [(tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])]:edge_support.setdefault(tuple(sorted((int(a),int(b)))),[]).append(index)
    def path_for(names):
        seeds=[int(nodes[np.argmin(np.linalg.norm(vertices[nodes]-hints[name],axis=1))]) for name in names];path=[]
        for a,b in zip(seeds,seeds[1:]):
            segment=shortest_mesh_path(graph,a,b);path+=segment if not path else segment[1:]
        if len(set(path))!=len(path):raise ValueError('Regional shortest paths revisit source vertices')
        return path
    def sample_path(path,fraction):
        k,w=segment_sample(vertices[path],fraction);w=np.maximum(w,0);w/=w.sum();a,b=path[k:k+2];options=edge_support[tuple(sorted((a,b)))];support=min(options,key=lambda j:float(normals[j,1]/max(np.linalg.norm(normals[j]),1e-15)));weights=np.zeros(3)
        for endpoint,weight in zip([a,b],w):weights[list(triangles[support]).index(endpoint)]=weight
        return {'source_triangle':int(support),'source_vertex_ids':source_ids[triangles[support]].tolist(),'barycentric':weights.tolist()}
    family_slots={};family_info={};geodesic_paths={}
    def family(name,slots,description):
        candidate=copy.deepcopy(base['slots'])
        for i,slot in slots.items():candidate[i].update(copy.deepcopy(slot));candidate[i]['human_reviewed']=False
        family_slots[name]=candidate;family_info[name]={'description':description,'requested_indices':sorted(slots),'exact_semantic_evidence':False}
    def add_curve(name,indices,names):
        path=path_for(names);geodesic_paths[name]={'rig_hints':names,'source_vertex_ids':source_ids[path].tolist()};p=canonical[indices];distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];return {i:sample_path(path,float(x/distance[-1])) for i,x in zip(indices,distance)}
    family('previous_full',{i:old_proposal['slots'][i] for i in editable},'Replay the failed proposal against original baseline, never as a reference')
    family('previous_pair',{i:old_proposal['slots'][i] for i in [84,314]},'Earlier local geometric support pair; semantic support not established')
    generated={}
    try:
        lip={}
        for label,arc,center,side in [('upper',upper,0,'T'),('lower',lower,17,'B')]:
            middle=arc.index(center)
            lip.update(add_curve('lip_'+label+'_negative',arc[:middle+1],[f'lip.{side}.R.001.tail',f'lip.{side}.R.tail',f'lip.{side}.R.head']))
            lip.update(add_curve('lip_'+label+'_positive',arc[middle:],[f'lip.{side}.L.head',f'lip.{side}.L.tail',f'lip.{side}.L.001.tail']))
        family('rig_lips',lip,'Quad-edge geodesics constrained by author-defined lip rig references, not neighboring rings');generated.update(lip)
        family('rig_corners',{i:lip[i] for i in [61,291]},'Shared upper/lower rig-chain junctions; projected means remain ambiguous')
    except ValueError as error:family_info['rig_lips']={'generation_failure':str(error),'exact_semantic_evidence':False}
    # A separate endpoint-only hypothesis exposes the ambiguity if intermediate rig controls backtrack.
    # It does not silently relax the failed waypoint family's constraints.
    try:
        direct={}
        for label,arc,center,side in [('upper',upper,0,'T'),('lower',lower,17,'B')]:
            middle=arc.index(center)
            direct.update(add_curve('lip_direct_'+label+'_negative',arc[:middle+1],[f'lip.{side}.R.001.tail',f'lip.{side}.R.head']))
            direct.update(add_curve('lip_direct_'+label+'_positive',arc[middle:],[f'lip.{side}.L.head',f'lip.{side}.L.001.tail']))
        family('rig_lips_endpoints',direct,'Endpoint-only quad-edge paths, evaluated separately from the stronger waypoint constraints');generated.update(direct)
        family('rig_corners',{i:direct[i] for i in [61,291]},'Shared lip rig junctions; not a certified surface commissure annotation')
    except ValueError as error:family_info['rig_lips_endpoints']={'generation_failure':str(error),'exact_semantic_evidence':False}
    try:
        forehead=sorted([i for i in editable if groups[i]=='forehead'],key=lambda i:canonical[i,0]);slots=add_curve('forehead',forehead,['forehead.R.002.head','forehead.R.001.head','forehead.R.head','forehead.L.head','forehead.L.001.head','forehead.L.002.head']);family('rig_forehead',slots,'Forehead rig arc is a region hint, not a canonical forehead boundary');generated.update(slots)
    except ValueError as error:family_info['rig_forehead']={'generation_failure':str(error),'exact_semantic_evidence':False}
    try:
        arcs=paths_between(oval,58,288);jaw=next(a for a in arcs if 152 in a);slots=add_curve('jaw_chin',jaw,['jaw.R.head','jaw.R.tail','chin.R.head','jaw.head','chin.L.head','jaw.L.tail','jaw.L.head']);family('rig_jaw_chin',slots,'Native mandibular/chin references; not a view-dependent silhouette');generated.update(slots)
    except ValueError as error:family_info['rig_jaw_chin']={'generation_failure':str(error),'exact_semantic_evidence':False}
    try:
        arcs=paths_between(oval,58,288);jaw=next(a for a in arcs if 152 in a)
        slots=add_curve('jaw_endpoints',jaw,['jaw.R.head','jaw.head','jaw.L.head'])
        family('rig_jaw_endpoints',slots,'Separate jaw endpoint-only hypothesis; does not waive the failed full rig-waypoint constraint');generated.update(slots)
    except ValueError as error:family_info['rig_jaw_endpoints']={'generation_failure':str(error),'exact_semantic_evidence':False}
    try:
        slots={}
        for side,indices in [('R',[21,162,127,234,93]),('L',[251,389,356,454,323])]:slots.update(add_curve('temple_'+side,indices,[f'forehead.{side}.002.head',f'temple.{side}.head',f'temple.{side}.tail']))
        family('rig_temples',slots,'Temporal geodesics use rig references, not canonical width/height ratios');generated.update(slots)
    except ValueError as error:family_info['rig_temples']={'generation_failure':str(error),'exact_semantic_evidence':False}
    if generated:family('rig_combined',generated,'Combined independently generated regional hypotheses; full dense topology remains constrained')
    # Test each new position alone too, rather than rejecting good local geometry only by association.
    for i,slot in generated.items():family('rig_point_'+str(i),{i:slot},'Single-point diagnostic of the corresponding regional source-surface hypothesis')
    print('REGIONAL_GENERATION',json.dumps({k:v['generation_failure'] for k,v in family_info.items() if 'generation_failure' in v}),flush=True)
    targets=CoreTargets(root,source_ids);cases=targets.cases(vertices)
    for case in cases:case['baseline_points']=evaluate_anchors(case['vertices'],triangles,base['slots'])
    baseline_cases=[]
    for case in cases:
        check=structural_checks(baseline,case['baseline_points'],skin,[upper,lower],[outer,oval]);check['support_failure_indices']=support_checks(vertices,case['vertices'],triangles,base['slots']);check['passed'] &= not check['support_failure_indices'];baseline_cases.append({'case':case['name'],**check})
    fixed=np.load(args.audit/'audit-geometry.npz',allow_pickle=False)['canonical_aligned']
    baseline_reference=structural_checks(fixed,baseline,skin,[upper,lower],[outer,oval]);evaluations={};coordinates={}
    for serial,(name,slots) in enumerate(family_slots.items()):
        points=evaluate_anchors(vertices,triangles,slots);coordinates[name]=points;checks=[];absolute_checks=[]
        for case in cases:
            carried=evaluate_anchors(case['vertices'],triangles,slots);check=structural_checks(case['baseline_points'],carried,skin,[upper,lower],[outer,oval]);support=support_checks(vertices,case['vertices'],triangles,slots);check['support_failure_indices']=support;check['passed'] &= not support;checks.append({'case':case['name'],**check});absolute_checks.append({'case':case['name'],**structural_checks(points,carried,skin,[upper,lower],[outer,oval])})
        reference_check=structural_checks(fixed,points,skin,[upper,lower],[outer,oval]);incremental_passed=all(c['passed'] for c in checks) and reference_check['passed'];passed=incremental_passed and all(c['passed'] for c in absolute_checks)
        evaluations[name]={'no_regression_passed':bool(incremental_passed),'absolute_case_checks':absolute_checks,'structurally_admissible':bool(passed),'decision':semantic_decision(passed,family_info[name]['exact_semantic_evidence']),'baseline_case_failures':[c['case'] for c in baseline_cases if not c['passed']],'case_checks':checks,'fixed_reference_check':reference_check,'changed_indices':[i for i in editable if np.linalg.norm(points[i]-baseline[i])>1e-10]}
        if serial%10==0:print('STRICT_CHECK',serial+1,'/',len(family_slots),flush=True)
    # Evaluate a native face patch using actual source faces. No Delaunay triangulation or MediaPipe relabeling.
    patch_triangles=triangles[eligible];patch_vertices=np.unique(patch_triangles);remap={int(i):j for j,i in enumerate(patch_vertices)};patch_faces=np.array([[remap[int(i)] for i in face] for face in patch_triangles]);native_checks=[]
    for case in cases:
        # Native diagnostics must not append stale candidate geometry to the
        # last regional family's already recorded absolute-case evidence.
        check=structural_checks(vertices[patch_vertices],case['vertices'][patch_vertices],patch_faces);native_checks.append({'case':case['name'],**check})
    native={'schema':'veyra-native-face-topology-evaluation-v1','status':'evaluation_only_not_training_labels','source_vertex_ids':source_ids[patch_vertices].tolist(),'source_triangle_ids':eligible.tolist(),'triangles':patch_faces.tolist(),'selection_rule':'neutral z .60-.84, abs(x)<.10, y<.005; outward-facing triangles; enclosed oral-cavity quads excluded; fixed before morph tests','semantic_scope':'exact native vertex identity, no MediaPipe anatomical equivalence','case_checks':native_checks,'rig_support_source_ids':inventory['native_rig_support_source_ids'],'requires_new_model_and_geometry_contract':True}
    bpy.ops.wm.open_mainfile(filepath=str(project/'annotation.blend'));scene=bpy.context.scene;obj=bpy.data.objects['CC0_MakeHuman_base'];actual=np.array([tuple(obj.matrix_world@v.co) for v in obj.data.vertices]);obj.data.calc_loop_triangles()
    if not np.allclose(actual,vertices,rtol=0,atol=1e-8) or not np.array_equal(np.array([tuple(t.vertices) for t in obj.data.loop_triangles]),triangles):raise ValueError('Blender frozen geometry mismatch')
    views=json.loads((project/'views.json').read_text());view_checks=[]
    for case in cases:
        for view in views:
            scene.camera.matrix_world=Matrix(view['matrix']);bpy.context.view_layer.update();uv=np.array([tuple(world_to_camera_view(scene,scene.camera,Vector(p))) for p in case['baseline_points']]);camera=np.c_[case['baseline_points'],np.ones(468)]@np.linalg.inv(np.array(view['matrix'])).T;analytic=np.c_[camera[:,0]/view['width']+.5,.5+camera[:,1]/view['width']];error=float(abs(uv[:,:2]-analytic).max()*view['resolution'])
            if error>.001:raise ValueError('Projection disagreement')
            view_checks.append({'case':case['name'],'view':view['name'],'max_error_px':error})
    # No exact anatomical source correspondence has been verified, so retain every baseline anchor.
    reliable=[name for name,result in evaluations.items() if result['decision']=='supported_candidate_not_anatomically_approved']
    if reliable:raise RuntimeError('Unexpected semantic approval; this study has no verified exact correspondence source')
    candidate_view_checks=[]
    for view in views:
        scene.camera.matrix_world=Matrix(view['matrix']);bpy.context.view_layer.update()
        for name,points in coordinates.items():
            uv=np.array([tuple(world_to_camera_view(scene,scene.camera,Vector(q))) for q in points[editable]]);cam=np.c_[points[editable],np.ones(56)]@np.linalg.inv(np.array(view['matrix'])).T;analytical=np.c_[cam[:,0]/view['width']+.5,.5+cam[:,1]/view['width']];error=float(abs(analytical-uv[:,:2]).max()*view['resolution'])
            if error>.001:raise ValueError('Candidate camera mismatch')
            candidate_view_checks.append({'family':name,'view':view['name'],'max_error_px':error})
    point_records=[]
    for i in editable:
        entries=[]
        for name,result in evaluations.items():
            if i not in family_info[name]['requested_indices']:continue
            slot=family_slots[name][i];entries.append({'family':name,'source_triangle':slot['source_triangle'],'barycentric':slot['barycentric'],'source_vertex_ids':slot['source_vertex_ids'],'displacement':float(np.linalg.norm(coordinates[name][i]-baseline[i])),'structurally_admissible':result['structurally_admissible'],'no_regression_passed':result['no_regression_passed'],'absolute_failure_cases':[c['case'] for c in result['absolute_case_checks'] if not c['passed']],'decision':result['decision'],'failure_cases':[c['case'] for c in result['case_checks'] if not c['passed']]})
        point_records.append({'veyra_index':i,'region':groups[i],'decision':'unchanged_unresolved','exact_semantic_evidence':False,'reason':'Rig references/morph support/curvature do not specify the exact MediaPipe anatomical site; structurally unsafe trials rejected','candidates':entries,'retained_anchor':base['slots'][i]})
    out.mkdir(parents=True);(out/'mapping-retained.json').write_bytes(baseline_bytes);(out/'semantic-inventory.json').write_text(json.dumps(inventory,indent=2));(out/'native-topology-evaluation.json').write_text(json.dumps(native,indent=2));(out/'point-evidence.json').write_text(json.dumps(point_records,indent=2));(out/'regional-paths.json').write_text(json.dumps(geodesic_paths,indent=2))
    full_proposals={name:{'status':'rejected_or_semantically_unresolved_not_training_ready','slots':slots} for name,slots in family_slots.items() if not name.startswith('rig_point_')};(out/'tested-candidates.json').write_text(json.dumps(full_proposals,indent=2))
    native_requests={'status':'diagnostic_configuration_screen_only_not_training_approval','accepted_for_geometry_inspection':[c['name'] for c,check in zip(cases,native_checks) if check['passed']],'rejected_for_geometry_inspection':[c['name'] for c,check in zip(cases,native_checks) if not check['passed']],'note':'Rejected raw morph combinations are retained in evidence, never repaired by moving labels'}
    (out/'native-configuration-screen.json').write_text(json.dumps(native_requests,indent=2))
    manifest={path:digest for path,digest in preserved.items()};changed=[path for path,digest in manifest.items() if sha(path)!=digest]
    if changed:raise AssertionError('Protected inputs changed: '+str(changed))
    report={'status':'strict_retention_unvalidated','baseline_mapping_sha256':original_hash,'baseline_description':'Original operational candidate baseline from checkpoint 516252e workflow; NOT anatomically accepted','github_checkpoint':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'reliably_supported_corrections':0,'unresolved_points':len(editable),'structurally_admissible_families':[name for name,e in evaluations.items() if e['structurally_admissible']],'baseline_cases':baseline_cases,'baseline_fixed_reference':baseline_reference,'families':family_info,'evaluations':evaluations,'targets':targets.provenance,'case_catalog':[{k:v for k,v in case.items() if k not in ['vertices','baseline_points']} for case in cases],'view_checks':view_checks,'candidate_view_checks':candidate_view_checks,'native_vertex_count':len(patch_vertices),'native_face_count':len(patch_faces),'native_failure_cases':[c['case'] for c in native_checks if not c['passed']],'preserved_input_count':len(manifest),'warning':'Structural admissibility is not anatomical approval. No labels or datasets exported.'}
    (out/'report.json').write_text(json.dumps(report,indent=2));(out/'preserved-inputs.json').write_text(json.dumps(manifest,indent=2));np.savez(out/'inspection-geometry.npz',baseline=baseline,retained=baseline,source_vertices=vertices,source_triangles=triangles,source_vertex_ids=source_ids,native_patch_vertices=patch_vertices,**{'candidate_'+name:p for name,p in coordinates.items() if not name.startswith('rig_point_')})
    print('STRICT_RESULT',json.dumps({k:report[k] for k in ['reliably_supported_corrections','unresolved_points','native_vertex_count','native_face_count','native_failure_cases','structurally_admissible_families']}),flush=True)

if __name__=='__main__':main()

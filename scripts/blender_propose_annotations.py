# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded evidence-ensemble proposals for 56 source-surface annotations; never accepts points."""
import argparse,copy,gzip,hashlib,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import bpy
from mathutils import Vector,Matrix
from mathutils.bvhtree import BVHTree
from bpy_extras.object_utils import world_to_camera_view
from annotation_core import rebuild
from correspondence_math import parse_obj,fit_similarity,evaluate_anchors,barycentric,deformation_metrics
from anatomical_constraints import body_quads,trace_quad_loop,paths_between,enclosed_quad_patch
from surface_proposal_math import mesh_evidence,percentile,ordered_loop,solve_chain,segment_crossings,select_curves
import blender_annotation_ui as ui


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--prototype',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--audit',type=Path);p.add_argument('--controls',type=Path,help='Versioned human annotation export: accepted surface positions are hard-pinned, not automatically approved in the new proposal');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:]);project=args.project.resolve();out=args.output.resolve()
    if out.exists():raise ValueError('Use a fresh output; originals are never overwritten')
    config=json.loads((project/'project.json').read_text());original_hash=sha(project/'base-map.json')
    if original_hash!=config['base_mapping_sha256'] or sha(project/'surface.npz')!=config['surface_sha256']:raise ValueError('Frozen project hash mismatch')
    base=json.loads((project/'base-map.json').read_text());session=json.loads((project/'annotations/draft-0000.json').read_text());editable=[a['veyra_index'] for a in session['annotations']];groups={a['veyra_index']:a['annotation_group'] for a in session['annotations']}
    if len(editable)!=56 or len(set(editable))!=56:raise ValueError('Expected 56 unique editable slots')
    g=np.load(project/'surface.npz',allow_pickle=False);v,t,ids=g['world_xyz'],g['triangles'],g['source_vertex_ids'];old=evaluate_anchors(v,t,base['slots'])
    if hashlib.sha256(t.astype(np.int32).tobytes()).hexdigest()!=base['topology_sha256']:raise ValueError('Topology mismatch')
    provenance=json.loads((args.reference/'provenance.json').read_text())
    for filename,meta in provenance['files'].items():
        if sha(args.reference/filename)!=meta['sha256']:raise ValueError('Reference hash mismatch')
    canonical,ct=parse_obj((args.reference/'canonical_face_model.obj').read_text());oriented=canonical[:,[0,2,1]]*[1,-1,1]
    controls=[33,133,263,362,1];scale,rotation,translation=fit_similarity(oriented[controls],old[controls]);reference=scale*oriented@rotation+translation
    root=Path(json.loads((args.prototype/'report.json').read_text())['mpfb_path']);asset=root/'data/3dobjs/base.obj'
    if sha(asset)!=base['asset_sha256'] or 'explicitly released as CC0' not in asset.read_text()[:2000]:raise ValueError('Expected verified CC0 source asset')
    pinned={}
    if args.controls:
        control_session=json.loads(args.controls.read_text());checked=rebuild(base,control_session,v,t,ids,original_hash)
        if set(checked['annotation_applied_indices'])-set(editable):raise ValueError('Controls must be among the 56 editable slots')
        pinned={i:checked['slots'][i] for i in checked['annotation_applied_indices']}
    source_lookup={int(x):i for i,x in enumerate(ids)};targets={};target_provenance={}
    def target(relative):
        if relative not in targets:
            path=root/'data/targets'/relative;delta=np.zeros_like(v)
            for line in gzip.decompress(path.read_bytes()).decode().splitlines():
                f=line.split()
                if len(f)==4 and not line.lstrip().startswith('#') and int(f[0]) in source_lookup:delta[source_lookup[int(f[0])]]=.1*np.array(list(map(float,f[1:]))) [[0,2,1]]*[1,-1,1]
            targets[relative]=delta;target_provenance[relative]={'sha256':sha(path),'scope':'installed MakeHuman bundled CC0 core target, not a segmentation label'}
        return targets[relative]
    semantic_names=['mouth/mouth-upperlip-ext-up.target.gz','mouth/mouth-upperlip-height-incr.target.gz','mouth/mouth-lowerlip-ext-down.target.gz','mouth/mouth-lowerlip-height-incr.target.gz','mouth/mouth-scale-horiz-incr.target.gz','chin/chin-height-incr.target.gz','chin/chin-width-incr.target.gz','forehead/forehead-scale-vert-incr.target.gz','forehead/forehead-temple-incr.target.gz']
    morphs=[target(x) for x in semantic_names];normals,curvature,morph_gradient,response,areas=mesh_evidence(v,t,morphs)
    quads=body_quads(asset.read_text());inner=trace_quad_loop(quads,474,473);cavity=enclosed_quad_patch(quads,inner,554);cavity_ids=set(sum([quads[k] for k in cavity],[]))
    face=v[t];n=np.cross(face[:,1]-face[:,0],face[:,2]-face[:,0]);head=(v[:,2]>.59)&(v[:,2]<.86)&(abs(v[:,0])<.12)&(v[:,1]<.015)
    allowed=np.flatnonzero(head[t].all(1)&(n[:,1]<.05*np.linalg.norm(n,axis=1)))
    allowed=np.array([k for k in allowed if not all(int(ids[j]) in cavity_ids for j in t[k])])
    # Samples are exact source vertices, edge midpoints and triangle centroids, never 2D offsets.
    weights=np.array([[1,0,0],[0,1,0],[0,0,1],[.5,.5,0],[0,.5,.5],[.5,0,.5],[1/3]*3])
    cloud=np.einsum('wj,tjk->twk',weights,v[t[allowed]]).reshape(-1,3);support=np.repeat(allowed,len(weights));barys=np.tile(weights,(len(allowed),1))
    cloud_c=np.einsum('wj,tj->tw',weights,curvature[t[allowed]]).ravel();cloud_m=np.einsum('wj,tj->tw',weights,morph_gradient[t[allowed]]).ravel()
    lip_ids=[i for i in editable if groups[i].startswith('lip')];oval=[i for i in editable if i not in lip_ids]
    inner_canonical=ordered_loop([e for e in base['region_edges']['lips'] if 13 in e or all(x not in lip_ids for x in e)],13)
    reference[lip_ids]+=old[inner_canonical].mean(0)-reference[inner_canonical].mean(0)
    loop=ordered_loop([e for e in base['region_edges']['lips'] if all(x in lip_ids for x in e)],0);upper,lower=paths_between(loop,61,291)
    if 0 not in upper:upper,lower=lower,upper
    oval_loop=ordered_loop(base['region_edges']['contour'],10)
    bvh=BVHTree.FromPolygons(v.tolist(),t[allowed].tolist(),all_triangles=True)
    # Add continuous projections of every prior, original, and interpolated hypothesis.
    for i in editable:
        for fraction in [0,.35,.65,1]:
            q=(1-fraction)*old[i]+fraction*reference[i];loc,_,local,_=bvh.find_nearest(Vector(q));tri=int(allowed[local]);w=barycentric(np.array(loc),v[t[tri]])
            cloud=np.vstack([cloud,w@v[t[tri]]]);support=np.r_[support,tri];barys=np.vstack([barys,w]);cloud_c=np.r_[cloud_c,w@curvature[t[tri]]];cloud_m=np.r_[cloud_m,w@morph_gradient[t[tri]]]
    pinned_cloud={}
    for i,slot in pinned.items():
        tri=slot['source_triangle'];w=np.array(slot['barycentric']);pinned_cloud[i]=len(cloud);cloud=np.vstack([cloud,w@v[t[tri]]]);support=np.r_[support,tri];barys=np.vstack([barys,w]);cloud_c=np.r_[cloud_c,w@curvature[t[tri]]];cloud_m=np.r_[cloud_m,w@morph_gradient[t[tri]]]
    pools={};pc={};pm={};indices={};region_indices={};ensemble=[];hypotheses=[(.35,.0,.0),(.35,.4,.4),(.35,.8,.8),(.65,.0,.0),(.65,.4,.4),(.65,.8,.8),(1.,.0,.0),(1.,.4,.4),(1.,.8,.8)]
    for i in editable:
        radius=.012 if i in lip_ids else .015
        region=np.linalg.norm(cloud-old[i],axis=1)<=radius
        if groups[i]=='lip_upper':region &= cloud[:,2]>old[13,2]+.00015
        if groups[i]=='lip_lower':region &= cloud[:,2]<old[14,2]-.00015
        if i in pinned_cloud:region[pinned_cloud[i]]=True
        eligible=np.flatnonzero(region)
        if not len(eligible):raise ValueError('Empty surface candidate region')
        c=percentile(cloud_c[eligible],cloud_c[eligible]);m=percentile(cloud_m[eligible],cloud_m[eligible]);scores=[]
        for f,cw,mw in hypotheses:
            prior=(1-f)*old[i]+f*reference[i]
            scores.append(np.sum((cloud[eligible]-prior)**2,axis=1)/.006**2+.3*np.sum((cloud[eligible]-old[i])**2,axis=1)/radius**2-cw*c-mw*m)
        # Union multiple hypotheses, deduplicated in 3D; keep 24 alternatives per point.
        candidates=np.unique(np.concatenate([np.argsort(s)[:5] for s in scores]));rank=np.min(np.array(scores)[:,candidates],axis=0);candidates=candidates[np.argsort(rank)]
        chosen=[]
        for k in candidates:
            if not chosen or np.min(np.linalg.norm(cloud[eligible[chosen]]-cloud[eligible[k]],axis=1))>1e-7:chosen.append(int(k))
            if len(chosen)==24:break
        selected=np.array([pinned_cloud[i]]) if i in pinned_cloud else eligible[chosen];chosen=[int(np.flatnonzero(eligible==pinned_cloud[i])[0])] if i in pinned_cloud else chosen;indices[i]=selected;region_indices[i]=eligible;pools[i]=cloud[selected];pc[i]=c[chosen];pm[i]=m[chosen]
    def solve(hypothesis):
        f,cw,mw=hypothesis;cost={}
        for i in editable:
            prior=(1-f)*old[i]+f*reference[i];radius=.012 if i in lip_ids else .015
            cost[i]=np.sum((pools[i]-prior)**2,axis=1)/.006**2+.3*np.sum((pools[i]-old[i])**2,axis=1)/radius**2-cw*pc[i]-mw*pm[i]
        result=select_curves(pools,cost,old,upper,lower,oval_loop)
        return {i:int(indices[i][k]) for i,k in result.items()}
    for h in hypotheses:ensemble.append(solve(h))
    choice=ensemble[4];proposal=copy.deepcopy(base);points=old.copy();records=[]
    for i in editable:
        j=choice[i];slot=proposal['slots'][i];slot.update(source_triangle=int(support[j]),source_vertex_ids=ids[t[support[j]]].tolist(),barycentric=barys[j].tolist(),human_reviewed=False)
        slot['proposal_evidence_scope']='automated local geometric hypothesis, not anatomical approval'
        slot['review_flags']=sorted(set(slot.get('review_flags',[]))|{'automated_proposal_not_anatomically_validated'})
        points[i]=cloud[j];spread=float(np.linalg.norm(np.array([cloud[e[i]] for e in ensemble])-points[i],axis=1).max());local=int(np.flatnonzero(indices[i]==j)[0])
        records.append({'veyra_index':i,'group':groups[i],'human_control_pinned':i in pinned,'displacement':float(np.linalg.norm(points[i]-old[i])),'ensemble_max_spread':spread,'curvature_raw':float(cloud_c[j]),'morph_gradient_raw':float(cloud_m[j]),'curvature_regional_percentile':float(pc[i][local]),'morph_gradient_regional_percentile':float(pm[i][local]),'canonical_prior_residual':float(np.linalg.norm(points[i]-reference[i])),'evidence_scope':'local geometric ranks and weight sensitivity; not calibrated anatomical confidence','anatomical_confidence':'unverified','decision':'uncertain','reviewer':'','source_triangle':int(support[j]),'barycentric':barys[j].tolist(),'flags':[]})
    # The actual Veyra skin topology stays separate from the canonical mouth caps.
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]));from veyra.topology_data import TRIANGLE_INDICES
    skin=np.array(TRIANGLE_INDICES);base_quality=deformation_metrics(old,points,skin)
    cases=[];failures={i:set() for i in editable};view_records=[];refit_max={i:0. for i in editable}
    identities={'base':np.zeros_like(v),'round':.5*target('head/head-round.target.gz'),'narrow':.5*target('head/head-scale-horiz-decr.target.gz'),'chin':.5*target('chin/chin-height-incr.target.gz'),'temple':.5*target('forehead/forehead-temple-incr.target.gz')}
    expr_root='expression/units/caucasian/';expressions={'neutral':np.zeros_like(v),'mouth_open':.35*target(expr_root+'mouth-open.target.gz'),'corner_pull':.35*target(expr_root+'mouth-corner-puller.target.gz'),'eye_close':.35*(target(expr_root+'eye-left-closure.target.gz')+target(expr_root+'eye-right-closure.target.gz'))}
    bpy.ops.wm.open_mainfile(filepath=str(project/'annotation.blend'));scene=bpy.context.scene;scene['veyra_annotation_root']=str(project);ui.register();ui.source_unchanged()
    views=json.loads((project/'views.json').read_text());old_support=v[t[[s['source_triangle'] for s in proposal['slots']]]];n0=np.cross(old_support[:,1]-old_support[:,0],old_support[:,2]-old_support[:,0])
    for identity,delta in identities.items():
        for expression,expression_delta in expressions.items():
            varied=v+delta+expression_delta;carried=evaluate_anchors(varied,t,proposal['slots']);supports=varied[t[[s['source_triangle'] for s in proposal['slots']]]];n1=np.cross(supports[:,1]-supports[:,0],supports[:,2]-supports[:,0]);bad=np.flatnonzero((np.linalg.norm(n1,axis=1)<1e-10)|((n0*n1).sum(1)<=0)).tolist()
            metrics=deformation_metrics(points,carried,skin);crossings=segment_crossings(carried,loop)+segment_crossings(carried,oval_loop)
            lip_order=[i for chain in [upper,lower] for a,b in zip(chain,chain[1:]) if carried[b,0]<=carried[a,0] for i in [a,b]]
            for i in set(bad+metrics['flagged_vertices']+lip_order+sum(crossings,[]))&set(editable):failures[i].add(identity+'/'+expression)
            # Re-select among the same anatomical candidate supports under changed geometry.
            # This tests cue drift beyond the tautological fact that barycentrics stay attached.
            _,case_c,case_m,_,_=mesh_evidence(varied,t,morphs)
            candidate_xyz=np.einsum('ni,nij->nj',barys,varied[t[support]])
            c_values=np.einsum('ni,ni->n',barys,case_c[t[support]]);m_values=np.einsum('ni,ni->n',barys,case_m[t[support]])
            case_old=evaluate_anchors(varied,t,base['slots']);sc,ro,tr=fit_similarity(oriented[controls],case_old[controls]);case_ref=sc*oriented@ro+tr
            case_ref[lip_ids]+=case_old[inner_canonical].mean(0)-case_ref[inner_canonical].mean(0)
            case_pools={i:candidate_xyz[indices[i]] for i in editable};case_cost={}
            for i in editable:
                prior=.35*case_old[i]+.65*case_ref[i];radius=.012 if i in lip_ids else .015
                c_rank=percentile(c_values[indices[i]],c_values[region_indices[i]]);m_rank=percentile(m_values[indices[i]],m_values[region_indices[i]])
                case_cost[i]=np.sum((case_pools[i]-prior)**2,axis=1)/.006**2+.3*np.sum((case_pools[i]-case_old[i])**2,axis=1)/radius**2-.4*c_rank-.4*m_rank
            try:
                reselection=select_curves(case_pools,case_cost,case_old,upper,lower,oval_loop)
                drift={i:float(np.linalg.norm(case_pools[i][reselection[i]]-carried[i])) for i in editable}
                for i,d in drift.items():refit_max[i]=max(refit_max[i],d)
                refit_error=None
            except ValueError as error:
                drift={};refit_error=str(error)
                for i in editable:failures[i].add('infeasible_reselection/'+identity+'/'+expression)
            uv_errors=[]
            for view in views:
                scene.camera.matrix_world=Matrix(view['matrix']);bpy.context.view_layer.update();uv=np.array([tuple(world_to_camera_view(scene,scene.camera,Vector(q))) for q in carried]);cam=np.c_[carried,np.ones(468)]@np.linalg.inv(np.array(view['matrix'])).T
                analytic=np.c_[cam[:,0]/view['width']+.5,.5+cam[:,1]/view['width']];error=float(np.max(abs(analytic-uv[:,:2]))*1024);uv_errors.append(error)
                if error>.001:raise ValueError('Independent camera projection mismatch')
                if identity=='base' and expression=='neutral':
                    camera_position=np.array(view['matrix'])[:3,3];direction=-np.array(view['matrix'])[:3,2];occluded=[];full_bvh=BVHTree.FromPolygons(v.tolist(),t.tolist(),all_triangles=True)
                    for i in editable:
                        origin=points[i]-direction;hit,_,_,_=full_bvh.ray_cast(Vector(origin),Vector(direction));occluded.append(i) if hit is not None and np.linalg.norm(np.array(hit)-points[i])>.0005 else None
                    view_records.append({'view':view['name'],'projection_max_error_px':error,'occluded_editable_indices':occluded,'note':'Occlusion in side views is expected; hidden points are not anatomical failures'})
            cases.append({'identity':identity,'expression':expression,'support_failures':bad,'crossings':crossings,'lip_x_order_failures':sorted(set(lip_order)),'fixed_pool_reselection_drift':drift,'reselection_error':refit_error,'projection_max_error_px':max(uv_errors),**metrics})
    base_crossings=segment_crossings(points,loop)+segment_crossings(points,oval_loop)
    for r in records:
        i=r['veyra_index'];r['fixed_pool_reselection_max_drift']=refit_max[i]
        if refit_max[i]>.002:r['flags'].append('morph_cue_reselection_drift_over_2mm_scene_units')
        r['deformation_failures']=sorted(failures[i])
        if r['ensemble_max_spread']>.002:r['flags'].append('weight_sensitive_over_2mm_scene_units')
        if r['displacement']>.011:r['flags'].append('near_local_search_limit')
        if r['canonical_prior_residual']>.012:r['flags'].append('canonical_shape_outside_local_search')
        if groups[i] in ['forehead','temples']:r['flags'].append('no_unique_mesh_boundary_for_canonical_patch_extent')
        if i in base_quality['flagged_vertices'] or any(i in x for x in base_crossings):r['flags'].append('new_dense_topology_or_curve_warning')
        if failures[i]:r['flags'].append('morph_or_expression_instability')
        supported=not r['flags'] and r['curvature_regional_percentile']>=.6 and r['morph_gradient_regional_percentile']>=.6 and r['displacement']>.0005 and r['morph_gradient_raw']>1e-8
        r['classification']='supported_geometric_proposal' if supported else 'ambiguous_proposal'
        if not supported and not r['flags']:r['flags'].append('insufficient_independent_boundary_evidence')
    audit=args.audit or project.parent/'anatomy-audit-001'
    audit_reference=np.load(audit/'audit-geometry.npz',allow_pickle=False)['canonical_aligned']
    out.mkdir(parents=True);(out/'annotations').mkdir()
    for filename in ['base-map.json','surface.npz','views.json','front.png','profile.png','oblique.png']:
        shutil.copy2(project/filename,out/filename)
    config.update(proposal_source_project=str(project),status='automated_proposals_unvalidated');(out/'project.json').write_text(json.dumps(config,indent=2))
    proposal.update(status='candidate_unvalidated',reviewer=None,proposal_only=True,anatomical_validation='NOT PERFORMED',annotation_applied_indices=[],proposal_editable_indices=editable)
    (out/'mapping-proposals.json').write_text(json.dumps(proposal,indent=2));(out/'point-evidence.json').write_text(json.dumps(records,indent=2))
    scene['veyra_annotation_root']=str(out)
    for obj in bpy.data.objects:
        if not obj.get('veyra_editable',False):continue
        i=obj['veyra_index'];r=next(x for x in records if x['veyra_index']==i);slot=proposal['slots'][i];ui.set_position(obj,points[i],slot['source_triangle'],slot['barycentric']);obj['decision']='uncertain';obj['note']=r['classification']+'; '+', '.join(r['flags']);obj['proposal_evidence']=json.dumps(r)
    (out/'annotations/draft-0000.json').write_text(json.dumps(ui.snapshot(),indent=2))
    scene.veyra_region='all';bpy.context.preferences.filepaths.file_preview_type='NONE';bpy.ops.wm.save_as_mainfile(filepath=str(out/'annotation.blend'));ui.unregister()
    before_distance=np.linalg.norm(old[:,None]-old[None,:],axis=2);after_distance=np.linalg.norm(points[:,None]-points[None,:],axis=2)
    report={'human_control_indices':sorted(pinned),'human_control_file_sha256':sha(args.controls) if args.controls else None,'before_duplicate_pairs':int(np.count_nonzero(np.triu(before_distance<1e-8,1))),'after_duplicate_pairs':int(np.count_nonzero(np.triu(after_distance<1e-8,1))),'status':'automated_proposals_unvalidated','original_mapping_sha256':original_hash,'proposed_points':len(records),'supported_geometric_indices':[r['veyra_index'] for r in records if r['classification']=='supported_geometric_proposal'],'ambiguous_indices':[r['veyra_index'] for r in records if r['classification']=='ambiguous_proposal'],'accepted_points':0,'surface_attachment_failures':0,'new_skin_metrics':base_quality,'new_curve_crossings':base_crossings,'fixed_audit_reference_sha256':sha(audit/'audit-geometry.npz'),'before_fixed_audit_reference':deformation_metrics(audit_reference,old,skin),'after_fixed_audit_reference':deformation_metrics(audit_reference,points,skin),'code_sha256':{name:sha(Path(__file__).with_name(name)) for name in ['blender_propose_annotations.py','surface_proposal_math.py']},'before_fixed_eye_reference':deformation_metrics(reference,old,skin),'after_fixed_eye_reference':deformation_metrics(reference,points,skin),'cases':cases,'views':view_records,'targets':target_provenance,'excluded_cavity_quads':len(cavity),'ensemble_hypotheses':hypotheses,'reference_controls':controls,'warning':'56 proposed points, no calibrated anatomical confidence, no acceptance, no dataset labels. Canonical geometry and morph support are priors, not a border annotation.'}
    (out/'proposal-report.json').write_text(json.dumps(report,indent=2));np.savez(out/'proposal-geometry.npz',before=old,after=points,reference=reference,ensemble=np.array([[cloud[e[i]] for i in editable] for e in ensemble]),editable_indices=editable,curvature=curvature,morph_gradient=morph_gradient)
    if sha(project/'base-map.json')!=original_hash:raise AssertionError('Original changed')
    print('PROPOSALS',json.dumps({k:report[k] for k in ['proposed_points','supported_geometric_indices','ambiguous_indices','accepted_points','surface_attachment_failures']}))

if __name__=='__main__':main()

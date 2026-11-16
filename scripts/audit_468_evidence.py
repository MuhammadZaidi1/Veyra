"""Build a conservative 468-point evidence ledger and preserve all previous artifacts."""
import argparse,json,hashlib,shutil,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scripts.mediapipe_evidence_policy import REGION_REQUIREMENTS,eligible_mask,coverage,training_readiness
from scripts.correspondence_math import evaluate_anchors
from veyra.topology_data import TRIANGLE_INDICES


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,default=Path('runs/veyra3d/annotation-project-001'));p.add_argument('--strict-result',type=Path,default=Path('runs/veyra3d/strict-correspondence-002'));p.add_argument('--reference',type=Path,default=Path('third_party/mediapipe-reference'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();repo=Path(__file__).resolve().parents[1]
    if out.exists():raise ValueError('Use a fresh output; no existing result is overwritten')
    protected={str(path.resolve()):sha(path) for folder in ['runs/veyra3d','scripts','docs','veyra3d','veyra','tests_research','third_party','configs'] for path in (repo/folder).rglob('*') if path.is_file() and '__pycache__' not in path.parts}
    config=json.loads((a.project/'project.json').read_text());baseline_bytes=(a.project/'base-map.json').read_bytes();baseline_hash=sha(a.project/'base-map.json')
    if baseline_hash!=config['base_mapping_sha256'] or sha(a.project/'surface.npz')!=config['surface_sha256']:raise ValueError('Frozen inputs changed')
    prior=json.loads((a.strict_result/'report.json').read_text());prior_points=json.loads((a.strict_result/'point-evidence.json').read_text())
    if prior['baseline_mapping_sha256']!=baseline_hash:raise ValueError('Prior structural evidence uses a different baseline')
    provenance=json.loads((a.reference/'provenance.json').read_text())
    for name,entry in provenance['files'].items():
        if sha(a.reference/name)!=entry['sha256']:raise ValueError('Reference hash mismatch')
    base=json.loads(baseline_bytes);g=np.load(a.project/'surface.npz');points=evaluate_anchors(g['world_xyz'],g['triangles'],base['slots']);triangles=np.array(TRIANGLE_INDICES)
    if len(base['slots'])!=468:raise ValueError('468 architecture required')
    entries=json.loads((a.project/'annotations/draft-0000.json').read_text())['annotations'];disputed={r['veyra_index']:r['annotation_group'] for r in entries}
    if len(disputed)!=56:raise ValueError('Expected 56 disputed points')
    canonical_hash=provenance['files']['canonical_face_model.obj']['sha256'];mh_mask=eligible_mask('makehuman_candidate',base['asset_sha256'],canonical_hash);canonical_mask=eligible_mask('canonical_reference',canonical_hash,canonical_hash)
    ledger=[]
    for i,slot in enumerate(base['slots']):
        region=disputed.get(i);previous=next((r for r in prior_points if r['veyra_index']==i),None)
        ledger.append({'veyra_index':i,'disputed':i in disputed,'region':region or 'other_candidate_site','decision':'retain_unchanged_exclude_from_supervision','eligible_for_ground_truth':False,
            'reason':REGION_REQUIREMENTS[region] if region else 'Not among the disputed 56 does not establish independent anatomical verification.',
            'available_evidence':['canonical index semantics','source surface attachment','rig/morph regional priors'] if region else ['candidate source surface attachment'],
            'missing_evidence':['independent exact source-mesh anatomical binding','verified complete-coordinate origin'],
            'prior_candidate_count':len(previous['candidates']) if previous else 0,'original_anchor':slot})
    out.mkdir(parents=True);(out/'mapping-retained.json').write_bytes(baseline_bytes);(out/'point-evidence.json').write_text(json.dumps(ledger,indent=2))
    registry={'checked_date':'2026-11-16','sources':[
        {'id':'mediapipe_canonical','license':'Apache-2.0','reference':str(a.reference),'geometry_sha256':canonical_hash,'scope':'exact index identity on this reference mesh only','transfer_to_makehuman_verified':False},
        {'id':'makehuman_core','license':'CC0 core graphical assets','url':'https://raw.githubusercontent.com/makehumancommunity/mpfb2/master/LICENSE.md','scope':'known source geometry and named rig/morph regions, not exact MediaPipe sites','transfer_to_makehuman_verified':False},
        {'id':'ict_facekit_light','license':'MIT (Light release only)','url':'https://github.com/USC-ICT/ICT-FaceKit','license_url':'https://raw.githubusercontent.com/USC-ICT/ICT-FaceKit/master/LICENSE','scope':'author-defined 68-point sites and extended jaw references in ICT topology; no assets downloaded or copied','transfer_to_makehuman_verified':False,'missing':['verified ICT-to-MakeHuman surface registration','verified semantic crosswalk from sparse 68 sites to dense 468 sites','forehead/temple patch-boundary bindings']},
        {'id':'detector_or_curvature_estimates','license':'source-dependent; not adopted','scope':'possible proposals or explicitly named pseudo-label experiments, not independent anatomical ground truth','transfer_to_makehuman_verified':False}]}
    (out/'reference-evidence.json').write_text(json.dumps(registry,indent=2))
    report={'status':'468_reference_evidence_audit_no_training','architecture_vertices':468,'architecture_triangles':len(triangles),'geometry_contract':'centered-crop-width-xyz-468-v1','native_geometry_adopted':False,'baseline_sha256':baseline_hash,'applied_corrections':0,'unresolved_disputed_points':56,'makehuman_anatomical_supervision':coverage(mh_mask,triangles),'makehuman_readiness':training_readiness(mh_mask),'canonical_template_geometry_supervision':coverage(canonical_mask,triangles),'canonical_scope':'exact template identity/geometry, not measured human anatomy or identity-recognition ground truth','partial_supervision_warning':'Masking only the 56 disputed points does not validate the other 412. Do not recenter partial labels using uncertain points or only the visible subset.','makehuman_valid_mask':mh_mask.tolist(),'prior_structural_evidence_sha256':sha(a.strict_result/'report.json'),'inherited_baseline_failure_cases':[c['case'] for c in prior['baseline_cases'] if not c['passed']],'structural_policy':['independent regional semantic evidence first; otherwise abstain without optimizing coordinates','valid triangle IDs/convex barycentrics','reject normal/area reversals, severe edge ratios, collisions and expression ordering failures','revalidate jointly against original posed baseline, not a rejected proposal'],'partial_makehuman_experiment':'BLOCKED: no independently verified anatomical subset supplied','permitted_current_experiment':'canonical-template render/projection/paired-frame-occlusion diagnostic only; no model training','identity_recognition_status':'not evaluated; one template cannot establish identity discrimination, glasses invariance or real webcam accuracy','training_started':False,'dataset_exported':False,'github_checkpoint':subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=repo).strip()}
    (out/'report.json').write_text(json.dumps(report,indent=2));(out/'preserved-inputs.json').write_text(json.dumps(protected,indent=2))
    # All panels use the SAME 3D anchors. No apparently better 2D coordinates are invented.
    try:font=ImageFont.truetype('DejaVuSans.ttf',18)
    except OSError:font=ImageFont.load_default(size=18)
    for view in json.loads((a.project/'views.json').read_text()):
        cam=np.c_[points,np.ones(468)]@np.linalg.inv(np.array(view['matrix'])).T;n=view['resolution'];xy=np.c_[(cam[:,0]/view['width']+.5)*n-.5,(.5-cam[:,1]/view['width'])*n-.5];sheet=Image.new('RGB',(n*2,n))
        for column,label in enumerate(['BEFORE: candidate anatomy unvalidated','AFTER: unchanged; disputed points excluded']):
            image=Image.open(a.project/(view['name']+'.png')).convert('RGB');d=ImageDraw.Draw(image)
            for region in ['lips','contour']:
                for i,j in base['region_edges'][region]:d.line([tuple(xy[i]),tuple(xy[j])],fill=(60,190,210),width=1)
            for i in disputed:
                x,y=xy[i];d.ellipse((x-2,y-2,x+2,y+2),fill=(255,170,40));d.text((x+3,y-6),str(i),font=font,fill=(255,200,60),stroke_width=1,stroke_fill='black')
            d.rectangle((0,0,n,65),fill='black');d.text((10,6),label,font=font,fill='white');d.text((10,36),'No other candidate point was automatically promoted to ground truth.',font=font,fill='white');sheet.paste(image,(column*n,0))
        sheet.save(out/('before-after-'+view['name']+'.png'))
    changed=[name for name,digest in protected.items() if sha(name)!=digest]
    if changed:raise AssertionError('Protected input changed: '+str(changed))
    print(json.dumps({k:report[k] for k in ['applied_corrections','unresolved_disputed_points','makehuman_anatomical_supervision','canonical_template_geometry_supervision','native_geometry_adopted']},indent=2))

if __name__=='__main__':main()

"""Test CC0 skin-weight interfaces as competing hypotheses, never anatomical borders."""
import argparse,copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scripts.makehuman_semantics import CoreTargets,sha
from scripts.correspondence_math import evaluate_anchors
from scripts.correspondence_gate import structural_checks,support_checks
from scripts.surface_proposal_math import ordered_loop
from scripts.anatomical_constraints import paths_between


def interface_segments(vertices,triangles,field,threshold):
    result=[]
    for index,tri in enumerate(triangles):
        values=field[tri];crossings=[]
        for a,b in [(0,1),(1,2),(2,0)]:
            if (values[a]<=threshold<values[b]) or (values[b]<=threshold<values[a]):
                fraction=(threshold-values[a])/(values[b]-values[a]);w=np.zeros(3);w[a]=1-fraction;w[b]=fraction;crossings.append(w)
        if len(crossings)==2:result.append((index,np.array(crossings)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['result','project','prototype','audit']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.result/'weight-interface-evidence.json'
    if out.exists():raise ValueError('Preserve existing weight-interface results')
    g=np.load(a.project/'surface.npz');v,t,ids=g['world_xyz'],g['triangles'],g['source_vertex_ids'];base=json.loads((a.project/'base-map.json').read_text());records=json.loads((a.project/'annotations/draft-0000.json').read_text())['annotations'];groups={r['veyra_index']:r['annotation_group'] for r in records};lip=[i for i in groups if groups[i] in ['lip_upper','lip_lower']];outer=set(lip+[61,291]);loop=ordered_loop([e for e in base['region_edges']['lips'] if set(e)<=outer],0);upper,lower=paths_between(loop,61,291);oval=ordered_loop(base['region_edges']['contour'],10)
    from veyra.topology_data import TRIANGLE_INDICES
    skin=np.array(TRIANGLE_INDICES);root=Path(json.loads((a.prototype/'report.json').read_text())['mpfb_path']);path=root/'data/rigs/rigify/weights.human_toes.json';document=json.loads(path.read_text())
    if document.get('license')!='CC0':raise ValueError('Expected explicit CC0 weights')
    lookup={int(i):j for j,i in enumerate(ids)};fields={s:np.zeros(len(v)) for s in ['T','B']}
    for name,weights in document['weights'].items():
        for side in fields:
            if name.startswith('lip.'+side):
                for source_id,weight in weights:
                    if source_id in lookup:fields[side][lookup[source_id]]+=weight
    allowed=np.array(json.loads((a.result/'native-topology-evaluation.json').read_text())['source_triangle_ids']);targets=CoreTargets(root,ids);cases=targets.cases(v);baseline=evaluate_anchors(v,t,base['slots']);fixed=np.load(a.audit/'audit-geometry.npz')['canonical_aligned'];trials=[];coordinates={}
    for threshold in [.5,.75,.9]:
        slots=copy.deepcopy(base['slots']);points=[];failure=None
        for i in lip:
            side='T' if groups[i]=='lip_upper' else 'B';field=fields[side];segments=interface_segments(v,t[allowed],field,threshold);best=None
            for local,bw in segments:
                tri=int(allowed[local]);q=bw@v[t[tri]];direction=q[1]-q[0];fraction=float(np.clip((baseline[i]-q[0])@direction/max(direction@direction,1e-20),0,1));w=(1-fraction)*bw[0]+fraction*bw[1];point=w@v[t[tri]]
                if side=='T' and point[2]<=baseline[13,2]:continue
                if side=='B' and point[2]>=baseline[14,2]:continue
                distance=float(np.linalg.norm(point-baseline[i]))
                if distance>.015:continue
                if best is None or distance<best[0]:best=(distance,tri,w,point)
            if best is None:failure='No eligible connected-surface interface near MP '+str(i);break
            distance,tri,w,point=best;w=np.maximum(w,0);w/=w.sum();slots[i].update(source_triangle=tri,barycentric=w.tolist(),source_vertex_ids=ids[t[tri]].tolist(),human_reviewed=False);points.append({'veyra_index':i,'source_triangle':tri,'source_vertex_ids':ids[t[tri]].tolist(),'barycentric':w.tolist(),'displacement':distance,'interpolated_skin_weight':float(w@field[t[tri]]),'semantic_evidence':'Skinning influence interface only, not a vermilion annotation'})
        if failure:trials.append({'threshold':threshold,'generation_failure':failure,'decision':'unchanged_unresolved'});continue
        neutral=evaluate_anchors(v,t,slots);coordinates[threshold]=neutral;checks=[]
        for case in cases:
            old=evaluate_anchors(case['vertices'],t,base['slots']);new=evaluate_anchors(case['vertices'],t,slots);inc=structural_checks(old,new,skin,[upper,lower],[loop,oval]);absolute=structural_checks(neutral,new,skin,[upper,lower],[loop,oval]);support=support_checks(v,case['vertices'],t,slots);checks.append({'case':case['name'],'versus_posed_baseline':inc,'versus_candidate_neutral':absolute,'support_failure_indices':support})
        reference=structural_checks(fixed,neutral,skin,[upper,lower],[loop,oval]);passed=reference['passed'] and all(x['versus_posed_baseline']['passed'] and x['versus_candidate_neutral']['passed'] and not x['support_failure_indices'] for x in checks)
        trials.append({'threshold':threshold,'structurally_admissible':bool(passed),'exact_semantic_evidence':False,'decision':'reject_structural' if not passed else 'retain_baseline_semantics_unresolved','points':points,'case_checks':checks,'fixed_reference_check':reference})
    report={'status':'competing_skinning_interfaces_not_anatomically_approved','source_relative_path':'data/rigs/rigify/weights.human_toes.json','source_sha256':sha(path),'source_license':document['license'],'field_definition':'sum of upper or lower lip bone skinning weights; NOT anatomical probability','hypotheses':[.5,.75,.9],'corners_retained':True,'reliably_supported_corrections':0,'trials':trials}
    out.write_text(json.dumps(report,indent=2));views=json.loads((a.project/'views.json').read_text())
    for view in views:
        def xy(p):
            cam=np.c_[p,np.ones(len(p))]@np.linalg.inv(np.array(view['matrix'])).T
            return np.c_[(cam[:,0]/view['width']+.5)*1024-.5,(.5-cam[:,1]/view['width'])*1024-.5]
        image=Image.open(a.project/(view['name']+'.png')).convert('RGB');d=ImageDraw.Draw(image);x=xy(baseline)
        for i,j in base['region_edges']['lips']:d.line([tuple(x[i]),tuple(x[j])],fill=(80,210,250),width=1)
        for threshold,color in zip([.5,.75,.9],[(255,180,40),(230,90,220),(255,80,80)]):
            if threshold not in coordinates:continue
            p=xy(coordinates[threshold])
            for i,j in base['region_edges']['lips']:
                if i in outer and j in outer:d.line([tuple(p[i]),tuple(p[j])],fill=color,width=2)
        d.rectangle((0,0,1024,55),fill='black');d.text((12,10),'Skinning interfaces: cyan baseline; orange .5; magenta .75; red .9',fill='white');d.text((12,30),'None is a verified vermilion annotation. No point accepted or changed.',fill='white');image.save(a.result/('weight-interfaces-'+view['name']+'.png'))
    print('WEIGHT_INTERFACES',[(r['threshold'],r.get('structurally_admissible'),len(r.get('points',[]))) for r in trials])
if __name__=='__main__':main()

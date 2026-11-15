"""Same-reference before/after metrics; never infer anatomical validation."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.correspondence_math import parse_obj,evaluate_anchors,deformation_metrics


def comparison(baseline,candidate,prototype,reference):
    baseline,candidate,prototype,reference=map(Path,(baseline,candidate,prototype,reference))
    surface=np.load(prototype/'surface-review.npz',allow_pickle=False)
    _,triangles=parse_obj((reference/'canonical_face_model.obj').read_text())
    fixed=np.load(baseline/'alignment-review.npz',allow_pickle=False)['aligned_world']
    result={'status':'unvalidated','fixed_reference':'baseline similarity-aligned canonical geometry, identical for both maps',
        'caveat':'Projection-stage metrics compare each map with its own fitted template. Fixed-reference metrics are supplied separately; a corrected closed lip seam differs from the generic canonical mouth.'}
    baseline_map=json.loads((baseline/'mapping-candidates.json').read_text())
    for name,folder in [('before',baseline),('after',candidate)]:
        mapping=json.loads((folder/'mapping-candidates.json').read_text())
        report=json.loads((folder/'review-report.json').read_text())
        if mapping['asset_sha256']!=baseline_map['asset_sha256'] or mapping['topology_sha256']!=baseline_map['topology_sha256']:
            raise ValueError('Cannot compare different asset/topology mappings')
        p=evaluate_anchors(surface['world_xyz'],surface['triangles'],mapping['slots'])
        distances=np.linalg.norm(p[:,None]-p[None,:],axis=2);np.fill_diagonal(distances,np.inf)
        fixed_metrics=deformation_metrics(fixed,p,triangles)
        result[name]={'anchors':len(p),'invalid_anchors':report['numerical_anchor_failures'],
            'projection_stage':report['projection_quality'],'fixed_reference':fixed_metrics,
            'duplicate_anchor_vertices':int((distances.min(1)<.0002).sum()),
            'duplicate_pairs':np.argwhere(np.triu(distances<.0002,1)).tolist(),
            'review_flags':report['flag_counts'],'flagged_candidates':report['flagged_candidates'],
            'deformation_cases':len(report['cases']),
            'support_triangle_failures':sum(c['support_triangle_failures'] for c in report['cases']),
            'deformation_flips':sum(c['normal_flip_triangles'] for c in report['cases']),
            'deformation_edge_failures':sum(c['collapsed_or_stretched_edges'] for c in report['cases']),
            'feature_ordering_failures':sum(len(c['feature_ordering_failures']) for c in report['cases']) if all('feature_ordering_failures' in c for c in report['cases']) else None,
            'view_checks':sum(len(c.get('view_checks',[])) for c in report['cases']),
            'unresolved_parameter_triangles':report.get('parameter_orientation',{}).get('unresolved_parameter_triangles'),
            'landmark_13':p[13].tolist(),'landmark_14':p[14].tolist(),
            'maximum_projection_error_px':max(r['projection_error_px'] for r in report['renders'])}
    (candidate/'before-after.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--prototype',type=Path,default=Path('runs/veyra3d/blender-prototype-002'))
    parser.add_argument('--reference',type=Path,default=Path('third_party/mediapipe-reference'))
    args=parser.parse_args();print(json.dumps(comparison(args.baseline,args.candidate,args.prototype,args.reference),indent=2))

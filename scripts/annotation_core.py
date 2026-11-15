"""Pure annotation validation/rebuild. A human point review never approves a map."""
import copy
import hashlib
import json
import numpy as np


def digest(data):return hashlib.sha256(data).hexdigest()


def validate_anchor(anchor,vertices,triangles):
    index=anchor.get('source_triangle');weights=np.asarray(anchor.get('barycentric'),dtype=float)
    if type(index) is not int or not 0<=index<len(triangles):raise ValueError('Invalid source triangle')
    if weights.shape!=(3,) or not np.isfinite(weights).all() or weights.min() < -1e-8 or abs(weights.sum()-1)>1e-7:
        raise ValueError('Invalid barycentric weights')
    return weights@vertices[triangles[index]]


def rebuild(base,annotations,vertices,triangles,source_ids,base_hash,allow_draft_preview=False):
    if annotations.get('base_mapping_sha256')!=base_hash:raise ValueError('Annotation/base mapping hash mismatch')
    if annotations.get('asset_sha256')!=base['asset_sha256'] or annotations.get('topology_sha256')!=base['topology_sha256']:
        raise ValueError('Annotation asset/topology mismatch')
    if digest(triangles.astype(np.int32).tobytes())!=base['topology_sha256']:raise ValueError('Frozen topology changed')
    if len(base['slots'])!=468 or [s['veyra_index'] for s in base['slots']]!=list(range(468)):
        raise ValueError('Base map must preserve 0..467 ordering')
    result=copy.deepcopy(base);seen=set();applied=[]
    for entry in annotations['annotations']:
        index=entry['veyra_index']
        if type(index) is not int or not 0<=index<468 or index in seen:raise ValueError('Invalid/repeated annotation index')
        seen.add(index);validate_anchor(entry,vertices,triangles)
        decision=entry.get('decision','unreviewed')
        if decision not in ('unreviewed','accepted','uncertain','rejected'):raise ValueError('Unknown point-review decision')
        if decision!='accepted':continue
        reviewer=entry.get('reviewer','').strip()
        if not reviewer:raise ValueError('Accepted point requires a named human reviewer')
        slot=result['slots'][index]
        slot.update(source_triangle=entry['source_triangle'],barycentric=list(entry['barycentric']),
            source_vertex_ids=source_ids[triangles[entry['source_triangle']]].tolist(),human_reviewed=False,
            manual_annotation_review={'decision':'accepted','reviewer':reviewer,'note':entry.get('note','')},
            review_flags=['manually_annotated_candidate_requires_anatomical_validation'])
        applied.append(index)
    if not applied and not allow_draft_preview:raise ValueError('No accepted human annotations; use --draft-preview only for a labeled no-correction workflow test')
    result.update(status='candidate_unvalidated',reviewer=None,
        annotation_status='human_point_reviews_applied_map_unvalidated' if applied else 'draft_preview_no_corrections',
        annotation_applied_indices=sorted(applied),
        warning='Manual point decisions do not approve this map. No training labels exported. Unreviewed anchors unchanged; no automatic ring replacement or neighborhood propagation.')
    return result

"""Evidence policy for the existing 468-point contract; no inferred anatomical approval."""
import numpy as np

REGION_REQUIREMENTS={
 'lip_upper': 'A source-specific external upper vermilion border and ordered binding to the canonical arc; a ring, rig influence or curvature ridge is insufficient.',
 'lip_lower': 'A source-specific external lower vermilion border distinguished from the inner rim and labiomental crease, with depth checked.',
 'lip_corners': 'Source surface commissure sites tied to MediaPipe 61/291, not an off-surface rig-control mean.',
 'forehead': 'A source-specific binding of the canonical facial-patch boundary; neither hairline nor skull height establishes it.',
 'temples': 'Side-specific canonical temporal/cheek patch bindings on the 3D surface, not view-dependent silhouettes.',
 'jaw': 'Ordered source mandibular sites and a stable dense-neighborhood binding under expression; bounds/rig controls are only priors.',
 'chin': 'A source chin-site definition matching canonical 152 and its neighbors in both depth and extent.'}


def coverage(mask,triangles):
    mask=np.asarray(mask)
    if mask.dtype!=np.bool_ or mask.shape!=(468,):raise ValueError('Expected bool[468] supervision eligibility')
    triangles=np.asarray(triangles,dtype=int)
    edges=np.unique(np.sort(np.concatenate([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]]),axis=1),axis=0)
    return {'vertices':int(mask.sum()),'edges':int(mask[edges].all(1).sum()),'triangles':int(mask[triangles].all(1).sum())}


def eligible_mask(source_kind,source_hash,canonical_hash):
    """Only the pinned canonical mesh has exact reference-vertex identity here.

    This is not a general anatomical-certification API. No verified MakeHuman
    binding has been supplied. Review flags, curvature scores and 412 other
    candidate anchors are not upgraded into ground truth.
    """
    if source_kind=='canonical_reference':
        if not source_hash or source_hash!=canonical_hash:raise ValueError('Canonical reference hash mismatch')
        return np.ones(468,dtype=bool)
    if source_kind=='makehuman_candidate':return np.zeros(468,dtype=bool)
    raise ValueError('Unknown source scope; no implicit transfer of reference semantics')


def training_readiness(mask):
    if np.asarray(mask).shape!=(468,) or np.asarray(mask).dtype!=np.bool_:raise ValueError('Invalid eligibility mask')
    return {'has_supervision':bool(mask.any()),'may_create_manifest':bool(mask.any()),
        'warning':'Nonempty eligibility alone is not approval: registration, coordinates, licensing and split design must also be verified.'}


def centered_camera_geometry(world_points,camera_matrix,frame_width):
    """Known complete geometry -> existing +x right,+y up,+z toward-camera contract.

    The centroid is from all 468 KNOWN source vertices, never a trusted subset
    or a mixture of verified and speculative source correspondences.
    """
    p=np.asarray(world_points,float)
    if p.shape!=(468,3) or not np.isfinite(p).all() or frame_width<=0:raise ValueError('Expected finite complete 468-point geometry and positive frame width')
    camera=np.c_[p,np.ones(468)]@np.linalg.inv(np.asarray(camera_matrix,float)).T
    xyz=camera[:,:3];return (xyz-xyz.mean(0))/frame_width

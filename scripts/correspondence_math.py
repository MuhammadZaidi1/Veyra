"""Numerical helpers for UNVALIDATED surface-correspondence review."""
import numpy as np


def parse_obj(text):
    vertices=[]
    triangles=[]
    for line in text.splitlines():
        fields=line.split()
        if fields and fields[0]=='v':
            vertices.append([float(x) for x in fields[1:4]])
        elif fields and fields[0]=='f':
            ids=[int(x.split('/')[0])-1 for x in fields[1:]]
            for i in range(1,len(ids)-1):
                triangles.append([ids[0],ids[i],ids[i+1]])
    vertices=np.asarray(vertices,dtype=float)
    triangles=np.asarray(triangles,dtype=np.int32)
    if vertices.shape!=(468,3) or not np.isfinite(vertices).all() or triangles.min()<0 or triangles.max()>=468:
        raise ValueError('Expected exactly 468 OBJ vertices in original file ordering')
    return vertices,triangles


def fit_similarity(source,target):
    source,target=np.asarray(source),np.asarray(target)
    a,b=source.mean(0),target.mean(0)
    x,y=source-a,target-b
    if np.linalg.matrix_rank(x)<2:
        raise ValueError('Alignment controls are degenerate')
    u,s,vt=np.linalg.svd(x.T@y)
    sign=np.ones(3);sign[-1]=np.sign(np.linalg.det(u@vt))
    rotation=(u*sign)@vt
    scale=float((s*sign).sum()/(x*x).sum())
    if scale<=0:
        raise ValueError('Invalid similarity scale')
    translation=b-scale*a@rotation
    return scale,rotation,translation


def radial_warp(points,controls,targets):
    """3D polyharmonic r kernel + affine tail; exact reviewed/coarse controls."""
    origin=controls.mean(0)
    unit=float(np.linalg.norm(np.ptp(controls,axis=0)))
    c=(controls-origin)/unit
    p=(points-origin)/unit
    k=np.linalg.norm(c[:,None]-c[None,:],axis=2)
    affine=np.column_stack((np.ones(len(c)),c))
    matrix=np.block([[k,affine],[affine.T,np.zeros((4,4))]])
    coefficients=np.linalg.lstsq(matrix,np.vstack((targets-controls,np.zeros((4,3)))),rcond=1e-10)[0]
    basis=np.column_stack((np.linalg.norm(p[:,None]-c[None,:],axis=2),np.ones(len(p)),p))
    return points+basis@coefficients


def barycentric(point,triangle):
    a,b,c=np.asarray(triangle)
    ab,ac=b-a,c-a
    gram=np.array([[ab@ab,ab@ac],[ab@ac,ac@ac]])
    if np.linalg.det(gram)<1e-20:
        raise ValueError('Degenerate anchor triangle')
    v,w=np.linalg.solve(gram,[(point-a)@ab,(point-a)@ac])
    weights=np.array([1-v-w,v,w])
    if weights.min() < -1e-4:
        raise ValueError('Projected point lies outside triangle')
    weights=np.maximum(weights,0)
    return weights/weights.sum()


def evaluate_anchors(vertices,triangles,slots):
    if len(slots)!=468 or [s['veyra_index'] for s in slots]!=list(range(468)):
        raise ValueError('Landmark slots must preserve exact 0..467 ordering')
    ids=np.array([s['source_triangle'] for s in slots])
    weights=np.array([s['barycentric'] for s in slots],dtype=float)
    if ids.dtype.kind not in 'iu' or ids.min()<0 or ids.max()>=len(triangles):
        raise ValueError('Invalid triangle IDs')
    if weights.shape!=(468,3) or not np.isfinite(weights).all() or weights.min() < -1e-8 or not np.allclose(weights.sum(1),1,atol=1e-7):
        raise ValueError('Invalid barycentric weights')
    return np.einsum('ni,nij->nj',weights,vertices[triangles[ids]])


def deformation_metrics(base,points,canonical_triangles):
    edges=np.unique(np.sort(np.concatenate([canonical_triangles[:,[0,1]],canonical_triangles[:,[1,2]],canonical_triangles[:,[2,0]]]),axis=1),axis=0)
    old=np.linalg.norm(base[edges[:,0]]-base[edges[:,1]],axis=1)
    new=np.linalg.norm(points[edges[:,0]]-points[edges[:,1]],axis=1)
    ratio=new/np.maximum(old,1e-12)
    a=base[canonical_triangles];b=points[canonical_triangles]
    n0=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]);n1=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0])
    flip=(n0*n1).sum(1)<0
    stretch=(ratio<.25)|(ratio>4)
    return {'nonfinite_values':int((~np.isfinite(points)).sum()),'collapsed_or_stretched_edges':int(stretch.sum()),
        'normal_flip_triangles':int(flip.sum()),'flagged_vertices':sorted(set(edges[stretch].ravel().tolist()+canonical_triangles[flip].ravel().tolist())),
        'maximum_displacement':float(np.linalg.norm(points-base,axis=1).max())}

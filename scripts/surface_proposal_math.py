"""Evidence-ranked surface proposals. Scores measure repeatability, NOT anatomical probability."""
import numpy as np


def mesh_evidence(vertices, triangles, morphs):
    """Normal-variation curvature and spatial morph-response gradients on source edges."""
    p=vertices[triangles];cross=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);area=np.linalg.norm(cross,axis=1)
    normals=np.zeros_like(vertices)
    for j in range(3):np.add.at(normals,triangles[:,j],cross)
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-15)
    edges=np.unique(np.sort(np.concatenate([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]]),axis=1),axis=0)
    a,b=edges.T;length=np.maximum(np.linalg.norm(vertices[a]-vertices[b],axis=1),1e-12)
    curvature=np.linalg.norm(normals[a]-normals[b],axis=1)/length
    response=np.stack([np.linalg.norm(m,axis=1) for m in morphs],axis=1)
    gradient=np.linalg.norm(response[a]-response[b],axis=1)/length
    counts=np.zeros(len(vertices));c=np.zeros(len(vertices));g=np.zeros(len(vertices))
    for side in [a,b]:np.add.at(counts,side,1);np.add.at(c,side,curvature);np.add.at(g,side,gradient)
    return normals,c/np.maximum(counts,1),g/np.maximum(counts,1),response.max(axis=1),area


def percentile(values, reference):
    """Empirical rank within a declared regional population, not a calibrated confidence."""
    ordered=np.sort(reference)
    return (np.searchsorted(ordered,values,side='right')+np.searchsorted(ordered,values,side='left'))/(2*max(len(reference),1))


def ordered_loop(edges, start):
    graph={}
    for a,b in edges:graph.setdefault(a,set()).add(b);graph.setdefault(b,set()).add(a)
    if any(len(v)!=2 for v in graph.values()):raise ValueError('Expected a simple closed loop')
    path=[start];previous=None;current=start
    while True:
        nxt=min(graph[current]-({previous} if previous is not None else set()))
        if nxt==start:break
        if nxt in path:raise ValueError('Loop repeats before closure')
        path.append(nxt);previous,current=current,nxt
    if len(path)!=len(graph):raise ValueError('Disconnected loop')
    return path


def solve_chain(pools, costs, baseline, closed=False, x_order=False):
    """Joint ordered-curve selection; no assumption that an existing mesh ring is anatomy.

    Penalize changes in neighboring 3D displacements. Disallow reversed tangents
    and extreme edge ratios. X monotonicity is used only for the two mouth arcs.
    """
    best=None
    for first in (range(len(pools[0])) if closed else [None]):
        scores=costs[0].copy()
        if first is not None:scores[np.arange(len(scores))!=first]=np.inf
        parents=[]
        for k in range(1,len(pools)):
            old=baseline[k]-baseline[k-1];delta=pools[k][None]-pools[k-1][:,None]
            ratio=np.linalg.norm(delta,axis=2)/max(np.linalg.norm(old),1e-12)
            allowed=(delta@old>0)&(ratio>.25)&(ratio<4)
            if x_order:allowed &= delta[:,:,0]>1e-7
            transition=.2*np.sum((delta-old)**2,axis=2)/(.004**2)
            total=scores[:,None]+np.where(allowed,transition,np.inf)
            parent=np.argmin(total,axis=0);scores=costs[k]+total[parent,np.arange(len(parent))];parents.append(parent)
        if closed:
            old=baseline[0]-baseline[-1];delta=pools[0][first]-pools[-1];ratio=np.linalg.norm(delta,axis=1)/np.linalg.norm(old)
            scores+=np.where((delta@old>0)&(ratio>.25)&(ratio<4),.2*np.sum((delta-old)**2,axis=1)/(.004**2),np.inf)
        last=int(np.argmin(scores));value=float(scores[last])
        if not np.isfinite(value):continue
        path=[last]
        for parent in reversed(parents):path.append(int(parent[path[-1]]))
        path.reverse()
        if best is None or value<best[0]:best=(value,path)
    if best is None:raise ValueError('No feasible ordered candidate curve; do not silently relax constraints')
    return best[1]


def segment_crossings(points,loop):
    """Strict nonadjacent crossings in the front x/z plane (depth still checked separately)."""
    points=points[:,[0,2]];bad=[]
    def orient(a,b,c):u=b-a;v=c-a;return u[0]*v[1]-u[1]*v[0]
    for k in range(len(loop)):
        a,b=loop[k],loop[(k+1)%len(loop)]
        for j in range(k+1,len(loop)):
            c,d=loop[j],loop[(j+1)%len(loop)]
            if len({a,b,c,d})<4:continue
            if orient(points[a],points[b],points[c])*orient(points[a],points[b],points[d])<-1e-18 and orient(points[c],points[d],points[a])*orient(points[c],points[d],points[b])<-1e-18:bad.append([a,b,c,d])
    return bad

def select_curves(pools,costs,baseline,upper,lower,oval):
    """Shared-corner mouth arcs plus a closed oval; deterministic DP selection."""
    result={i:int(np.argmin(costs[i])) for i in [61,291]}
    for chain in [upper,lower]:
        cp=[costs[i].copy() for i in chain]
        for j in [0,len(chain)-1]:cp[j][np.arange(len(cp[j]))!=result[chain[j]]]=np.inf
        result.update(zip(chain,solve_chain([pools[i] for i in chain],cp,baseline[chain],x_order=True)))
    result.update(zip(oval,solve_chain([pools[i] for i in oval],[costs[i] for i in oval],baseline[oval],closed=True)))
    return result

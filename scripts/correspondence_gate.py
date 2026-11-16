"""Fail-closed structural tests for surface correspondence; never anatomical approval."""
import heapq
import numpy as np
from scripts.surface_proposal_math import segment_crossings


def structural_checks(reference, points, faces, mouth_arcs=(), loops=(), collision_distance=.0002):
    reference=np.asarray(reference,float);points=np.asarray(points,float);faces=np.asarray(faces,int)
    if points.shape!=reference.shape or not np.isfinite(points).all():
        return {'passed':False,'nonfinite_or_shape_failure':True,'reversals':[],'degenerate_triangles':[],'distorted_edges':[],'collisions':[],'mouth_order_failures':[],'curve_crossings':[]}
    a=reference[faces];b=points[faces];n0=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]);n1=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0]);l0=np.linalg.norm(n0,axis=1);l1=np.linalg.norm(n1,axis=1)
    reversed_ids=np.flatnonzero((n0*n1).sum(1)<=0).tolist()
    degenerate=np.flatnonzero((l1<=1e-12)|(l1/np.maximum(l0,1e-15)<.05)).tolist()
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    length0=np.linalg.norm(reference[edges[:,0]]-reference[edges[:,1]],axis=1);length1=np.linalg.norm(points[edges[:,0]]-points[edges[:,1]],axis=1)
    ratio=length1/np.maximum(length0,1e-15);bad_edges=edges[(ratio<=.25)|(ratio>=4)].tolist()
    distance=np.linalg.norm(points[:,None]-points[None,:],axis=2);collisions=np.argwhere(np.triu(distance<collision_distance,1)).tolist()
    order=[[int(a),int(b)] for arc in mouth_arcs for a,b in zip(arc,arc[1:]) if points[b,0]-points[a,0]<=1e-7]
    crossings=[x for loop in loops for x in segment_crossings(points,loop)]
    result={'nonfinite_or_shape_failure':False,'reversals':reversed_ids,'degenerate_triangles':degenerate,'distorted_edges':bad_edges,'collisions':collisions,'mouth_order_failures':order,'curve_crossings':crossings}
    result['passed']=not any(result.values())
    return result


def support_checks(reference_vertices,vertices,triangles,slots):
    ids=np.array([s['source_triangle'] for s in slots]);w=np.array([s['barycentric'] for s in slots],float)
    if ids.dtype.kind not in 'iu' or ids.min()<0 or ids.max()>=len(triangles) or w.shape!=(len(slots),3) or not np.isfinite(w).all() or w.min()<0 or not np.allclose(w.sum(1),1,rtol=0,atol=1e-7):raise ValueError('Invalid surface anchors')
    a=reference_vertices[triangles[ids]];b=vertices[triangles[ids]];n0=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]);n1=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0])
    return np.flatnonzero((np.linalg.norm(n1,axis=1)<=1e-12)|((n0*n1).sum(1)<=0)).tolist()


def quad_graph(vertices,quads,source_ids,allowed):
    lookup={int(x):i for i,x in enumerate(source_ids)};graph={int(i):{} for i in np.flatnonzero(allowed)}
    for quad in quads:
        for a,b in zip(quad,quad[1:]+quad[:1]):
            i,j=lookup[a],lookup[b]
            if i in graph and j in graph:
                length=float(np.linalg.norm(vertices[i]-vertices[j]));graph[i][j]=length;graph[j][i]=length
    return graph


def shortest_mesh_path(graph,start,end):
    """Dijkstra on real source quad edges. Never substitutes straight spatial chords."""
    if start not in graph or end not in graph:raise ValueError('Endpoint outside region')
    best={start:0.};parent={};queue=[(0.,start)]
    while queue:
        distance,i=heapq.heappop(queue)
        if distance!=best[i]:continue
        if i==end:
            path=[i]
            while path[-1]!=start:path.append(parent[path[-1]])
            return list(reversed(path))
        for j,length in sorted(graph[i].items()):
            candidate=distance+length
            if candidate<best.get(j,float('inf')):best[j]=candidate;parent[j]=i;heapq.heappush(queue,(candidate,j))
    raise ValueError('No connected source-surface path')


def semantic_decision(structurally_passed,exact_semantic_evidence):
    if not structurally_passed:return 'reject_structural'
    if not exact_semantic_evidence:return 'retain_baseline_semantics_unresolved'
    return 'supported_candidate_not_anatomically_approved'

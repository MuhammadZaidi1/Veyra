"""Topology-constrained feature curves for the pinned hm08 mesh. UNVALIDATED anatomy."""
import numpy as np
try:
    from scripts.correspondence_math import barycentric
except ModuleNotFoundError:
    from correspondence_math import barycentric


def body_quads(text):
    quads=[];group=None
    for line in text.splitlines():
        f=line.split()
        if f and f[0]=='g':group=f[1]
        if f and f[0]=='f' and group=='body':
            q=[int(x.split('/')[0])-1 for x in f[1:]]
            if len(q)!=4:raise ValueError('Expected frozen MakeHuman quad topology')
            quads.append(q)
    return quads


def trace_quad_loop(quads,start,second):
    adjacency={};edge_faces={}
    for q in quads:
        for a,b in zip(q,q[1:]+q[:1]):
            adjacency.setdefault(a,set()).add(b);adjacency.setdefault(b,set()).add(a)
            edge_faces.setdefault(tuple(sorted((a,b))),[]).append(q)
    path=[start];previous=start;current=second
    while current!=start:
        if current in path or len(path)>500:raise ValueError('Nonclosing feature loop')
        path.append(current);forbidden={previous}
        for q in edge_faces.get(tuple(sorted((previous,current))),[]):forbidden.update(q)
        options=adjacency[current]-forbidden
        if len(options)!=1:raise ValueError('Feature loop branches at extraordinary vertex')
        previous,current=current,options.pop()
    return path


def paths_between(loop,start,end):
    a=loop.index(start);ordered=loop[a:]+loop[:a];b=ordered.index(end)
    return ordered[:b+1],[start]+list(reversed(ordered[b:]))


def segment_sample(points,fraction):
    lengths=np.linalg.norm(np.diff(points,axis=0),axis=1);total=lengths.sum()
    if total<=0:raise ValueError('Degenerate feature curve')
    position=np.clip(fraction,0,1)*total
    index=min(int(np.searchsorted(np.cumsum(lengths),position,side='right')),len(lengths)-1)
    before=lengths[:index].sum();weight=float((position-before)/lengths[index])
    return index,np.array([1-weight,weight])


def harmonic_warp(points,triangles,indices,targets,regularization=1e-4):
    """Biharmonic displacement interpolation with hard feature-curve constraints."""
    n=len(points);adj=np.zeros((n,n))
    for a,b,c in triangles:
        for i,j in ((a,b),(b,c),(c,a)):adj[i,j]=adj[j,i]=1
    lap=np.eye(n)-adj/np.maximum(adj.sum(1,keepdims=True),1)
    q=lap.T@lap+regularization*np.eye(n)
    fixed=np.array(indices,dtype=int);free=np.setdiff1d(np.arange(n),fixed)
    if len(set(fixed))!=len(fixed):raise ValueError('Repeated hard constraint')
    displacement=np.zeros_like(points)
    displacement[fixed]=np.asarray(targets)-points[fixed]
    displacement[free]=np.linalg.solve(q[np.ix_(free,free)],-q[np.ix_(free,fixed)]@displacement[fixed])
    return points+displacement


def feature_constraints(canonical,region_edges,vertices,triangles,source_ids,quads):
    lookup={int(value):i for i,value in enumerate(source_ids)}
    def world(ids):return vertices[[lookup[i] for i in ids]]
    # Seeds select CLOSED QUAD LOOPS, not hand-positioned projected labels.
    inner=trace_quad_loop(quads,474,473)
    outer=trace_quad_loop(quads,362,357)
    eye_negative=trace_quad_loop(quads,0,1)
    mirror=[]
    for source_id in (0,1):
        p=world([source_id])[0]*[-1,1,1]
        i=int(np.linalg.norm(vertices-p,axis=1).argmin())
        if np.linalg.norm(vertices[i]-p)>1e-7:raise ValueError('Expected hm08 symmetric eye loop')
        mirror.append(int(source_ids[i]))
    eye_positive=trace_quad_loop(quads,*mirror)
    loops={'mouth_inner':inner,'mouth_outer':outer,'eye_negative_x':eye_negative,'eye_positive_x':eye_positive}
    # Official connected components: two mouth rings and one ring per eye.
    canonical_loops=[]
    for name in ('lips','eyes','other_eye'):
        graph={}
        for a,b in region_edges[name]:graph.setdefault(a,set()).add(b);graph.setdefault(b,set()).add(a)
        remaining=set(graph)
        while remaining:
            start=min(remaining);path=[start];previous=None;current=start
            while True:
                choices=sorted(graph[current]-({previous} if previous is not None else set()))
                nxt=choices[0]
                if nxt==start:break
                path.append(nxt);previous,current=current,nxt
                if len(path)>len(graph):raise ValueError('Invalid canonical feature cycle')
            remaining-=set(path);canonical_loops.append(path)
    edge_triangles={}
    for i,t in enumerate(triangles):
        for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
            edge_triangles.setdefault(tuple(sorted((int(source_ids[a]),int(source_ids[b])))),[]).append(i)
    constraints={};records=[]
    for cloop in canonical_loops:
        if 13 in cloop:name='mouth_inner';top,bottom=13,14;st,sb=474,493
        elif 0 in cloop:name='mouth_outer';top,bottom=0,17;st,sb=362,5337
        else:
            name='eye_negative_x' if canonical[cloop,0].mean()<0 else 'eye_positive_x'
            top=max(cloop,key=lambda i:canonical[i,1]);bottom=min(cloop,key=lambda i:canonical[i,1])
            sloop=loops[name];st=max(sloop,key=lambda i:world([i])[0,2]);sb=min(sloop,key=lambda i:world([i])[0,2])
        sloop=loops[name]
        cleft=min(cloop,key=lambda i:canonical[i,0]);cright=max(cloop,key=lambda i:canonical[i,0])
        sl=min(sloop,key=lambda i:world([i])[0,0]);sr=max(sloop,key=lambda i:world([i])[0,0])
        cup,clower=paths_between(cloop,cleft,cright)
        if top not in cup:cup,clower=clower,cup
        sup,slower=paths_between(sloop,sl,sr)
        if st not in sup:sup,slower=slower,sup
        for cp,sp,cm,sm in ((cup,sup,top,st),(clower,slower,bottom,sb)):
            ci=cp.index(cm);si=sp.index(sm)
            for cpart,spart in ((cp[:ci+1],sp[:si+1]),(cp[ci:],sp[si:])):
                cpoints=canonical[cpart];dist=np.r_[0,np.cumsum(np.linalg.norm(np.diff(cpoints,axis=0),axis=1))]
                for j,landmark in enumerate(cpart):
                    seg,w=segment_sample(world(spart),dist[j]/dist[-1])
                    pair=spart[seg:seg+2];position=w@world(pair)
                    possible=edge_triangles[tuple(sorted(pair))]
                    # Prefer the outward/skin-side triangle of a fold edge; weights remain on the exact edge.
                    def normal_y(i):
                        a,b,c=vertices[triangles[i]];n=np.cross(b-a,c-a);return n[1]/max(np.linalg.norm(n),1e-20)
                    support=min(possible,key=normal_y)
                    weights=barycentric(position,vertices[triangles[support]])
                    constraints[landmark]={'source_triangle':support,'barycentric':weights.tolist(),
                        'point':position,'feature_curve':name,'source_edge':pair}
        records.append({'name':name,'source_loop':sloop,'canonical_loop':cloop,
            'canonical_upper_center':top,'canonical_lower_center':bottom,'source_upper_center':st,'source_lower_center':sb,
            'source_negative_x_corner':sl,'source_positive_x_corner':sr,'anatomically_validated':False})
    if len(constraints)!=72:raise ValueError(f'Expected 72 ordered feature-boundary constraints, found {len(constraints)}')
    return constraints,records


def preserve_parameter_orientation(reference,points,triangles,fixed,iterations=80):
    """Positive-area constraint on the anatomical surface parameterization.

    Only free registration coordinates move; feature-curve constraints stay fixed.
    Final points must still be attached to the actual 3D source mesh by the caller.
    """
    result=points.copy();p=result[:,[0,2]].copy();r=reference[:,[0,2]]
    rt=r[triangles]
    reference_area=(rt[:,1,0]-rt[:,0,0])*(rt[:,2,1]-rt[:,0,1])-(rt[:,1,1]-rt[:,0,1])*(rt[:,2,0]-rt[:,0,0])
    sign=np.sign(reference_area);minimum=np.maximum(abs(reference_area)*.005,1e-12)
    locked=set(fixed)
    for sweep in range(iterations):
        changed=0
        for k,ids in enumerate(triangles):
            a,b,c=p[ids]
            area=sign[k]*((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))
            if area>=minimum[k]:continue
            gradient=sign[k]*np.array([[b[1]-c[1],c[0]-b[0]],[c[1]-a[1],a[0]-c[0]],[a[1]-b[1],b[0]-a[0]]])
            for j,i in enumerate(ids):
                if int(i) in locked:gradient[j]=0
            denominator=(gradient*gradient).sum()
            if denominator<1e-20:continue
            p[ids]+=(minimum[k]-area)/denominator*gradient
            changed+=1
        if not changed:break
    t=p[triangles]
    area=sign*((t[:,1,0]-t[:,0,0])*(t[:,2,1]-t[:,0,1])-(t[:,1,1]-t[:,0,1])*(t[:,2,0]-t[:,0,0]))
    result[:,0]=p[:,0];result[:,2]=p[:,1]
    return result,{'iterations':sweep+1,'unresolved_parameter_triangles':np.flatnonzero(area<minimum*.99).tolist(),
        'maximum_parameter_adjustment':float(np.linalg.norm(result-points,axis=1).max())}


def lip_ribbon_constraints(canonical,canonical_triangles,vertices,triangles,source_ids,quads,records):
    """Intrinsic annular lip-band coordinates, bounded by the two anatomical loops.

    Returns 3D target points for the intermediate vermilion rows; caller attaches
    them to the ACTUAL source lip ribbon, never to a flat projected overlay.
    """
    mouth={r['name']:r for r in records if r['name'].startswith('mouth')}
    outer=mouth['mouth_outer']['canonical_loop'];inner=mouth['mouth_inner']['canonical_loop']
    def edges(loop):return {tuple(sorted((a,b))) for a,b in zip(loop,loop[1:]+loop[:1])}
    barriers=edges(outer)|edges(inner)
    edge_faces={}
    for k,t in enumerate(canonical_triangles):
        for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):edge_faces.setdefault(tuple(sorted((int(a),int(b)))),[]).append(k)
    start=next(k for k,t in enumerate(canonical_triangles) if 11 in t)
    visited=set();stack=[start]
    while stack:
        k=stack.pop()
        if k in visited:continue
        visited.add(k);t=canonical_triangles[k]
        for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
            edge=tuple(sorted((int(a),int(b))))
            if edge not in barriers:stack.extend(j for j in edge_faces[edge] if j not in visited)
    band=canonical_triangles[sorted(visited)];domain=sorted(set(band.ravel().tolist()))
    if len(domain)!=80:raise ValueError(f'Expected canonical 80-node lip ribbon, found {len(domain)}')
    lookup={int(value):i for i,value in enumerate(source_ids)}
    def world(ids):return vertices[[lookup[i] for i in ids]]
    boundary={}
    for name,loop,top,bottom in [('outer',outer,0,17),('inner',inner,13,14)]:
        left=min(loop,key=lambda i:canonical[i,0]);right=max(loop,key=lambda i:canonical[i,0])
        upper,lower=paths_between(loop,left,right)
        if top not in upper:upper,lower=lower,upper
        # Four oriented arcs: top -> right -> bottom -> left -> top.
        arcs=[upper[upper.index(top):],list(reversed(lower[lower.index(bottom):])),list(reversed(lower[:lower.index(bottom)+1])),upper[:upper.index(top)+1]]
        # Last arc currently left->top, already increasing phase.
        for quadrant,arc in enumerate(arcs):
            p=canonical[arc];d=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];fraction=d/d[-1]
            for i,f in zip(arc,fraction):
                angle=(quadrant+f)*np.pi/2
                boundary[i]=[0 if name=='outer' else 1,np.cos(angle),np.sin(angle)]
    index={i:j for j,i in enumerate(domain)};n=len(domain);adj=np.zeros((n,n))
    for t in band:
        for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):adj[index[int(a)],index[int(b)]]=adj[index[int(b)],index[int(a)]]=1
    lap=np.diag(adj.sum(1))-adj;fixed=sorted(boundary);free=sorted(set(domain)-set(fixed));f=np.array([index[i] for i in fixed]);u=np.array([index[i] for i in free])
    parameters=np.zeros((n,3));parameters[f]=[boundary[i] for i in fixed]
    parameters[u]=np.linalg.solve(lap[np.ix_(u,u)],-lap[np.ix_(u,f)]@parameters[f])
    source_loops=[trace_quad_loop(quads,*seed) for seed in [(362,357),(455,454),(466,463),(467,464),(509,508),(468,465),(474,473)]]
    source_arcs=[]
    for loop in source_loops:
        top=loop[0];bottom=min((i for i in loop if abs(world([i])[0,0])<1e-7),key=lambda i:world([i])[0,2])
        left=min(loop,key=lambda i:world([i])[0,0]);right=max(loop,key=lambda i:world([i])[0,0])
        upper,lower=paths_between(loop,left,right)
        if top not in upper:upper,lower=lower,upper
        source_arcs.append([upper[upper.index(top):],list(reversed(lower[lower.index(bottom):])),list(reversed(lower[:lower.index(bottom)+1])),upper[:upper.index(top)+1]])
    def sample_ring(k,phase):
        quadrant=min(int(phase/(np.pi/2)),3);fraction=phase/(np.pi/2)-quadrant
        arc=source_arcs[k][quadrant];segment,weights=segment_sample(world(arc),fraction)
        return weights@world(arc[segment:segment+2])
    results={}
    for i in free:
        radial,c,s=parameters[index[i]];phase=float(np.arctan2(s,c)%(2*np.pi));position=np.clip(radial,0,1)*6;k=min(int(position),5);fraction=position-k
        point=(1-fraction)*sample_ring(k,phase)+fraction*sample_ring(k+1,phase)
        results[i]={'point':point,'feature_curve':'lip_ribbon_interior','radial_coordinate':float(radial),'phase':phase}
    source_band=set(sum(source_loops,[]));allowed=np.flatnonzero(np.array([all(int(source_ids[i]) in source_band for i in t) for t in triangles]))
    return results,allowed,{'canonical_band_vertices':domain,'source_rings':source_loops,'status':'unvalidated'}


def enclosed_quad_patch(quads,boundary,seed_vertex):
    """Flood the cavity side of a closed anatomical loop, never through its rim."""
    barriers={tuple(sorted(e)) for e in zip(boundary,boundary[1:]+boundary[:1])}
    edge_faces={}
    for k,q in enumerate(quads):
        for a,b in zip(q,q[1:]+q[:1]):edge_faces.setdefault(tuple(sorted((a,b))),[]).append(k)
    start=next(k for k,q in enumerate(quads) if seed_vertex in q)
    stack=[start];visited=set()
    while stack:
        k=stack.pop()
        if k in visited:continue
        visited.add(k)
        if len(visited)>2000:raise ValueError('Cavity flood escaped expected anatomical patch')
        q=quads[k]
        for a,b in zip(q,q[1:]+q[:1]):
            edge=tuple(sorted((a,b)))
            if edge not in barriers:stack.extend(j for j in edge_faces[edge] if j not in visited)
    return sorted(visited)

"""Read-only inventory and core-delta loading. No external assets or program execution."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rig_inventory(root,vertices,source_ids):
    root=Path(root);path=root/'data/rigs/rigify/rig.human.json';rig=json.loads(path.read_text());lookup={int(i):j for j,i in enumerate(source_ids)};items=[]
    prefixes=('lip.','chin','jaw','temple.','forehead.','nose','eye.','brow.')
    for name,bone in sorted(rig['bones'].items()):
        if not name.startswith(prefixes):continue
        for end in ['head','tail']:
            definition=bone[end];strategy=definition.get('strategy');ids=definition.get('vertex_indices',[definition.get('vertex_index')])
            if strategy not in ['VERTEX','MEAN'] or not all(i in lookup for i in ids):continue
            points=vertices[[lookup[i] for i in ids]]
            items.append({'name':name+'.'+end,'bone':name,'end':end,'strategy':strategy,'source_vertex_ids':ids,
                'position':points.mean(0).tolist(),'support_span':float(np.linalg.norm(np.ptp(points,axis=0))),
                'semantic_scope':'author-defined rig endpoint; NOT a MediaPipe landmark or verified vermilion border',
                'mean_is_surface_point':strategy=='VERTEX','exact_mediapipe_correspondence':False})
    extra_path=root/'entities/socketobject/_extra_vertex_groups.py';groups={}
    for node in ast.parse(extra_path.read_text()).body:
        if not isinstance(node,ast.Assign):continue
        for target in node.targets:
            try:
                if isinstance(target,ast.Subscript) and isinstance(target.value,ast.Subscript) and ast.literal_eval(target.value.slice)=='basemesh':groups[ast.literal_eval(target.slice)]=ast.literal_eval(node.value)
            except (TypeError,ValueError):continue
    metadata=json.loads((root/'data/mesh_metadata/basemesh_vertex_groups.json').read_text())
    return {'rig_endpoints':items,'extra_group_sizes':{k:len(v) for k,v in groups.items()},
        'metadata_group_names':list(metadata),'metadata_warning':'joint-* groups are helper cubes/pivots, not skin landmarks',
        'lips_group_warning':'418-vertex editing region includes oral/perioral support; no external vermilion annotation',
        'default_position_warning':'rig default_position coordinates were not used; actual source vertex IDs determine geometry',
        'native_rig_support_source_ids':sorted({i for item in items for i in item['source_vertex_ids']}),
        'source_hashes':{relative:sha(root/relative) for relative in ['data/3dobjs/base.obj','data/rigs/rigify/rig.human.json','data/mesh_metadata/basemesh_vertex_groups.json','data/mesh_metadata/hm08_config.json','data/mesh_metadata/hm08.mirror','entities/socketobject/_extra_vertex_groups.py','data/uv_layers/lips_solid.json.gz']},
        'license_scope':'base/rig/targets/JSON mesh data: bundled graphical CC0 assets; inspected Python module is GPL logic, not vendored/executed',
        'license_source':'https://raw.githubusercontent.com/makehumancommunity/mpfb2/master/LICENSE.md',
        'exact_mediapipe_correspondence_found':False}


class CoreTargets:
    def __init__(self,root,source_ids):
        self.root=(Path(root)/'data/targets').resolve();self.lookup={int(x):i for i,x in enumerate(source_ids)};self.cache={};self.provenance={}
    def load(self,relative):
        if relative in self.cache:return self.cache[relative]
        path=(self.root/relative).resolve()
        if not path.is_relative_to(self.root):raise ValueError('Target path escapes bundled core')
        result=np.zeros((len(self.lookup),3),float)
        for line in gzip.decompress(path.read_bytes()).decode().splitlines():
            fields=line.split()
            if not fields or line.lstrip().startswith('#'):continue
            if len(fields)!=4:raise ValueError('Malformed target')
            index=int(fields[0]);value=np.array(list(map(float,fields[1:])))
            if not np.isfinite(value).all():raise ValueError('Nonfinite target')
            if index in self.lookup:result[self.lookup[index]]=.1*value[[0,2,1]]*[1,-1,1]
        self.cache[relative]=result;self.provenance[relative]={'sha256':sha(path),'affected_body_vertices':int(np.count_nonzero(np.linalg.norm(result,axis=1))),
            'license':'bundled CC0 core target','semantic_scope':'displacement field, not a landmark or segmentation annotation'}
        return result
    def cases(self,vertices):
        identities={'base':[], 'round':[('head/head-round.target.gz',.5)], 'narrow':[('head/head-scale-horiz-decr.target.gz',.5)],
            'chin':[('chin/chin-height-incr.target.gz',.5)],'temple':[('forehead/forehead-temple-incr.target.gz',.5)],
            'wide':[('head/head-scale-horiz-incr.target.gz',.5)],'tall':[('head/head-scale-vert-incr.target.gz',.5)],'lip_volume':[('mouth/mouth-lowerlip-volume-incr.target.gz',.4)]}
        ex='expression/units/caucasian/'
        expressions={'neutral':[],'mouth_open':[(ex+'mouth-open.target.gz',.35)],'corner_pull':[(ex+'mouth-corner-puller.target.gz',.35)],
            'eye_close':[(ex+'eye-left-closure.target.gz',.35),(ex+'eye-right-closure.target.gz',.35)],
            'open_pull':[(ex+'mouth-open.target.gz',.25),(ex+'mouth-corner-puller.target.gz',.25)]}
        cases=[]
        for identity,di in identities.items():
            for expression,de in expressions.items():
                spec=di+de;delta=sum((weight*self.load(name) for name,weight in spec),start=np.zeros_like(vertices))
                cases.append({'name':identity+'/'+expression,'identity':identity,'expression':expression,'targets':[{'path':name,'weight':weight} for name,weight in spec],'vertices':vertices+delta})
        return cases

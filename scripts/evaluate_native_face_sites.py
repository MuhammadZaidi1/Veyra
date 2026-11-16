"""Evaluate native rig-support IDs without inventing anatomical or MediaPipe labels."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.makehuman_semantics import CoreTargets

def main():
    p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--project',type=Path,required=True);p.add_argument('--prototype',type=Path,required=True);a=p.parse_args();output=a.result/'native-rig-support-evaluation.json'
    if output.exists():raise ValueError('Preserve existing native evaluation; use a fresh result')
    inventory=json.loads((a.result/'semantic-inventory.json').read_text());source=np.array(inventory['native_rig_support_source_ids']);g=np.load(a.project/'surface.npz');lookup={int(x):i for i,x in enumerate(g['source_vertex_ids'])};indices=np.array([lookup[int(i)] for i in source]);root=Path(json.loads((a.prototype/'report.json').read_text())['mpfb_path']);reader=CoreTargets(root,g['source_vertex_ids']);checks=[]
    for case in reader.cases(g['world_xyz']):
        points=case['vertices'][indices];dist=np.linalg.norm(points[:,None]-points[None,:],axis=2);mask=np.triu(np.ones_like(dist,dtype=bool),1);pairs=np.argwhere(mask&(dist<.0002)).tolist();checks.append({'case':case['name'],'finite':bool(np.isfinite(points).all()),'collision_pairs':pairs,'minimum_distinct_site_distance':float(dist[mask].min())})
    sites=[]
    for index,source_id in enumerate(source):
        roles=[row['name'] for row in inventory['rig_endpoints'] if int(source_id) in row['source_vertex_ids']]
        sites.append({'native_site_index':index,'source_vertex_id':int(source_id),'rig_support_roles':roles,'anatomical_landmark_label':None})
    report={'schema':'veyra-native-rig-support-evaluation-v1','status':'evaluation_only_no_training_labels','site_count':len(sites),'sites':sites,'case_checks':checks,'sparse_faces':None,'warning':'These are actual rig-support source vertices. They are not 62 standard anatomical landmarks and must not be relabeled as 468 MediaPipe points. No synthetic triangulation is supplied.','requires_new_model_and_geometry_contract':True}
    output.write_text(json.dumps(report,indent=2));print('NATIVE_SITES',len(sites),'collision cases',sum(bool(c['collision_pairs']) for c in checks))
if __name__=='__main__':main()

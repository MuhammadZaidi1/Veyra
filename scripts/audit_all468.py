"""Audit masks and retain every original candidate; never approve an index by default."""
import argparse,json
from pathlib import Path
from scripts.audit_468_evidence import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Use a fresh output')
    mapping=Path('runs/veyra3d/annotation-project-001/base-map.json');base=json.loads(mapping.read_text());previous=json.loads(Path('runs/veyra3d/468-evidence-001/report.json').read_text())
    if sha(mapping)!=previous['baseline_sha256']:raise ValueError('Source changed; evidence must be rechecked')
    ledger=json.loads(Path('runs/veyra3d/468-evidence-001/point-evidence.json').read_text());rows=[]
    if len(base['slots'])!=468 or [s['veyra_index'] for s in base['slots']]!=list(range(468)):raise ValueError('Invalid ordering')
    for slot,evidence in zip(base['slots'],ledger):
        rows.append({'index':slot['veyra_index'],'classification':'candidate-only','disputed':evidence['disputed'],
            'reason':evidence['reason'],'eligible':False,'surface_triangle':slot['source_triangle']})
    a.output.mkdir(parents=True)
    (a.output/'report.json').write_text(json.dumps({'count':468,'eligible':0,'status':'blocked','rows':rows},indent=2))

if __name__=='__main__':main()

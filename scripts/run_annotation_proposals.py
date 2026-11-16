"""Portable proposal pipeline launcher; original maps and review decisions are never overwritten."""
import argparse,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_blender_prototype import locate_blender

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['propose','open']);p.add_argument('--project',type=Path,required=True);p.add_argument('--prototype',type=Path,default=Path('runs/veyra3d/blender-prototype-002'));p.add_argument('--reference',type=Path,default=Path('third_party/mediapipe-reference'));p.add_argument('--audit',type=Path);p.add_argument('--controls',type=Path);p.add_argument('--output',type=Path);p.add_argument('--render-checks',action='store_true');p.add_argument('--blender');a=p.parse_args();scripts=Path(__file__).resolve().parent;blender=locate_blender(a.blender)
    if a.action=='open':subprocess.Popen([blender,'--factory-startup',str((a.project/'annotation.blend').resolve()),'--python',str(scripts/'blender_proposal_ui.py')]);return
    if a.output is None:p.error('propose requires --output with a fresh directory')
    command=[blender,'--background','--factory-startup','--python-exit-code','1','--python',str(scripts/'blender_propose_annotations.py'),'--','--project',str(a.project.resolve()),'--prototype',str(a.prototype.resolve()),'--reference',str(a.reference.resolve()),'--output',str(a.output.resolve())]
    if a.controls:command.extend(['--controls',str(a.controls.resolve())])
    if a.audit:command.extend(['--audit',str(a.audit.resolve())])
    subprocess.run(command,check=True,timeout=300)
    if a.render_checks:subprocess.run([blender,'--background','--factory-startup','--python-exit-code','1','--python',str(scripts/'blender_render_proposals.py'),'--','--project',str(a.output.resolve()),'--prototype',str(a.prototype.resolve())],check=True,timeout=300)
    subprocess.run([sys.executable,str(scripts/'visualize_annotation_proposals.py'),'--project',str(a.output.resolve())],check=True,timeout=120)
if __name__=='__main__':main()

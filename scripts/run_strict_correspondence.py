"""Reproduce strict correspondence study in fresh ignored output; no training or approval."""
import argparse,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_blender_prototype import locate_blender

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,default=Path('runs/veyra3d/annotation-project-001'));p.add_argument('--prototype',type=Path,default=Path('runs/veyra3d/blender-prototype-002'));p.add_argument('--proposal',type=Path,default=Path('runs/veyra3d/annotation-proposals-005'));p.add_argument('--reference',type=Path,default=Path('third_party/mediapipe-reference'));p.add_argument('--audit',type=Path,default=Path('runs/veyra3d/anatomy-audit-001'));p.add_argument('--output',type=Path,required=True);p.add_argument('--blender');p.add_argument('--render-checks',action='store_true');a=p.parse_args();scripts=Path(__file__).resolve().parent;blender=locate_blender(a.blender)
    command=[blender,'--background','--factory-startup','--python-exit-code','1','--python',str(scripts/'blender_strict_correspondence.py'),'--']
    for flag in ['project','prototype','proposal','reference','audit','output']:command.extend(['--'+flag,str(getattr(a,flag).resolve())])
    subprocess.run(command,check=True,timeout=300)
    subprocess.run([sys.executable,str(scripts/'evaluate_native_face_sites.py'),'--result',str(a.output.resolve()),'--project',str(a.project.resolve()),'--prototype',str(a.prototype.resolve())],check=True,timeout=120)
    subprocess.run([sys.executable,str(scripts/'evaluate_lip_weight_interfaces.py'),'--result',str(a.output.resolve()),'--project',str(a.project.resolve()),'--prototype',str(a.prototype.resolve()),'--audit',str(a.audit.resolve())],check=True,timeout=120)
    if a.render_checks:subprocess.run([blender,'--background','--factory-startup','--python-exit-code','1','--python',str(scripts/'blender_render_strict.py'),'--','--result',str(a.output.resolve()),'--project',str(a.project.resolve()),'--prototype',str(a.prototype.resolve())],check=True,timeout=180)
    subprocess.run([sys.executable,str(scripts/'visualize_strict_correspondence.py'),'--result',str(a.output.resolve()),'--project',str(a.project.resolve())],check=True,timeout=120)
if __name__=='__main__':main()

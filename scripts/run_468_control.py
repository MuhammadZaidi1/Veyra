"""One command for a fresh 468 evidence audit and canonical-only control. Never trains."""
import argparse,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_blender_prototype import locate_blender

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--blender');a=p.parse_args();scripts=Path(__file__).resolve().parent
    subprocess.run([sys.executable,str(scripts/'audit_468_evidence.py'),'--output',str(a.output.resolve())],check=True)
    subprocess.run([locate_blender(a.blender),'--background','--factory-startup','--python-exit-code','1','--python',str(scripts/'blender_canonical_468_control.py'),'--','--reference',str((scripts.parent/'third_party/mediapipe-reference').resolve()),'--output',str((a.output/'canonical-control').resolve())],check=True)
    subprocess.run([sys.executable,str(scripts/'finalize_canonical_468_control.py'),'--control',str((a.output/'canonical-control').resolve())],check=True)
if __name__=='__main__':main()

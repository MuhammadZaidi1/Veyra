"""Windows-friendly one-face Blender launcher and independent PNG projection check.

Use the existing research venv (NumPy/Pillow). No downloads or training.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw


def locate_blender(explicit=None):
    if explicit:
        path=Path(explicit)
        if not path.is_file():
            raise ValueError(f'Blender executable missing: {path}')
        return str(path.resolve())
    found=shutil.which('blender')
    if found:
        return found
    program_files=os.environ.get('ProgramFiles')
    candidates=sorted((Path(program_files)/'Blender Foundation').glob('Blender 5.3*/blender.exe')) if program_files else []
    if len(candidates) != 1:
        raise ValueError('Pass --blender with the full path to blender.exe')
    return str(candidates[0])


def verify_raster(image, pixel_xy, tolerance=2.0):
    """Match predicted locations to cyan pixels in a separately rendered marker PNG."""
    pixel_xy=np.asarray(pixel_xy,dtype=float)
    if pixel_xy.ndim != 2 or pixel_xy.shape[1] != 2 or not len(pixel_xy) or not np.isfinite(pixel_xy).all():
        raise ValueError('Expected nonempty finite Nx2 pixel coordinates')
    array=np.asarray(image.convert('RGB'))
    if array.shape[:2] != (512,512):
        raise ValueError('Expected 512x512 diagnostic render')
    cyan=(array[:,:,0]<80)&(array[:,:,1]>160)&(array[:,:,2]>160)
    yy,xx=np.nonzero(cyan)
    if len(xx)<12:
        raise ValueError('Diagnostic render has too few cyan marker pixels')
    errors=[]
    for x,y in pixel_xy:
        distance=np.hypot(xx-x,yy-y)
        errors.append(float(distance.min()))
    if max(errors)>tolerance:
        raise ValueError(f'Raster/projection disagreement: {max(errors):.3f}px')
    return errors


def make_overlay(output):
    output=Path(output)
    report=json.loads((output/'report.json').read_text())
    with Image.open(output/'coordinate-probes.png') as probes:
        errors=verify_raster(probes,report['diagnostic_pixel_xy'])
    with Image.open(output/'face.png') as source:
        overlay=source.convert('RGB')
    draw=ImageDraw.Draw(overlay)
    for number,(x,y) in enumerate(report['diagnostic_pixel_xy']):
        draw.ellipse((x-4,y-4,x+4,y+4),outline=(0,255,255),width=1)
        draw.text((x+5,y-5),str(number),fill=(255,255,0))
    draw.rectangle((0,0,511,40),fill=(0,0,0))
    draw.text((8,5),'DIAGNOSTIC SURFACE POINTS - NOT 468 LANDMARKS',fill=(255,255,0))
    draw.text((8,22),'Anatomical correspondence: UNMAPPED / NOT TRAINING READY',fill=(255,255,255))
    overlay.save(output/'overlay.png')
    report.update(status='passed_coordinate_checks_anatomy_unmapped',
        raster_max_distance_px=max(errors), raster_distances_px=errors)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blender',type=Path)
    parser.add_argument('--output',type=Path,default=Path('runs/veyra3d/blender-prototype-001'))
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    script=Path(__file__).with_name('blender_face_prototype.py').resolve()
    command=[locate_blender(args.blender),'--background','--python-exit-code','1','--python',str(script),'--']
    if args.check_only:
        command.append('--check-only')
    else:
        if args.output.exists():
            parser.error('Output already exists; use a fresh --output path')
        command.extend(['--output',str(args.output.resolve())])
    # List arguments preserve Windows paths/spaces; never build a shell command.
    subprocess.run(command,check=True,timeout=300)
    if not args.check_only:
        report=make_overlay(args.output)
        print(json.dumps({'output':str(args.output.resolve()),'status':report['status'],
            'projection_error_px':report['projection_max_abs_error_px'],
            'raster_error_px':report['raster_max_distance_px'],'training_ready':False},indent=2))


if __name__=='__main__':
    main()

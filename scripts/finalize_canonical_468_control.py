"""Finalize canonical control diagnostics outside Blender (NumPy/Pillow), never training data."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image,ImageDraw,ImageFont
from scripts.run_blender_prototype import verify_raster

def main():
    p=argparse.ArgumentParser();p.add_argument('--control',type=Path,required=True);a=p.parse_args();root=a.control
    if (root/'report.json').exists():raise ValueError('Canonical diagnostic already finalized; preserve it')
    report=json.loads((root/'geometry-report.json').read_text());errors=verify_raster(Image.open(root/'coordinate-probes.png'),report['probe_pixel_xy']);report['independent_raster_max_error_px']=max(errors)
    try:font=ImageFont.truetype('DejaVuSans.ttf',18)
    except OSError:font=ImageFont.load_default(size=18)
    labels={0,1,4,10,13,14,17,33,61,133,152,234,263,291,362,454}
    for item in report['poses']:
        image=Image.open(root/(item['name']+'.png')).convert('RGB');d=ImageDraw.Draw(image);occluded=set(item['frame_occluded_indices'])
        for index,(x,y) in enumerate(item['pixel_xy']):
            color=(255,90,150) if index in occluded else (255,190,40);d.ellipse((x-1,y-1,x+1,y+1),fill=color)
            if index in labels:d.text((x+4,y-6),str(index),font=font,fill=color,stroke_width=1,stroke_fill='black')
        d.rectangle((0,0,1024,65),fill='black');d.text((10,5),item['name']+' - exact template indices; one synthetic identity',font=font,fill='white');d.text((10,34),'Orange: known template point. Pink: frame-occluded. No real-face accuracy claim.',font=font,fill='white');image.save(root/(item['name']+'-overlay.png'))
    for pose in ['left','front','right']:
        sheet=Image.new('RGB',(2048,1024));sheet.paste(Image.open(root/(pose+'-clear-overlay.png')),(0,0));sheet.paste(Image.open(root/(pose+'-frames-overlay.png')),(1024,0));sheet.save(root/(pose+'-paired-comparison.png'))
    (root/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ['landmark_count','render_triangle_count','paired_clear_frame_target_max_difference','projection_max_error_px','independent_raster_max_error_px','training_started']},indent=2))
if __name__=='__main__':main()

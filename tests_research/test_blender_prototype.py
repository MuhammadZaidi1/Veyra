"""No Blender launch, facial data, or downloads in these unit tests."""
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

from scripts.run_blender_prototype import locate_blender, verify_raster


class BlenderPrototypeTests(unittest.TestCase):
    def probes(self, offset=0):
        image=Image.new('RGB',(512,512))
        draw=ImageDraw.Draw(image)
        points=[(40+i*30,200) for i in range(12)]
        for x,y in points:
            draw.ellipse((x-2+offset,y-2,x+2+offset,y+2),fill=(0,255,255))
        return image,points

    def test_raster_check_accepts_matching_projection(self):
        image,points=self.probes()
        self.assertEqual(max(verify_raster(image,points)),0)

    def test_raster_check_rejects_shifted_render(self):
        image,points=self.probes(offset=10)
        with self.assertRaisesRegex(ValueError,'disagreement'):
            verify_raster(image,points)

    def test_raster_check_rejects_empty_image_and_invalid_points(self):
        image,points=self.probes()
        with self.assertRaisesRegex(ValueError,'too few'):
            verify_raster(Image.new('RGB',(512,512)),points)
        for bad in ([], [[float('nan'),1]], [[1,2,3]]):
            with self.assertRaisesRegex(ValueError,'finite Nx2'):
                verify_raster(image,bad)

    def test_explicit_windows_style_path_with_spaces_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'Blender Folder'/'blender.exe'
            path.parent.mkdir()
            path.touch()
            self.assertEqual(locate_blender(path),str(path.resolve()))
        with self.assertRaisesRegex(ValueError,'missing'):
            locate_blender(path)

    def test_missing_installation_has_setup_error(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch('shutil.which',return_value=None), mock.patch.dict('os.environ',{'ProgramFiles':directory}):
            with self.assertRaisesRegex(ValueError,'--blender'):
                locate_blender()

"""Opt-in real OpenGL tests: public generic model template, no camera or files."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENABLED = sys.platform == 'darwin' and os.environ.get('VEYRA_NATIVE_UI_TEST') == '1'


@unittest.skipUnless(ENABLED, 'set VEYRA_NATIVE_UI_TEST=1 on macOS for GPU surface checks')
class NativeSurfaceTests(unittest.TestCase):
    def test_real_shaded_surface_modes_light_rotation_and_tracking_loss(self):
        code = r'''
            import traceback
            import numpy as np
            from PySide6 import QtCore, QtGui, QtWidgets
            from veyra.core import DEFAULT_MODEL
            from veyra.diagnostics import reference_geometry
            from veyra.mesh import SURFACE_EDGES
            from veyra.render_gl import configure_surface_format
            from veyra.view3d import Face3DWindow
            points = reference_geometry(DEFAULT_MODEL)
            configure_surface_format()
            app = QtWidgets.QApplication([])
            window = Face3DWindow()
            window.resize(720,800)
            window.update_face(points, SURFACE_EDGES)
            window.show()
            outcome = [1]
            def frame():
                canvas=window.canvas
                canvas.makeCurrent()
                prior=canvas.gl.glGetError()
                canvas.paintGL()
                canvas.gl.glFinish()
                error=canvas.gl.glGetError()
                assert error == 0, f'GL error after paint: {error}, prior: {prior}, renderer: {canvas.render_error}'
                canvas.doneCurrent()
                image=canvas.grabFramebuffer().convertToFormat(QtGui.QImage.Format.Format_RGB888)
                assert not image.isNull()
                return np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.bytesPerLine())[:, :image.width()*3].reshape(image.height(), image.width(),3).copy()
            def checks():
                try:
                    assert window.accelerated, window.renderer_label.text()
                    canvas=window.canvas
                    assert canvas.render_error is None, canvas.render_error
                    images=[]
                    for mode in ('Solid','Wireframe','Solid + Wireframe'):
                        window.display_mode.setCurrentText(mode)
                        assert canvas.mode == mode
                        images.append(frame())
                    assert np.mean(np.abs(images[0].astype(float)-images[1])) > 2
                    assert np.mean(np.abs(images[0].astype(float)-images[2])) > .1
                    mask = np.min(images[0],axis=2)>55
                    assert np.count_nonzero(mask)>5000
                    assert len(np.unique(images[0][mask,0]))>40  # Smooth neutral shading has a tonal range.
                    window.display_mode.setCurrentText('Solid')
                    window.light_strength.setValue(0)
                    dark=frame()
                    window.light_strength.setValue(140)
                    lit=frame()
                    assert lit.mean()>dark.mean()+2
                    original=canvas.points.copy()
                    canvas.yaw=35;canvas.pitch=15;canvas.zoom=1.1
                    rotated=frame()
                    np.testing.assert_array_equal(canvas.points,original)
                    assert np.mean(np.abs(rotated.astype(float)-lit))>1
                    window.update_face(None,SURFACE_EDGES)
                    cleared=frame()
                    assert canvas.points is None
                    clear_count=np.count_nonzero(np.min(cleared,axis=2)>55)
                    # The centered no-face message has pixels too; the face area must vanish.
                    assert clear_count<np.count_nonzero(mask)*.05
                    assert np.count_nonzero(np.min(cleared[:cleared.shape[0]//3],axis=2)>55)==0
                    assert np.count_nonzero(np.min(cleared[cleared.shape[0]*2//3:],axis=2)>55)==0
                    window.update_face(points,SURFACE_EDGES)
                    frame()
                    assert canvas.render_error is None
                    assert canvas._draw_count>=6
                    window._fallback('synthetic unavailable context')
                    assert not window.accelerated
                    assert not window.display_mode.isEnabled()
                    assert window.canvas.points is not None
                    print('SURFACE_NATIVE_OK', flush=True)
                    outcome[0]=0
                except Exception:
                    traceback.print_exc()
                finally:
                    window.close()
                    app.exit(outcome[0])
            QtCore.QTimer.singleShot(500,checks)
            raise SystemExit(app.exec())
        '''
        environment = os.environ.copy()
        environment['QT_QPA_PLATFORM'] = 'cocoa'
        environment['PYTHONDONTWRITEBYTECODE'] = '1'
        environment['MPLCONFIGDIR'] = str(ROOT/'.cache/matplotlib')
        result=subprocess.run(['/usr/bin/sandbox-exec','-p','(version 1)(allow default)(deny network*)',
                               sys.executable,'-u','-c',textwrap.dedent(code)],cwd=ROOT,env=environment,
                              capture_output=True,text=True,timeout=20)
        diagnostic=result.stdout+'\n'+result.stderr
        self.assertEqual(result.returncode,0,diagnostic)
        self.assertIn('SURFACE_NATIVE_OK',result.stdout,diagnostic)
        self.assertNotIn('Traceback',result.stderr,diagnostic)

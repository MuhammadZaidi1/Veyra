# SPDX-License-Identifier: GPL-3.0-or-later
"""Background integration checks; never records human acceptance."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bpy
import blender_annotation_ui as ui
ui.register();original=ui.snapshot();assert len(original['annotations'])==56;assert all(e['decision']=='unreviewed' for e in original['annotations'])
assert bpy.ops.veyra_annotation.next()=={'FINISHED'}
obj=ui.active();saved=obj.location.copy();obj.location.x+=.001
try:ui.snapshot()
except ValueError:pass
else:raise AssertionError('Free movement was exported')
obj.location=saved
assert bpy.ops.veyra_annotation.snap()=={'FINISHED'};assert len(ui.snapshot()['annotations'])==56
source=bpy.data.objects['CC0_MakeHuman_base'];old=source.data.vertices[0].co.copy();source.data.vertices[0].co.x+=.001
try:ui.snapshot()
except ValueError:pass
else:raise AssertionError('Modified source was exported')
source.data.vertices[0].co=old
ui.source_unchanged()
assert bpy.ops.veyra_annotation.export()=={'FINISHED'}
print('PASS: 56 markers, selection, snap, free-move rejection, source-edit rejection, versioned export; no human acceptance recorded')

ui.unregister()

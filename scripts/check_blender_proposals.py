# SPDX-License-Identifier: GPL-3.0-or-later
"""Background integration of proposal evidence/compare UI; does not accept or save points."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bpy,numpy as np
import blender_annotation_ui as ui
import blender_proposal_ui as proposal_ui
ui.register();proposal_ui.register();snapshot=ui.snapshot();assert len(snapshot['annotations'])==56;assert all(e['decision']=='uncertain' and not e['reviewer'] for e in snapshot['annotations'])
assert bpy.ops.veyra_annotation.next()=={'FINISHED'};obj=ui.active();index=obj['veyra_index'];root=ui.root();g=ui.geometry();base=json.loads((root/'base-map.json').read_text());candidate=json.loads((root/'mapping-proposals.json').read_text())
for source,mapping in [('before',base),('after',candidate)]:
 assert bpy.ops.veyra_annotation.proposal_position(source=source)=={'FINISHED'};slot=mapping['slots'][index];p=np.array(slot['barycentric'])@g['vertices'][g['triangles'][slot['source_triangle']]];np.testing.assert_allclose(obj.location,p,atol=2e-7);assert obj['decision']=='uncertain' and not obj['reviewer']
assert not any(e['decision']=='accepted' for e in ui.snapshot()['annotations'])
for cls in [proposal_ui.VEYRA_PT_proposal_evidence,proposal_ui.VEYRA_OT_proposal_position]:bpy.utils.unregister_class(cls)
ui.unregister();print('PASS: 56 uncertain surface markers; evidence UI registration; original/proposed comparison; zero acceptance, no saved changes')

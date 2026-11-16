# SPDX-License-Identifier: GPL-3.0-or-later
"""Extra evidence panel and reversible original/proposed placement. No acceptance operator."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bpy,numpy as np
import blender_annotation_ui as ui

class VEYRA_OT_proposal_position(bpy.types.Operator):
    bl_idname='veyra_annotation.proposal_position';bl_label='Use original/proposed surface position';bl_options={'REGISTER','UNDO'}
    source:bpy.props.EnumProperty(items=[('before','Original','Restores original candidate, clears point decision'),('after','Proposed','Restores automated candidate, clears point decision')])
    def execute(self,context):
        obj=ui.active()
        if not obj:return {'CANCELLED'}
        folder=ui.root();filename='base-map.json' if self.source=='before' else 'mapping-proposals.json';slot=json.loads((folder/filename).read_text())['slots'][obj['veyra_index']];g=ui.geometry();point=np.array(slot['barycentric'])@g['vertices'][g['triangles'][slot['source_triangle']]]
        ui.set_position(obj,point,slot['source_triangle'],slot['barycentric']);obj['decision']='uncertain';obj['note']='Human comparison: '+self.source+' candidate, not anatomically accepted';return {'FINISHED'}

class VEYRA_PT_proposal_evidence(bpy.types.Panel):
    bl_label='Proposal evidence - NOT APPROVAL';bl_idname='VEYRA_PT_proposal_evidence';bl_space_type='VIEW_3D';bl_region_type='UI';bl_category='Veyra'
    def draw(self,context):
        layout=self.layout;layout.label(text='Full proposal has topology failures.',icon='ERROR');layout.label(text='Review 7 region cards before accepting points.')
        obj=ui.active()
        if not obj or not obj.get('proposal_evidence'):return
        r=json.loads(obj['proposal_evidence']);layout.label(text=r['classification']);layout.label(text='Anatomical confidence: UNVERIFIED')
        layout.label(text=f"Ensemble spread: {r['ensemble_max_spread']*1000:.2f} scene mm");layout.label(text=f"Morph reselect drift: {r['fixed_pool_reselection_max_drift']*1000:.2f} scene mm")
        layout.label(text=f"Curvature/morph ranks: {r['curvature_regional_percentile']:.2f}/{r['morph_gradient_regional_percentile']:.2f}")
        for flag in r['flags']:layout.label(text=flag.replace('_',' '),icon='QUESTION')
        row=layout.row()
        for source,label in [('before','Original position'),('after','Proposed position')]:op=row.operator('veyra_annotation.proposal_position',text=label);op.source=source
        layout.label(text='Comparing resets this point decision.')

def register():
    for cls in [VEYRA_OT_proposal_position,VEYRA_PT_proposal_evidence]:
        old=getattr(bpy.types,cls.__name__,None)
        if old:bpy.utils.unregister_class(old)
        bpy.utils.register_class(cls)

if __name__=='__main__':
    ui.register();register()

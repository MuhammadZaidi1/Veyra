# SPDX-License-Identifier: GPL-3.0-or-later
"""Blender N-panel: click-place surface markers, source-ID inspection, explicit point review."""
import json,sys
from pathlib import Path
import bpy
import blf
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from bpy_extras import view3d_utils

_HANDLER=None
_CACHE={}


def root():
    return Path(bpy.context.scene['veyra_annotation_root'])


def geometry():
    key=str(root())
    if _CACHE.get('key')!=key:
        archive=np.load(root()/'surface.npz',allow_pickle=False)
        _CACHE.clear();_CACHE.update(key=key,vertices=archive['world_xyz'],triangles=archive['triangles'],source_ids=archive['source_vertex_ids'])
        _CACHE['bvh']=BVHTree.FromPolygons(_CACHE['vertices'].tolist(),_CACHE['triangles'].tolist(),all_triangles=True)
    return _CACHE


def markers():
    region=bpy.context.scene.veyra_region
    objects=[o for o in bpy.data.objects if o.get('veyra_editable',False)]
    if region!='all':objects=[o for o in objects if o.get('annotation_group')==region or (region=='lips' and o.get('annotation_group','').startswith('lip')) or (region=='oval' and not o.get('annotation_group','').startswith('lip'))]
    return sorted(objects,key=lambda o:o['veyra_index'])


def active():
    obj=bpy.context.view_layer.objects.active
    return obj if obj and obj.get('veyra_editable',False) else None


def set_position(obj,point,index,weights):
    obj.location=point;obj['source_triangle']=int(index);obj['barycentric']=list(map(float,weights))
    obj['decision']='unreviewed';obj['reviewer']='';obj['note']='Moved: requires point review';obj.color=(1,.25,.1,1)


def weights_at(point,triangle):
    a,b,c=triangle;u,v=b-a,c-a
    vw=np.linalg.solve(np.array([[u@u,u@v],[u@v,v@v]]),[(point-a)@u,(point-a)@v])
    w=np.array([1-vw.sum(),*vw]);w=np.maximum(w,0);return w/w.sum()


def source_unchanged():
    g=geometry();obj=bpy.data.objects.get('CC0_MakeHuman_base')
    if obj is None or obj.modifiers:raise ValueError('Frozen source mesh missing or modified')
    xyz=np.array([tuple(obj.matrix_world@v.co) for v in obj.data.vertices])
    if xyz.shape!=g['vertices'].shape or not np.allclose(xyz,g['vertices'],atol=1e-8,rtol=0):raise ValueError('Source geometry changed; do not sculpt/remesh annotation source')
    obj.data.calc_loop_triangles();tri=np.array([tuple(t.vertices) for t in obj.data.loop_triangles],dtype=np.int32)
    if not np.array_equal(tri,g['triangles']):raise ValueError('Source triangle order changed')


def snapshot():
    source_unchanged();g=geometry();config=json.loads((root()/'project.json').read_text())
    entries=[]
    for obj in sorted((o for o in bpy.data.objects if o.get('veyra_editable',False)),key=lambda o:o['veyra_index']):
        if obj.parent is not None:raise ValueError('Annotation markers must not be parented')
        index=int(obj['source_triangle']);w=np.array(obj['barycentric']);p=w@g['vertices'][g['triangles'][index]]
        if np.linalg.norm(p-np.array(obj.location))>2e-7:raise ValueError(f'Marker {obj["veyra_index"]} was freely moved. Use Click Place or Snap before review/export.')
        decision=obj.get('decision','unreviewed');reviewer=obj.get('reviewer','').strip()
        if decision=='accepted' and not reviewer:raise ValueError('Accepted points require reviewer name')
        entries.append({'veyra_index':int(obj['veyra_index']),'annotation_group':obj['annotation_group'],
            'source_triangle':index,'barycentric':w.tolist(),'source_vertex_ids':g['source_ids'][g['triangles'][index]].tolist(),
            'decision':decision,'reviewer':reviewer,'note':obj.get('note','')})
    return {key:config[key] for key in ['base_mapping_sha256','asset_sha256','topology_sha256']}|{
        'schema':'veyra-surface-annotation-v1','status':'point_annotations_not_map_approval','annotations':entries}


class VEYRA_OT_next(bpy.types.Operator):
    bl_idname='veyra_annotation.next';bl_label='Next point'
    def execute(self,context):
        items=markers();current=active()
        if not items:return {'CANCELLED'}
        j=(items.index(current)+1)%len(items) if current in items else 0
        for obj in context.selected_objects:obj.select_set(False)
        obj=items[j];obj.hide_set(False);obj.select_set(True);context.view_layer.objects.active=obj
        return {'FINISHED'}


class VEYRA_OT_place(bpy.types.Operator):
    bl_idname='veyra_annotation.place';bl_label='Click-place selected on skin'
    def invoke(self,context,event):
        if not active() or context.area.type!='VIEW_3D':self.report({'ERROR'},'Select a marker in the 3D view');return {'CANCELLED'}
        self.marker=active().name;context.window_manager.modal_handler_add(self)
        context.area.header_text_set('Click the actual skin location. ESC cancels. Point review resets after moving.')
        return {'RUNNING_MODAL'}
    def modal(self,context,event):
        if event.type in ('ESC','RIGHTMOUSE'):
            context.area.header_text_set(None);return {'CANCELLED'}
        if event.type=='LEFTMOUSE' and event.value=='PRESS':
            region=next(r for r in context.area.regions if r.type=='WINDOW');xy=(event.mouse_x-region.x,event.mouse_y-region.y);rv=context.space_data.region_3d
            origin=view3d_utils.region_2d_to_origin_3d(region,rv,xy);direction=view3d_utils.region_2d_to_vector_3d(region,rv,xy)
            p,n,index,distance=geometry()['bvh'].ray_cast(origin,direction)
            if p is None:self.report({'WARNING'},'No source-skin hit');return {'RUNNING_MODAL'}
            weights=weights_at(np.array(p),geometry()['vertices'][geometry()['triangles'][index]])
            exact=weights@geometry()['vertices'][geometry()['triangles'][index]]
            set_position(bpy.data.objects[self.marker],exact,index,weights)
            context.area.header_text_set(None);return {'FINISHED'}
        return {'RUNNING_MODAL'}


class VEYRA_OT_snap(bpy.types.Operator):
    bl_idname='veyra_annotation.snap';bl_label='Snap freely moved marker to nearest skin'
    def execute(self,context):
        obj=active()
        if not obj:return {'CANCELLED'}
        g=geometry();p,n,index,distance=g['bvh'].find_nearest(obj.location)
        if distance>.005:self.report({'ERROR'},'Too far from skin; use click-place instead');return {'CANCELLED'}
        w=weights_at(np.array(p),g['vertices'][g['triangles'][index]])
        set_position(obj,w@g['vertices'][g['triangles'][index]],index,w);return {'FINISHED'}


class VEYRA_OT_decision(bpy.types.Operator):
    bl_idname='veyra_annotation.decision';bl_label='Record point decision'
    decision:bpy.props.EnumProperty(items=[('accepted','Accept this point','Human point decision only'),('uncertain','Uncertain','Keep for review'),('rejected','Reject','Keep for review')])
    def execute(self,context):
        obj=active()
        if not obj:return {'CANCELLED'}
        if not context.scene.veyra_reviewer.strip():self.report({'ERROR'},'Enter your reviewer name first');return {'CANCELLED'}
        g=geometry();p=np.array(obj['barycentric'])@g['vertices'][g['triangles'][int(obj['source_triangle'])]]
        if np.linalg.norm(np.array(obj.location)-p)>2e-7:self.report({'ERROR'},'Snap the marker onto skin first');return {'CANCELLED'}
        obj['decision']=self.decision;obj['reviewer']=context.scene.veyra_reviewer.strip();obj['note']=context.scene.veyra_note
        obj.color=(.2,1,.3,1) if self.decision=='accepted' else (1,.6,.1,1)
        return {'FINISHED'}


class VEYRA_OT_export(bpy.types.Operator):
    bl_idname='veyra_annotation.export';bl_label='Save versioned annotations (no map approval)'
    def execute(self,context):
        try:
            data=snapshot();folder=root()/'annotations';folder.mkdir(exist_ok=True)
            index=1
            while (folder/f'annotations-{index:04d}.json').exists():index+=1
            path=folder/f'annotations-{index:04d}.json';path.write_text(json.dumps(data,indent=2)+'\n')
            self.report({'INFO'},f'Saved {path.name}. Map remains unvalidated.')
        except (ValueError,KeyError) as error:self.report({'ERROR'},str(error));return {'CANCELLED'}
        return {'FINISHED'}


class VEYRA_OT_view(bpy.types.Operator):
    bl_idname='veyra_annotation.view';bl_label='Set annotation view'
    view:bpy.props.EnumProperty(items=[('front','Front',''),('profile','Profile',''),('oblique','Oblique','')])
    def execute(self,context):
        if context.area.type!='VIEW_3D':return {'CANCELLED'}
        target=Vector((0,-.04,.725));position={'front':Vector((0,-1.1,.725)),'profile':Vector((1.1,-.04,.725)),'oblique':Vector((.9,-.56,.725))}[self.view]
        region=context.space_data.region_3d;region.view_rotation=(target-position).to_track_quat('-Z','Y');region.view_distance=.45;region.view_location=target;region.view_perspective='ORTHO'
        return {'FINISHED'}


def filter_region(self,context):
    allowed={o.name for o in markers()}
    for obj in bpy.data.objects:
        if obj.get('veyra_editable',False):obj.hide_set(obj.name not in allowed)


def draw_source_ids():
    context=bpy.context;obj=active()
    if not obj or not context.scene.veyra_show_ids or context.area.type!='VIEW_3D':return
    g=geometry();p=np.array(obj.location);d=np.linalg.norm(g['vertices']-p,axis=1)
    selected=np.flatnonzero(d<context.scene.veyra_id_radius);selected=selected[np.argsort(d[selected])[:55]]
    region=context.region;rv=context.space_data.region_3d;font=0;blf.size(font,13)
    chosen=set(selected.tolist());edge_set=set()
    for tri in g['triangles']:
        for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
            if int(a) in chosen and int(b) in chosen:edge_set.add(tuple(sorted((int(a),int(b)))))
    for i in selected:
        xy=view3d_utils.location_3d_to_region_2d(region,rv,Vector(g['vertices'][i]))
        if xy is not None:blf.color(font,.9,.9,.9,1);blf.position(font,xy.x,xy.y,0);blf.draw(font,'V'+str(int(g['source_ids'][i])))
    for a,b in sorted(edge_set)[:45]:
        xy=view3d_utils.location_3d_to_region_2d(region,rv,Vector((g['vertices'][a]+g['vertices'][b])/2))
        if xy is not None:blf.color(font,1,.7,.15,1);blf.position(font,xy.x,xy.y,0);blf.draw(font,f'E{int(g["source_ids"][a])}-{int(g["source_ids"][b])}')


class VEYRA_PT_annotation(bpy.types.Panel):
    bl_label='Veyra surface annotation — UNVALIDATED';bl_idname='VEYRA_PT_annotation';bl_space_type='VIEW_3D';bl_region_type='UI';bl_category='Veyra'
    def draw(self,context):
        layout=self.layout;scene=context.scene
        layout.label(text='Point acceptance is NOT map approval.',icon='ERROR')
        layout.prop(scene,'veyra_region');row=layout.row()
        for view in ['front','profile','oblique']:op=row.operator('veyra_annotation.view',text=view.title());op.view=view
        layout.operator('veyra_annotation.next');obj=active()
        if obj:
            layout.label(text=f'MP {obj["veyra_index"]}: {obj["annotation_group"]}')
            layout.label(text='Decision: '+obj.get('decision','unreviewed'))
            layout.label(text=f'Source triangle {obj["source_triangle"]}')
            layout.operator('veyra_annotation.place');layout.operator('veyra_annotation.snap')
        layout.prop(scene,'veyra_show_ids');layout.prop(scene,'veyra_id_radius')
        layout.prop(scene,'veyra_reviewer');layout.prop(scene,'veyra_note')
        row=layout.row()
        for decision,label in [('accepted','Accept point'),('uncertain','Uncertain'),('rejected','Reject')]:op=row.operator('veyra_annotation.decision',text=label);op.decision=decision
        layout.operator('veyra_annotation.export')
        layout.label(text='Guides: project folder / reference-board.png')
        layout.label(text='G moves freely; Snap then re-review.')


CLASSES=[VEYRA_OT_next,VEYRA_OT_place,VEYRA_OT_snap,VEYRA_OT_decision,VEYRA_OT_export,VEYRA_OT_view,VEYRA_PT_annotation]


def register():
    global _HANDLER
    for cls in CLASSES:
        old=getattr(bpy.types,cls.__name__,None)
        if old:
            try:bpy.utils.unregister_class(old)
            except RuntimeError:pass
        bpy.utils.register_class(cls)
    bpy.types.Scene.veyra_region=bpy.props.EnumProperty(name='Review region',items=[(i,n,'') for i,n in [('all','All 56 points'),('lips','All external lip border'),('lip_upper','Upper border'),('lip_lower','Lower border'),('lip_corners','Mouth corners'),('oval','All face oval'),('forehead','Forehead'),('temples','Temples / cheeks'),('jaw','Jaw'),('chin','Chin')]],default='lips',update=filter_region)
    bpy.types.Scene.veyra_reviewer=bpy.props.StringProperty(name='Human reviewer')
    bpy.types.Scene.veyra_note=bpy.props.StringProperty(name='Point note / uncertainty')
    bpy.types.Scene.veyra_show_ids=bpy.props.BoolProperty(name='Nearby source vertex/edge IDs',default=False)
    bpy.types.Scene.veyra_id_radius=bpy.props.FloatProperty(name='ID radius',default=.008,min=.001,max=.025)
    if _HANDLER:bpy.types.SpaceView3D.draw_handler_remove(_HANDLER,'WINDOW')
    _HANDLER=bpy.types.SpaceView3D.draw_handler_add(draw_source_ids,(),'WINDOW','POST_PIXEL')
    filter_region(None,bpy.context)


def unregister():
    global _HANDLER
    if _HANDLER is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_HANDLER,'WINDOW');_HANDLER=None
    for name in ['veyra_region','veyra_reviewer','veyra_note','veyra_show_ids','veyra_id_radius']:
        if hasattr(bpy.types.Scene,name):delattr(bpy.types.Scene,name)
    for cls in reversed(CLASSES):bpy.utils.unregister_class(cls)
    _CACHE.clear()


if __name__=='__main__':register()

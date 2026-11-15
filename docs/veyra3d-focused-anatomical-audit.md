# Focused anatomical audit — 2026-11-16

**Disposition: unvalidated; mapping unchanged.** This pass inspected source quads,
source IDs, material-region metadata, 3D profiles, and same-camera canonical versus
MakeHuman views. No labels were relocated, approved or exported for training.

## Confirmed findings and supported defects

- **The broad lip shape is enforced by the selected source rings, not an overlay
  error or a loose alignment parameter.** Outer/inner boundaries are hard surface
  constraints. The chosen outer ring uses source 362 above and 5337 below; the
  inner rim uses 474/493. Changing global alignment alone cannot correct their
  shape while those constraints remain fixed.
- The mapped mouth is broader relative to outer eye-corner span: **0.625 versus
  canonical 0.552**. Upper-band height/mouth-width is **0.207 versus 0.120**;
  lower-band height/width is **0.208 versus 0.167**. These are real 3D proportion
  differences, not proof that every difference is anatomically wrong.
- The lower outer point 17 is on source **5337**, where the meridian turns backward
  into the lower lip/chin transition. Relative anterior displacement from inner
  point 14 is **-4.28% of mouth width**, versus **+2.66%** in the canonical mesh.
  This accounts for the recessed/deep appearance. Source 492 is almost the same
  height but farther forward; the geometry does not justify silently switching
  17 to it, or to higher ring 496, without a border annotation.
- **A confirmed classification weakness is calling the selected regular quad ring
  an anatomical vermilion border.** Neither loop closure nor the source material
  mask establishes that claim. The UV lip mask spans 394 quads, with 88 boundary
  vertices including perioral/oral regions; its boundary is not the selected loop
  and is not ground-truth vermilion segmentation. Thus the source anatomy-versus-
  misclassification question remains partly unresolved.
- The mapped face oval is materially taller/narrower: width/height **0.654 versus
  0.868**, and forehead-above-eye-line fraction **0.476 versus 0.319**. The largest
  discrepancies include 10,338,109,152,377,148. Point 10 is the unreviewed source
  vertex **1013** at z=0.828950; 152 is source **787** at z=0.616870. These heuristic
  extent controls require anatomical review, not automatic canonical-ratio fitting.

## Expected topology/shape differences: all seven remaining warnings traced

The canonical OBJ has 898 triangles; Veyra uses 852 with different triangulation
(434 unordered faces shared). The canonical has 18 triangles capping the mouth
aperture. **All six normal warnings are on those caps, not Veyra skin triangles.**

| Canonical triangle | Landmark indices | Normal angle |
|---:|---|---:|
| 888 | 311,312,402 | 93.56 degrees |
| 889 | 81,178,82 | 93.56 degrees |
| 892 | 310,311,318 | 96.68 degrees |
| 893 | 80,88,81 | 96.68 degrees |
| 894 | 318,324,310 | 96.07 degrees |
| 895 | 88,80,95 | 96.07 degrees |

They connect opposite sides of the aperture, rather than sampling MakeHuman skin
across that gap. The near-closed source mouth changes the cap planes relative to
the generic canonical mouth. A negative reference-normal dot product here is not
proof that a real source skin triangle has flipped.

Edge **13–14** likewise spans the aperture and is absent from Veyra's skin edges.
Its length is 0.001498 versus 0.006111 in the similarity reference: ratio **0.24517**,
just below the existing 0.25 warning threshold. Its endpoints are distinct surface
points on opposite rim edges, not duplicate anchors. Do not widen the mouth just
to cross that threshold.

Using Veyra's actual 852 triangles with the SAME fixed reference yields **0 normal
reversals and 0 extreme edge distortions**. Original cap warnings are retained in
audit.json; they were not removed by changing coordinates or thresholds.

The official 36-edge face oval exactly matches the boundary of the canonical face
patch and is also a boundary of Veyra's skin topology. It is not a full skull,
hairline annotation, or view-dependent silhouette. Subject shape differences are
expected. The existing 36 anchors are surface-attached, but that alone does not
validate their anatomical locations on a full MakeHuman head.

## Unresolved ambiguities and precise next fixes

1. **Diagnostic fix supported now:** report actual Veyra-skin metrics separately
   from canonical aperture-cap/shape metrics. The new audit does this without
   modifying historical results or the mapping.
2. **Mapping fix needs annotation first:** mark the upper/lower external vermilion
   border on the pinned MakeHuman mesh, especially the center and corners. Then
   rebuild the ordered outer curve/ribbon from that reviewed boundary. The inner
   rim assignments are geometrically coherent; this pass provides no reason to
   move 13/14 merely to resemble the canonical mouth opening.
3. Review source 1013/787 and the lateral oval controls against forehead, chin and
   cheek/temple semantics. Replace only controls shown to be wrong. Do not stretch
   the face oval to the head silhouette or force this identity to canonical ratios.
4. A closed-loop ring is a useful reproducible candidate, not anatomical authority.
   Human review must distinguish actual border mislabeling from this source face's
   genuine lip fullness. The eight earlier parameter-orientation warnings also
   remain unapproved; these are not resolved by the cap-topology finding.

No unconditional numerical replacement for the lip border or oval is supported
by the available annotation. Mapping approval remains withheld.

## Inspection artifacts and reproduction

All figures are in runs/veyra3d/anatomy-audit-001:

- comparison-front.png, comparison-oblique.png and comparison-right/left-profile.png:
  canonical reference versus unchanged MakeHuman mapping; identical cameras and
  global similarity scale, no local display warp. Orange canonical faces are caps.
- source-mouth-topology.png: actual source quad rings and meridian vertices,
  front and sagittal views, independent of rendered shading.
- sagittal-profile.png: equal-axis 3D depth/height profiles.
- six-cap-warnings.png: each affected landmark labeled; orange chords on the
  source are diagnostic cap connections, not source skin faces.
- audit.json: full angles, areas, ratios, supports, topology scopes and input SHA256.

    .\.venv-research\Scripts\python.exe -m scripts.audit_correspondence_anatomy --output runs/veyra3d/audit-next
    & 'C:\Program Files\Blender Foundation\Blender 5.3\blender.exe' --background --python-exit-code 1 --python scripts/blender_anatomical_audit.py -- --audit runs/veyra3d/audit-next --prototype runs/veyra3d/blender-prototype-002
    .\.venv-research\Scripts\python.exe -m scripts.audit_correspondence_anatomy --figures-only --output runs/veyra3d/audit-next

Mapping bytes are checked unchanged. The 468 ordering and barycentric attachment
format are preserved. Three focused regression checks passed, covering topology
classification, mapping integrity and overwrite refusal. No new assets were
fetched. Existing Apache-2.0 canonical and CC0 core-asset provenance remain intact.

Canonical reference: https://github.com/google-ai-edge/mediapipe/blob/02ca022d573c004c463d8c61eb3f8ce17e95a7fc/mediapipe/modules/face_geometry/data/canonical_face_model.obj
Connectivity: https://github.com/google-ai-edge/mediapipe/blob/02ca022d573c004c463d8c61eb3f8ce17e95a7fc/mediapipe/python/solutions/face_mesh_connections.py

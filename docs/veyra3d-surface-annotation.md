# Surface annotation workflow — map remains unvalidated

The isolated project is `runs/veyra3d/annotation-project-001/annotation.blend`.
Its frozen base is an exact copy of `correspondence-anatomical-final/mapping-candidates.json`.
No anatomical corrections have been made. `annotations/draft-0000.json` contains 56
unreviewed markers, not training labels. `annotations-0001.json` and `annotations-0002.json` are background
integration-test exports with no accepted points; begin reviewing from the blend.

## Minimal review sequence

From the repository root in PowerShell:

```powershell
.\.venv-research\Scripts\python.exe scripts/run_anatomical_annotation.py open
```

This registers the local UI explicitly without installing an add-on or enabling
Blender's global automatic script execution. Blender and MPFB already exist;
this frozen-mesh workflow needs no additional installation or MPFB execution.
Alternatively open the blend and run its `START_HERE.py` text in Scripting.
In the 3D viewport press N, choose **Veyra**, then **Front** and **Next point**.
The skin cannot be selected accidentally. Orange named sphere empties are the
editable markers, not extra geometry in the render.

1. Open `reference-board.png` and the relevant `point-cards/MP-NNN.png` alongside
   Blender. The board labels canonical reference versus unverified source mapping.
   These are synthetic geometry references, **not an independently annotated
   anatomical atlas**. Rendering/shading alone cannot establish a vermilion border.
2. Enter your name in Human reviewer. Start with lip points 0, 17, 61, 291, then
   forehead 10, chin 152, and paired temple/cheek points 21/251 and 234/454.
3. Inspect Front, Oblique and Profile, zoom locally, and enable Nearby source
   vertex/edge IDs when needed. `V` is the exported MakeHuman vertex ID;
   `Ea-b` identifies a frozen triangle edge by its endpoint source IDs, including
   triangulation diagonals. This does not certify an edge as an anatomical border.
4. **Click-place selected on skin**, then click the actual desired surface
   location. It raycasts the frozen source mesh and calculates triangle ID plus
   barycentric weights. Frontmost skin is chosen, so orbit first if hidden.
   ESC cancels. Alternatively G-move the marker, then **Snap** (maximum 5 mm).
   Nearest-surface snapping is a placement convenience, not an anatomical decision.
5. Inspect its depth and source support; **Accept point**, **Uncertain**, or
   **Reject**, with a note. Movement clears acceptance. Do not accept an ambiguous
   border because it follows a convenient mesh ring. Use Next point and region
   filters to review the remaining markers. Ctrl-S saves editing progress.
6. **Save versioned annotations** writes a new JSON without overwriting previous
   versions. Unattached freely moved markers and altered source geometry are
   refused. Acceptance applies to one point only, not the entire mapping.

The compact cards show eight nearest source vertices and their connecting edges;
the viewport provides wider inspection. Cards are front projections and may hide
occlusion/depth; use Blender's profile/oblique views as well. The whole-mouth
`mouth-source-*` images are dense topology context; individual cards are the
readable source-ID references. Guide lines in before/after images are straight
3D landmark chords, not curves continuously following the source surface.

## Exact human review queue

All **56** editable points require manual anatomical review:

- External upper border: 0, 37, 39, 40, 185, 267, 269, 270, 409.
- External lower border: 17, 84, 91, 146, 181, 314, 321, 375, 405.
- Commissures: 61, 291.
- Forehead: 10, 54, 67, 103, 109, 284, 297, 332, 338.
- Temples/cheeks: 21, 93, 127, 162, 234, 251, 323, 356, 389, 454.
- Jaw: 58, 132, 136, 149, 150, 172, 288, 361, 365, 378, 379, 397.
- Chin: 148, 152, 176, 377, 400.

These group names organize the queue; they do not validate the initial points.
For lips identify the external vermilion/skin transition, distinguish it from the
inner aperture, and verify lower-lip versus labiomental transition in profile.
For oval points compare canonical intended relative position and actual forehead,
temple, mandibular and chin surfaces; do not force a tall/narrow head into the
canonical silhouette. The oval is not a view-dependent silhouette to slide with
camera angle. Record unresolved hairline/temple and soft jaw-boundary ambiguity.
The other **412** anchors remain untouched and unvalidated, including inner lip
13/14 and eyelids. Border correction can expose problems in the neighboring lip
ribbon; it does not automatically validate or refit those neighboring points.

## Rebuild after actual human review

```powershell
.\.venv-research\Scripts\python.exe scripts/run_anatomical_annotation.py rebuild --annotations runs/veyra3d/annotation-project-001/annotations/annotations-NNNN.json --output runs/veyra3d/annotation-review-001
```

Use the actual exported filename and a fresh output directory. Only explicitly
accepted named-human entries replace anchors. All other slots remain unchanged;
there is no automatic ring substitution, mirroring, interpolation, global fit,
or propagation to neighboring landmarks. Exact 0..467 ordering is preserved.
The source geometry archive, original map and triangle hashes are checked; source
vertex IDs are regenerated from the frozen triangles. Candidate output remains
`candidate_unvalidated` and `human_reviewed:false`. Point decisions have separate
provenance. Save notes explaining unresolved border sections before continuing
with any later regional fitting work.

Outputs: candidate JSON, annotation snapshot, checks.json, front/profile/oblique
before-after panels. Checks compare duplicates, finite values, triangle orientation
and edge ratios on Veyra's 852 triangles; canonical 898-triangle checks are shown
separately to retain the six mouth-cap reversal warnings. Thresholds are unchanged
(0.25–4 edge ratio). Neither absence nor reduction of warnings approves anatomy.

The prepared `annotation-preview-002` is explicitly a **draft preview**: 468
anchors, 0 applied reviews, 0 displacement, 0 duplicate pairs and 0 attachment
failures. Before/after Veyra skin checks both have 0 normal reversals and 0 extreme
edge ratios; canonical checks both retain 6 reversals. This is a workflow test,
not evidence that the visible lip/oval defects were corrected.

## Reproducibility and validation

Preparation: `run_anatomical_annotation.py prepare --project <fresh-directory>`.
Reference cards: `annotation_reference_images.py <project-directory>`.
Preview only: `rebuild ... --draft-preview` explicitly permits zero accepted points.
Normal rebuild rejects zero accepted annotations. Before/after raw backgrounds use
recorded orthographic camera matrices; geometry is independently projected.

The checkpoint research suite includes all annotation/correspondence/audit tests.
Six pure unit tests exercise single-slot preservation, rejection of missing
reviewers/invalid weights/wrong provenance/duplicates, and explicit draft gating.
`check_blender_annotation.py` tests 56 markers, selection, nearest-surface snapping,
free-move rejection, source-edit rejection and versioned export in Blender 5.3
background mode. Mouse placement and anatomical interpretation still need human
interactive inspection. No anatomy acceptance was simulated on the real map.

Licenses are unchanged: MakeHuman core mesh CC0 and our procedural source scene;
MediaPipe canonical reference and landmark topology Apache 2.0 with existing
LICENSE/provenance in `third_party/mediapipe-reference/`. Keep that notice with
redistributed reference derivatives. Blender-side scripts use GPL-3.0-or-later.
No new assets were downloaded and no dataset or training was launched.


Checkpoint portability: preparation also accepts `--mapping <candidate.json>`
and `--prototype <prototype-directory>`; rebuild accepts
`--reference-geometry <audit-geometry.npz>`. Reference card generation accepts
`--audit <audit-directory>`. Defaults refer to historical local run names.
Generated runs are intentionally excluded from Git: on a fresh checkout recreate
the single-face prototype, candidate fit and focused audit first using the linked
documentation. The public canonical reference mesh and its notices are tracked.
Local JSON/blend files record resolved paths for local provenance; they are not
portable source files and must not be added to the repository.

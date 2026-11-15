# Semi-automated 468-point correspondence review

Historical first-fit documentation. The default fitting algorithm has since been
refined with surface-loop and lip-ribbon constraints. See
[anatomical refinement](veyra3d-anatomical-refinement.md) for the current method,
results and remaining defects. Use --fitting legacy --resolution 512 to reproduce
the original fitting policy below.

Status: CANDIDATE / UNVALIDATED. No training dataset, manifest or trainable label
NPZ is produced. All 468 anchors require human anatomical review, including those
without heuristic flags. This workflow does not certify an edited map as valid.

## Inputs and provenance

Uses the existing prototype-002 inspection.blend and surface-review.npz: 13,380
body-surface vertices, 26,756 triangles. Helper geometry is excluded. The integer
makehuman_source_vertex_id attribute and exported source_vertex_ids are checked
against the scene; the exported triangle array defines the stable triangle ID
namespace. IDs are NOT Blender polygon IDs or MediaPipe face IDs. Every anchor
also records its three original MakeHuman source vertex IDs for inspection.
Asset and topology hashes must match before fitting/replaying.

The canonical OBJ and official region connectivity were fetched from Google
MediaPipe at revision 02ca022d573c004c463d8c61eb3f8ce17e95a7fc, not a floating URL.
Exactly 468 OBJ v records are parsed without sorting, merging or renumbering;
slot i is canonical vertex i, zero-based. OBJ face references are converted from
one-based to zero-based indexing. Iris points are not included. Canonical
triangles are used for candidate quality checks only; Veyra's existing topology
in veyra/topology_data.py is not replaced by the reference's connectivity.

The pinned files, source URLs, SHA256 and unmodified Apache LICENSE are under
third_party/mediapipe-reference. Its README explains provenance and restoration.
The pinned public canonical OBJ is included through a narrow ignore exception.
The explicit fetch command below restores missing references with checksum checks.
No MediaPipe neural weights, photographs or facial dataset were fetched.

## Running on Windows (project root)

    .\.venv-research\Scripts\python.exe -m scripts.run_correspondence_review --fetch-reference
    .\.venv-research\Scripts\python.exe -m scripts.run_correspondence_review --output runs/veyra3d/correspondence-next

Only --fetch-reference uses network, and exits after fetching the pinned mesh,
connectivity and license. The current local reference is already present. Normal
fitting stays offline and uses the existing research NumPy/Pillow environment
plus Blender's bundled NumPy/mathutils. MPFB 2.0.18 and Blender 5.3 were tested.
Use --blender 'C:\path with spaces\blender.exe' for another installation.
Output directories must not exist. The subprocess timeout is five minutes.

Semi-automated iteration:

1. Inspect alignment-controls.json and numbered overlays. Thirteen heuristic
   coarse controls use canonical IDs and explicit MakeHuman source vertex IDs:
   nose, chin, forehead, four eye corners, two mouth corners, and four midline
   lip rows. They are unreviewed seed estimates, not validated landmark labels.
2. Copy/edit that JSON outside the original result folder. Select source skin
   vertices from inspection.blend / surface-review.npz. Confirm left/right from
   canonical coordinates, not ambiguous viewer-side names. Do not select joint
   cube centers or eyeball surfaces as skin correspondences.
3. Rerun with --controls path/to/edited-controls.json and a fresh --output. Broad
   alignment changes should be made through these controls, not hundreds of
   independent nearest-neighbor corrections.
4. For individual corrections, copy mapping-candidates.json, edit source_triangle
   and barycentric weights for the affected slots, and pass --anchors path/to/map.
   The weights must be finite, nonnegative and sum to one. Source triangle must
   index the frozen exported triangle array. Recomputed source_vertex_ids replace
   any hand-edited source IDs. Ordering and asset/topology hashes are rechecked.
5. Every --anchors run recomputes all overlays, flags and deformation checks.
   It retains candidate_unvalidated and resets human_reviewed=false: a status
   string is not a certificate of anatomical review.

    .\.venv-research\Scripts\python.exe -m scripts.run_correspondence_review --controls runs/veyra3d/correspondence-002/alignment-controls.json --anchors runs/veyra3d/correspondence-002/mapping-candidates.json --output runs/veyra3d/correspondence-edited

--overlays-only rebuilds pictures from the EXISTING report's projected points.
It does not refit/recompute edited anchors: use --anchors for that purpose.
Do not edit input/output files in place while a run is executing.

## Algorithm

1. Convert canonical axes (x right, y up, z forward) to the MakeHuman scene
   convention (x right, y back, z up). No reflection is fitted afterward.
2. Choose skin vertices nearest the explicit default seed locations or consume
   user-edited source IDs. Fit a positive-scale, proper-rotation least-squares
   similarity transform with SVD using the 13 controls.
3. Fit a 3D polyharmonic r-kernel residual warp with an affine tail, normalized
   for numerical conditioning. This interpolates the coarse control displacements.
   It is NOT an anatomically constrained registration and can fold locally.
4. Project each warped canonical vertex to its nearest triangle on the restricted
   body head surface (z .59-.86, abs(x)<.12, y<.01 in this pinned scene). Excludes
   the separate procedural eyeballs, helper cages and body/neck beyond this ROI.
5. Compute triangle barycentric weights, clamp only tiny numerical negatives,
   and verify reconstruction against Blender's float32 BVH (tolerance 2e-7 scene
   units). Preserve slots 0..467 and the frozen original surface triangle IDs.
6. Flag projection distances >.008 scene units; near-duplicate anchors <.0002;
   anchors with a barycentric weight <.001; positive-y winding normals; and local
   connectivity distortion. Threshold names mention mm as a nominal .1-scaled
   MakeHuman convention, not a claim of measured/metric human accuracy.
7. Flag all official eye/lip/face-oval vertices plus heuristic nose/jaw regions
   for mandatory focused review regardless of numerical scores. Other slots also
   need review; no heuristic flag means only that the checks did not trigger.

Connectivity checks compare warped-reference and projected edge lengths (ratios
outside [.25,4]) and triangle normal reversals (dot product <0). They detect
folding/collapse, not necessarily the cause. Winding normals can be inconsistent
around cavities and are an uncertainty cue, not definitive posterior-skin proof.

## Deformation stability tests

The anchor triangle IDs and weights are held FIXED. No refitting per case.
Tests cover four identity shapes: base, core head-round at .6, head-width decrease
at .5, and nose-height increase at .5. These are three individual identity feature
morphs, not a diverse population or macro demographic model. Each named CC0 core
.target.gz is recorded with file path, SHA256 and weight. Target deltas use source
vertex IDs and the same OBJ-to-Blender axis/scale conversion as the prototype.

Each identity is tested in neutral and three authored deformation proxies:
smile, jaw opening and blink. There are 16 numerical cases. The proxies are smooth
procedural displacements, not imported MPFB expression assets, FACS-validated
expressions, a physiological rig, or production-quality expression tests. The
installed core expression directory has no built-in presets, so the workflow
never silently searches user/community libraries for replacements. Eye spheres
remain simplified and fixed; these tests are not eye/teeth collision validation.

For every case, evaluated Blender mesh coordinates are used. Checks include
finite anchors, support triangle degeneracy/inversion, face-connectivity normal
flips, extreme edge distortion and nonzero evaluated morph change. Numerical
stability means anchors stick to their triangles, not that they stay on the
correct anatomical lip/eyelid. More extreme identities and a validated expression
rig remain required before any training-label approval.

Nine inspection renders are generated: neutral front/left/right, three base
expression views, and three identity-neutral views. Each has all 468 numbered
anchors; side overlays intentionally show through-surface points for inspection
and are not visibility masks. Every orthographic projection is checked against
an independent camera-coordinate formula (tolerance .001 pixel). Fixed anatomical
contours are not view-dependent silhouette labels; reviewers must settle the
intended contour semantics before pose expansion.

## Outputs and final measured results

Main result folder: runs/veyra3d/correspondence-002. The earlier correspondence-001
is retained as a poorer first fit. correspondence-replay exercises edited-map
input and repeated checks. No original draft/scene is overwritten.

- mapping-candidates.json: exactly 468 candidate slots, source triangle/vertex IDs,
  barycentric weights, distances and review flags. All human_reviewed=false.
- alignment-controls.json: editable heuristic alignment controls.
- alignment-review.npz: original canonical, similarity-aligned, warped and
  projected coordinates for algorithm inspection, NOT trainable geometry NPZ.
- review-report.json: thresholds/results, 16 cases, input morph hashes, per-view
  pixel projections and camera errors.
- *-numbered.png: 1536x1536 overlays with all points/IDs; red flags are heuristics.
- region-eyes/other_eye/lips/nose/jaw/contour.png: enlarged numbered close-ups.
- inspection-sheet.png: overview of all nine views.
- review-queue.md: all 468 indices, triangle IDs and reasons for review.

Final results:
- Candidate anchors: 468; invalid IDs/weights/nonfinite anchor failures: 0.
- Reconstruction max error: 1.014e-7 scene units.
- Camera projection max error: 0.0000153 pixel across nine views.
- Heuristically flagged candidates: 297 (overlapping categories).
- Sensitive-region review: 222; large projection distance: 19; boundary anchors:
  92; winding-normal warnings: 43; near-duplicate candidates: 4.
- Initial projection connectivity: 6 reversed triangles and 6 extremely distorted
  edges, involving 24 candidate indices. These are unresolved quality failures.
- 16 deformation cases: 0 failed support triangles, 0 additional face normal flips,
  0 extreme deformation-edge distortions. The projection's pre-existing defects
  remain; they are not erased by passing the subsequent stability checks.

Visual inspection found central lip rows not reliably matching the rendered
mouth seam. Eye boundaries also need correction/confirmation. These results are
not ready to approve for training, even with zero numerical anchor failures.

## Exactly what requires human review

Review ALL 468 semantics and the 13 coarse control assignments. Prioritize:
upper/lower and inner/outer lip rows (including center 13/14), mouth corners and
cavity-facing projections; eye corners, upper/lower eyelids and pupil/skin mixups;
nose tip/ala/nostril boundaries; chin/jaw/temple/forehead contour locations and
left-right conventions. Resolve the 6 reversed triangles, 6 distorted edges,
4 near-duplicate candidates and remaining flagged anchors. Inspect front and side
views across actual morphs and better expression rigs, not just neutral overlays.

Manual inspection may require different controls or region-constrained
registration rather than local edits alone. Approval needs a named reviewer,
versioned reviewed mapping, corrected overlays, documented contour/visibility
semantics and retesting. This tool intentionally has no automatic approve or
training export switch. Nothing here establishes real-world face accuracy.

## Licenses

MediaPipe canonical geometry/region connectivity: Apache License 2.0. Preserve
upstream LICENSE, copyright headers, provenance and applicable notices when
sharing the reference or derived mappings; mark modifications. Upstream URLs:
https://github.com/google-ai-edge/mediapipe/blob/02ca022d573c004c463d8c61eb3f8ce17e95a7fc/LICENSE
https://github.com/google-ai-edge/mediapipe/blob/02ca022d573c004c463d8c61eb3f8ce17e95a7fc/mediapipe/modules/face_geometry/data/canonical_face_model.obj

MakeHuman/MPFB bundled base and named core morphs: CC0 assets; no community
assets used. MPFB program source is GPLv3. Blender-side automation is marked
GPL-3.0-or-later. Data/reference terms do not automatically relicense training
weights; this workflow creates no weights and makes no claim about future
unreviewed third-party assets.
https://github.com/makehumancommunity/mpfb2/blob/master/LICENSE.md

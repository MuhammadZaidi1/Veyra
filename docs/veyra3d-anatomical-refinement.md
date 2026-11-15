# Anatomical correspondence refinement (2026-11-16)

Subsequent [focused anatomical audit](veyra3d-focused-anatomical-audit.md) traced
the six fixed-reference warnings and edge 13-14 to canonical mouth caps absent
from Veyra skin topology. It also found unverified outer-lip and oval assignments.
The historical measurements below are retained; they do not establish approval.


The final output remains candidate_unvalidated, reviewer=null, training_ready=false,
with all 468 human_reviewed fields false. The workflow exports inspection artifacts
only. No dataset, manifest, trainable vertices/valid NPZ or AI training is created.

## Inspect the result

runs/veyra3d/correspondence-anatomical-final contains native 1536x1536 RGB renders
and numbered overlays, enlarged eye/lip/nose/jaw/outline close-ups, a lip_ribbon
close-up showing all 80 vermilion vertices, and a nine-view inspection sheet.
There are nine inspection renders, not 1536px upscales of the original 512px input.
The regional close-ups crop these native renders and enlarge them for readable
IDs; enlargement does not add physical detail. All 468 points remain numbered
with their canonical MediaPipe indices. Blue curves are the official boundary
connections. All source positions are attached to MakeHuman triangles.

## Root causes established from geometry and overlays

The old controls attached both 13 and 14 to the upper-lip strip. Old 17, intended
as the lower outer lip, was actually on what is now the upper inner-lip seam.
Nearest-surface fitting also allowed intermediate lip rows to collapse onto rim
anchors. Some chin/lower-lip points selected INNER ORAL CAVITY walls instead of
external skin: winding direction alone did not reliably exclude those walls.

These were 3D surface-assignment errors, not just pixel alignment errors. Labels
were not translated in an overlay or detached from the surface to conceal them.

## Anatomical constraints and fitting

scripts/anatomical_constraints.py traces actual closed edge loops on the pinned
hm08 QUAD mesh using opposite-edge continuation at regular vertices. It fails
on changed/branching topology rather than silently inventing a curve. Original
MakeHuman vertex IDs are used, independently of the MediaPipe index namespace.

The provisional anatomical loop assignments are:

- Inner mouth rim: loop seeded by MakeHuman source edge (474,473); upper center
  source 474, lower center source 493. Canonical 13 and 14 attach to those separate
  rim centers. Neutral vertical separation is 0.0006700 scene units. This is not
  a measured human lip gap or an anatomical validation threshold.
- Outer vermilion border: source edge (362,357); upper center 362, lower center
  5337. Canonical 0 and 17 attach to the corresponding centers. Inner and outer
  corner IDs are different and follow their respective rings.
- Eyelid aperture: source edge (0,1) and its exact mirrored source edge. Skin-loop
  corners and upper/lower arcs replace seed coordinates near arbitrary skin or
  orbital surfaces. Source IDs 0/1 are NOT MediaPipe labels 0/1.

Seventy-two official canonical boundary vertices (40 lips and 32 eyelids) are
mapped monotonically along four anatomical arcs per loop: upper center, corner,
lower center, opposite corner. Arc length determines interpolation on source
edges. Each point becomes a frozen triangle ID and barycentric weights on that
exact edge, favoring the skin-facing supporting triangle at folds. These are
provisional anatomical correspondences requiring review, not certified landmarks.

The 40 intermediate lip vertices need more than boundary fitting. The canonical
lip annulus is recovered by triangle flood fill bounded by the two official lip
rings (80 total vertices). Harmonic radial and angular coordinates are solved on
that annulus. The source has seven successive quad loops spanning outer border
to inner rim. The intrinsic coordinates interpolate across those source ribbons,
then attach to REAL triangles restricted to the vermilion band. This prevents
intermediate rows from jumping to the other lip, chin or oral interior. All 112
feature targets remain on the original surface, with no modifications to the
rendered face mesh to accommodate labels.

The default registration is a smooth 3D polyharmonic residual warp constrained
by those 112 feature targets plus the nose/chin/forehead coarse controls (1,152,10).
In anatomical mode the old eye/lip coarse guesses are superseded by the loop/ribbon
constraints; the saved controls explicitly identify effective coarse indices.
A positive-area parameterization constraint adjusts only free registration
coordinates; anatomical pins stay fixed. It attempts to avoid folding, but has
8 unresolved small lip-region parameter triangles, which remain flagged.

Dense candidates project to external head skin. The oral cavity is explicitly
segmented by a flood bounded by the inner mouth ring (474 source quads) and is
excluded from external fitting. A winding filter also removes clearly back/cavity-
facing candidates. Winding is still only a heuristic: 18 anchors retain winding
warnings. Arbitrary perturbations, duplicate-separating jitter, image-space label
translations and per-view refitting are not used.

Biharmonic registration and parameter-ray projection were investigated with the
same checks; they did not improve the overall geometry and are not the selected
default. --registration/--projection expose them for reproducible investigation.
No metric thresholds were relaxed to make the selected map pass. Barycentric
positions are reevaluated exactly from stored weights after verifying the BVH
position tolerance, avoiding float32 roundoff in subsequent deformation metrics.

## Before/after metrics and interpretation

Baseline: runs/veyra3d/correspondence-002.
Selected result: runs/veyra3d/correspondence-anatomical-final.
Full machine-readable comparison: before-after.json in the selected result.

| Check | Before | After |
|---|---:|---:|
| Ordered, valid candidate anchors | 468 | 468 |
| Invalid ID/weight/nonfinite anchor failures | 0 | 0 |
| Projection-stage reversed triangles | 6 | 0 |
| Projection-stage extreme edge distortions | 6 | 0 |
| Near-duplicate anchor vertices (<0.0002 scene units) | 4 (2 pairs) | 0 |
| Fixed-reference normal-reversal warnings | 6 | 6 |
| Fixed-reference extreme edge distortions | 10 | 1 |
| Flagged candidates, overlapping categories | 297 | 312 |
| Mandatory focused anatomical review | 222 | 262 |
| Large projection-distance flags (>0.008 units) | 19 | 52 |
| Triangle-boundary flags | 92 | 139 |
| Positive-y winding warnings | 43 | 18 |
| Unresolved parameterization triangle warnings | Not measured | 8 (12 vertices) |
| Support failures over 16 deformation cases | 0 | 0 |
| Additional deformation flips/extreme edges | 0/0 | 0/0 |
| Upper/lower feature ordering failures | Not measured | 0 |
| Explicit deformation/view checks | Not measured | 48 |

Projection-stage metrics compare each map with ITS OWN anatomically fitted
reference before surface projection. They show that projection no longer folds
or collapses that fitted map; they are not proof of anatomical correctness.

To avoid hiding the effect of changing the fit, a second comparison uses EXACTLY
the old similarity-aligned canonical geometry for both maps. That common reference
still reports six lip-region normal reversals and one highly compressed 13-14
edge. The generic reference has a different mouth shape/opening from this nearly
closed source mouth. These discrepancies MUST be reviewed rather than dismissed
or removed from the results. before-after.json lists the affected indices.

Flag count is deliberately not optimized downward. Forty intermediate lip-row
vertices now receive mandatory review, exact edge anchors legitimately generate
boundary flags, and new parameter warnings are included. Large residual flags
increased: the global fit is still imperfect around facial contours/cheeks and
must not be represented as resolved. Thresholds are unchanged and flags are
uncertainty cues, not calibrated error probabilities.

## Stability and viewing angles

Four shapes (base, head-round .6, width decrease .5, nose-height increase .5) x
four expression states (neutral plus authored smile/jaw-opening/blink proxies)
give 16 cases. Same triangle IDs and barycentric weights are held fixed throughout;
there is no retargeting to hide drift. Tests use evaluated Blender surface points.

All 16 cases have zero support-triangle degeneracy/inversion, no additional normal
flips/extreme edge distortion, and correct upper/lower center ordering. Minimum
13-to-14 vertical separation remains positive (0.0006700 scene units). Every case
is checked at front, left and right views, totaling 48 camera/projection checks:
zero out-of-frame points, maximum analytical-vs-Blender discrepancy 0.0000458px.
Nine representative views are actually rendered at native 1536px, 32 samples.

These are bounded geometric stress tests. Expressions are smooth procedural
proxies, not validated FACS/MPFB facial rigs. The procedural eyeballs are fixed;
blinks and identity shapes may expose interpenetration/tear-duct realism issues.
Side overlays show all anchors, including hidden points; they are not visibility
or confidence masks. Fixed anatomical face-oval attachments do not track the
view-dependent apparent silhouette. No demographic or real-world accuracy claim
follows from these tests.

## Human review still required

Review all 468 points. In particular:

1. Confirm the selected source loops represent the intended inner mouth seam,
   outer vermilion boundary and eyelid margin; waterline vs visible lid edge is
   not automatically settled by a topological loop.
2. Review 13/14, cupid's bow, inner/outer mouth corners and intermediate lip rows
   in lip_ribbon close-ups and open-mouth cases. Resolve six common-reference
   normal warnings, the compressed 13-14 edge and eight unresolved parameter
   triangles (canonical triangle IDs 66,67,68,69,464,465,514,515).
3. Confirm inner/outer canthi and upper/lower lid arcs on both eyes, especially
   whether the tear-duct location and eyeball contacts are acceptable.
4. Inspect nostril/alar boundaries, nose-tip semantics, philtrum and the chin now
   that cavity attachments are removed. Absence of projection flips does not
   establish a correct anatomical nose correspondence.
5. Inspect jaw, temple and forehead contour attachments in side views. The
   increased residual flags show that fitting outside the constrained features
   is not yet reliable enough for automatic training approval.
6. Repeat on stronger identity changes and a proper reviewed facial rig before
   accepting a map. Document named reviewer, map version and contour semantics.

No command marks the output validated. Candidate anchor JSON preserves the exact
0..467 slot ordering, source_triangle, barycentric and source_vertex_ids fields.
No training export switch is introduced.

## Reproduction and licenses

    .\.venv-research\Scripts\python.exe -m scripts.run_correspondence_review --output runs/veyra3d/anatomical-next --fitting anatomical --registration polyharmonic --projection closest --resolution 1536
    .\.venv-research\Scripts\python.exe -m scripts.compare_correspondence --baseline runs/veyra3d/correspondence-002 --candidate runs/veyra3d/anatomical-next

Use --geometry-only for numerical inspection without PNG rendering. Default
fitting is now anatomical; --fitting legacy --resolution 512 retains the original
policy for investigation. Outputs must use fresh directories. Offline reference
checks and pinned asset hashes remain unchanged.

No new assets were downloaded. Existing MediaPipe reference/connectivity remains
Apache-2.0 with original notices, LICENSE and pinned provenance in
third_party/mediapipe-reference. The MakeHuman base/core targets remain CC0; only
our own procedural expression displacements were added. Blender-side automation
retains its GPL-3.0-or-later declaration. See the existing reference README and
MPFB asset license for the data/software distinction.

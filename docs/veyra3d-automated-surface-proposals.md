# Automated surface proposals — unvalidated

The reproducible result is `runs/veyra3d/annotation-proposals-005/`.
The original annotation project, original mapping and all existing human decisions
were preserved. No proposal is accepted. This pass does not establish anatomical
truth or produce training labels.

## What automation achieved, and what it cannot establish

Every one of the 56 editable points now has a surface-attached proposed position,
quantified evidence, alternative-hypothesis spread and deformation tests. The
remaining 412 slot dictionaries are exactly unchanged. The result does **not**
justify declaring most anatomy resolved: **2 points meet local numerical support
criteria (84 and 314); 54 remain ambiguous**. Even the supported pair is not a
confirmed vermilion-border annotation.

A full automatic replacement fails dense topology checks because external-border
moves conflict with the unchanged intermediate lip/jaw anchors. That is a real
failure, not an error to suppress. The practical reduction in effort is to inspect
**seven regional decisions**, then pin a few anatomical controls and refit, rather
than hand-drag every point independently. There is no evidence-backed guarantee
that only a handful of individual landmarks need review.

## Evidence and method

1. Verify the frozen project/base-map/archive/topology hashes and all pinned
   Apache-2.0 MediaPipe reference hashes. Check the MakeHuman base OBJ's explicit
   CC0 declaration and asset hash against the annotation project. Preserve 468
   MediaPipe indices, original source IDs and frozen triangle numbering.
2. Compute area-weighted source normals and normal variation per source edge
   length as a curvature proxy. Compute spatial gradients of displacement
   amplitudes from nine named bundled core morphs: upper/lower lip extension and
   height, mouth width, chin height/width, forehead height and temple shape.
   These are deformation cues, **not labeled vermilion segmentation**. Targets
   are read only from the installed MPFB core library; no downloaded/user assets.
3. Use canonical geometry as a prior fitted to current canthi/nose anchors
   (33,133,263,362,1). Translate its mouth to the current inner-rim centroid.
   Those source controls themselves remain unvalidated. No claim is made that
   MakeHuman should have the canonical identity's face-oval proportions.
4. Exclude back-facing/head-external triangles and the topologically enclosed
   oral cavity (474 quads, traced from the existing inner rim). This is a surface
   eligibility rule, not an assertion that a neighboring quad ring is the outer
   lip border. Search actual vertices, edge midpoints and triangle centroids,
   plus continuous nearest-surface projections of original/interpolated priors.
5. Jointly select upper/lower mouth arcs with shared commissures and a closed
   oval using dynamic programming. Enforce mouth x ordering, forward neighboring
   tangents and edge ratios 0.25–4, with a smooth 3D displacement penalty. Keep
   all points attached through exact triangle/barycentric supports. Search is
   bounded within 0.012 source units for lips and 0.015 for the oval; exceeding
   available evidence is flagged rather than silently widening the search.
6. Compare nine hypotheses: canonical/original blend 0.35, 0.65 or 1.0 and
   curvature/morph weights 0, 0.4 or 0.8. The central configuration is the proposal;
   maximum 3D spread quantifies sensitivity. It is **not a confidence probability**.
   Regional percentile ranks use midranks; a flat zero signal cannot score as
   strong evidence just because every sample ties.
7. Test 5 identities x 4 expression conditions: base, round, narrow, chin-height,
   temple changes; neutral, mouth-open, corner-pull and eye-closure core units.
   Identity amplitudes are 0.5, expression amplitudes 0.35. These are raw linear
   CC0 core deltas, **not a validated expression rig**. Also reselect curves on
   deformed geometry within the original candidate supports to measure cue drift,
   beyond the automatic attachment property of fixed barycentrics. This does not
   prove that the same semantic border remains correct on unseen identities.
8. Independently compare Blender camera projection with analytical projection
   in front, profile and oblique views for all 20 cases. Record occlusion rather
   than treating hidden far-side points as mismatched labels. Render nine
   diagnostic identity/expression/view combinations for visual review. Procedural
   eyeballs are unchanged; these images are inspection artifacts, not a dataset.

The complete scoring expression is in `scripts/blender_propose_annotations.py`.
Morph inputs, reference controls, hypothesis parameters and algorithm-source
hashes are recorded in `proposal-report.json`. No random fitting is used.

## Results and limits

| Check | Result |
|---|---:|
| Proposed editable anchors | 56 |
| Original slots preserved exactly | 412 |
| Invalid source attachments / duplicate pairs | 0 / 0 |
| Accepted points / anatomical approvals | 0 / 0 |
| Locally supported geometric candidates | 84, 314 |
| Ambiguous candidates | All other 54 editable indices |
| Full change: new skin normal reversals vs original | 23 |
| Full change: extreme edge ratios vs original | 6 |
| Fixed focused-audit reference: reversals before ? after | 0 ? 25 |
| Fixed focused-audit reference: extreme edges before ? after | 0 ? 8 |
| Supported-only two-point alternative: new reversals / extreme edges | 0 / 0 |
| Surface-support failures across 20 cases | 0 |
| Curve crossings across 20 cases | 0 |
| Additional skin reversals / extreme edges under tested deformations | 0 / 0 |
| Mouth ordering failure | narrow + mouth-open: 61/185 and 291/409 |
| Maximum independent projection disagreement | <0.000486 px |

The deformation comparison uses the already proposed neutral geometry; zero
additional reversals there does **not** cancel its 23 pre-existing new reversals.
Fixed-audit and original-map comparisons use different reference normals and are
reported separately. Eye-fit canonical-prior diagnostics are also retained in JSON;
that prior has its own shape/normal differences and is not an approved surface.
No numerical threshold was changed to conceal warnings.

Of the 56 points, 43 change by over 0.002 source units across scoring hypotheses,
22 exceed 0.002 units in fixed-pool deformation reselection, and 19 forehead/temple
points lack a uniquely identifiable geometric patch boundary. Fourteen proposals
approach local search limits. Counts overlap. Source units labeled "scene mm" in
cards are nominal 0.001 units, not measured human millimeters.

The locally supported pair moves about 0.002527 source units. Its curvature rank
is only about 0.604, morph-gradient rank 0.755, hypothesis spread about 0.001369,
and maximum morph reselection drift about 0.001309. These meet chosen heuristic
cutoffs (ranks >=0.6, nonzero morph signal, move >0.0005, spread/drift <=0.002,
no local flags). The curvature score is marginal; neither point is automatically
accepted. Symmetry was not counted as independent evidence.

The unchanged lower-lip/oval classification problem is not solved by a better
projection. Sparse outer corrections cannot safely update the entire dense lip
ribbon or chin patch while those interior anchors remain frozen. After anatomical
boundaries are reviewed, a separately tested regional refit may be needed; this
pass does not silently move the other 412 points to hide that problem.

## Minimal-effort review

Start with `review-board.png` or the seven files under `review-cards/`:

1. **Upper lip:** Is the proposed ridge the external vermilion border? Start at
   MP0, then Cupid's-bow and lateral arcs. Source ring selection alone is insufficient.
2. **Lower lip:** Distinguish the vermilion/skin boundary from the labiomental
   crease and inspect MP17 depth in profile. Also inspect the tentative 84/314 pair.
3. **Corners:** Check 61/291 as commissures. The narrow/open case reverses ordering
   with 185/409; it is specifically rendered under `stability-renders/`.
4. **Forehead:** Review MP10 and its neighboring arc as a canonical facial-patch
   boundary, not an inferred hairline. Current canonical/source shape disagreement
   is too large for the local search to resolve reliably.
5. **Temples/cheeks:** Inspect 21/251 and 234/454 in oblique/profile. Do not slide
   points to whichever surface is the camera silhouette.
6. **Jaw:** Inspect the mandibular curve, especially 172/397 and 132/361. Shape
   sensitivity is measurable; source curvature is not a unique landmark label.
7. **Chin:** Review 152 and paired chin points in profile; front-only overlays hide
   depth and lower-face occlusion. Do not approve from the warning count alone.

For 3D comparison, from the repository root:

```powershell
.\.venv-research\Scripts\python.exe scripts/run_annotation_proposals.py open --project runs/veyra3d/annotation-proposals-005
```

Press N ? Veyra. The existing region filters, Click-place/Snap, source IDs, explicit
point decisions and versioned export remain available. The additional **Proposal
evidence** panel shows uncertainty and switches the active marker between Original
and Proposed positions; switching clears its decision. Evidence values describe
the automated proposal, not later manual edits. All initial points are uncertain,
including 84/314. Use the seven cards to decide which regions need actual movement
before inspecting individual source-ID cards. Card edges include triangulation
diagonals and are not certified anatomical boundaries. Overlay chords do not
continuously follow the mesh surface; hidden points are drawn for inspection.

## Pin a few reviewed controls and refit the curves

An optional `--controls` accepts an existing versioned annotation export containing
named-human **accepted** positions. It checks original-map/asset/topology hashes and
valid barycentrics, allows only the 56 editable indices, and hard-pins those exact
surface positions in every hypothesis. No positions are inferred from a 2D drawing.
The source annotation file is preserved. New proposals remain uncertain; this
command does not create or transfer map approval. No real human controls were
supplied in this pass.

A useful starting control set is **0,17,61,291,10,152,234,454** (eight points), with
21/251 or mandibular points added where the region cards show ambiguity. This is
an efficient review order, not proof that eight controls settle all anatomy.

```powershell
.\.venv-research\Scripts\python.exe scripts/run_annotation_proposals.py propose --project runs/veyra3d/annotation-project-001 --controls path/to/reviewed-annotations.json --output runs/veyra3d/annotation-proposals-reviewed-001 --render-checks
```

A fresh output is required. If pins make an ordered curve infeasible, fitting
fails instead of silently relaxing constraints. Other points are proposed by the
same surface search/regularization; they are not automatically approved. Inspect
regional overlays and dense topology again. All prior candidates remain available.

## Reproduce the current no-control proposal

```powershell
.\.venv-research\Scripts\python.exe scripts/run_annotation_proposals.py propose --project runs/veyra3d/annotation-project-001 --output runs/veyra3d/annotation-proposals-next --render-checks
```

Optional `--prototype`, `--reference`, `--audit`, and `--blender` select local inputs.
The result includes `mapping-proposals.json`, `mapping-supported-only-proposals.json`,
`point-evidence.json`, `proposal-report.json`, comparisons, seven region cards,
56 numbered source-vertex/edge cards, and nine rendered deformation comparisons.
The Blender project is a separate copy; nothing is applied to the original map.
The supported-only file is another **unaccepted proposal**, not a training map.

Validation: 32 relevant unit tests passed (7 new proposal math, 13 correspondence,
6 annotation, 3 audit, 3 hygiene). Blender integration verified 56 uncertain
surface markers, evidence-panel registration, original/proposed switching, and
zero acceptance without saving changes. An independent file check verified exact
468 order, every source ID/barycentric support, 412 unchanged slots, original byte
hash, 20 cases and nine renders. Mouse interaction still needs human inspection.

Licenses: only installed MakeHuman CC0 core geometry/morphs and existing procedural
assets were used. MediaPipe reference geometry/connectivity remains Apache 2.0
with the existing pinned LICENSE/provenance under `third_party/mediapipe-reference`.
Blender-side scripts carry GPL-3.0-or-later notices; the full license text is already
included under `third_party/`. No new assets were downloaded. No training data,
real training, commit, push or tag was created. No files were staged by this pass.

# Strict correspondence study and native-topology alternative — 2026-11-16

Result: **0 reliably supported anatomical corrections; 56 disputed points remain
unresolved and unchanged.** No landmark movement in Blender is required to run
this workflow. It is an automated investigation/rejection pipeline, not an
anatomical approval system.

The original operational candidate baseline is
`runs/veyra3d/annotation-project-001/base-map.json`, SHA256
`92bc26a6b7ece346e4b96990097422861946f9321a31a1de7183ab1d9548a3dd`.
It originates from the `correspondence-anatomical-final` workflow and is associated
with source checkpoint `516252ea818e7e67345e8316991934524ffea1a9`. There has never
been an anatomically accepted 468-point mapping. The failed 56-point proposal is
an explicitly separate comparator, never the reference used to accept changes.

The new result is `runs/veyra3d/strict-correspondence-002/`.
`mapping-retained.json` is a byte-identical copy of the original baseline. All
468 slots retain their ordering, support triangles, source IDs and barycentrics;
all 56 disputed points have separate unresolved evidence records. The previous
scripts, reports and images remain intact. New implementation files are separate.

## Semantic information actually available

The installed MakeHuman/MPFB data was inspected before generating new trials:

| Source | Useful information | What it does not establish |
|---|---|---|
| Frozen hm08 mesh and source quad graph | Stable vertex IDs, actual connected surface edges | Exact MediaPipe anatomical site for a vertex |
| `basemesh_vertex_groups.json` / `hm08_config.json` | Body/helper ranges, joint cubes, symmetry, coarse dimensions | A skin jawline, vermilion segmentation, hairline or canonical oval |
| `_extra_vertex_groups.py` | A 418-vertex lips editing region | An external lip border; the region includes oral/perioral support |
| `lips_solid.json.gz` | Material/UV lip-region information | A verified vermilion annotation; the focused audit already demonstrated this |
| `rigify/rig.human.json` | 118 usable named facial endpoints supported by 62 unique body vertices | Equality with MediaPipe indices, or an external lip boundary |
| `rigify/weights.human_toes.json` | Explicitly CC0 soft deformation weights for lips, chin, jaw, forehead and temples | A uniquely correct anatomical threshold |
| Bundled identity/expression targets | Sparse, named source-vertex displacement fields | Annotation of the vertices as particular MediaPipe landmarks |
| Pinned MediaPipe canonical mesh | Ordered template and regional connectivity | The same identity/proportions or a certified MakeHuman correspondence |

Rig `VERTEX` endpoints refer directly to body vertices. `MEAN` endpoints often
average separated skin supports and need not lie on the skin. The lip means span
up to about 0.011785 source units. For example, upper central lip references use
source 509/571, lower central references 495/557, and the shared lip-chain
junctions use 402/435 or 7162/7195. These are useful regional constraints, but they
are not author-labeled external vermilion or exact surface commissure landmarks.
The code uses source IDs, **not stale rig default_position coordinates**.

The official MPFB license explicitly lists rigs and JSON mesh information among
its CC0 graphical assets, alongside the mesh and targets. Its Python program
logic is GPL. The socket-group Python file was inspected with AST literal parsing,
not executed or copied as program logic. Only installed core graphical data and
our existing procedural scene were used. No user-contributed assets were loaded.
See [MPFB's asset/source distinction](https://raw.githubusercontent.com/makehumancommunity/mpfb2/master/LICENSE.md)
and the [official license overview](https://static.makehumancommunity.org/about/license.html).
The pinned Apache-2.0 MediaPipe notices remain unchanged.

Other publicly visible correspondence efforts were screened without importing data:

- [HumanForge's MIT code license](https://raw.githubusercontent.com/abdulrehman043/humanforge/main/LICENSE)
  and [generated face map](https://raw.githubusercontent.com/abdulrehman043/humanforge/main/src/human/faceMap.ts):
  the map is generated from canonical/template geometry and includes repeated
  source indices and detector-derived reference positions. It is not independent
  anatomical evidence for our disputed borders. No map or portrait-derived data
  was adopted, and its confidence weights were not treated as calibrated truth.
- [FaceFit's calibration source](https://raw.githubusercontent.com/VGCLUB/facefit-mpfb/master/scripts/calibrate_correspondences.py)
  derives its map by detecting a rendered face and raycasting pixels. Its
  [code license is GPL-3.0](https://raw.githubusercontent.com/VGCLUB/facefit-mpfb/master/LICENSE).
  Reprojection agreement is useful, but detector-derived sites are not independent
  measured/validated 3D anatomical correspondence. No code or map was imported.
- [FaReT](https://github.com/fsotoc/FaReT) offers MakeHuman-based face stimulus
  generation. The reviewed project description did not provide a verified exact
  hm08-to-MediaPipe vermilion/oval annotation to substitute here.

This is a bounded source investigation, not a claim that no suitable map could
exist anywhere. No verified exact anatomical map was found in the inspected data.

## Improved automated method

New files preserve the older algorithm and results:

- `makehuman_semantics.py`: safe data inventory, core target loader, source hashes.
- `correspondence_gate.py`: fail-closed structural and surface-support checks,
  shortest paths restricted to real source quad edges, separate semantic decision.
- `blender_strict_correspondence.py`: baseline comparison, regional hypotheses,
  40-case evaluation, per-point evidence and native-topology feasibility study.
- `evaluate_lip_weight_interfaces.py`: competing skin-weight interface tests.
- `evaluate_native_face_sites.py`: evaluation-only native rig-support specification.
- `blender_render_strict.py`, `visualize_strict_correspondence.py`: fresh comparisons.
- `run_strict_correspondence.py`: portable launcher; fresh outputs required.

The regional proposals use author-defined lip center/corner references, forehead
controls, mandibular/chin references and temporal chains. Each curve follows
shortest paths on the actual source quad graph; samples use actual source edge
supports and barycentrics. No neighboring ring is declared anatomically correct,
no 2D projection offset is applied, and no canonical width/height ratio is forced.
Canonical ordering only supplies curve parameter positions.

The complete mandibular waypoint sequence revisited mesh vertices and was
rejected at generation time. A separately named endpoint-only jaw hypothesis was
then evaluated, not silently substituted as though it satisfied the stronger
constraint. The mandibular arc covers 58–288; transition points 132/361 remain
without a new geodesic proposal and are retained/flagged. All 56 still receive
baseline, prior-proposal and semantic evidence records. Single-point trials avoid
rejecting local geometry merely because another point in its region failed.

There are **63 evaluated regional/single-point configurations**, plus the rejected
waypoint-generation attempt. Three additional skin-weight hypotheses use upper
and lower lip bone-influence thresholds 0.5, 0.75 and 0.9. Their interfaces are
computed continuously inside source triangles, not projected image curves. Each
supplies 18 external lip proposals; commissures are deliberately left unchanged.
All three fail the complete structural gate. No threshold is promoted to an
anatomical border, and no threshold was tuned until warning counts disappeared.

Hard tests operate on all 468 landmarks and Veyra's 852 skin triangles, comparing
with the original baseline in the **same identity/expression pose**. Independent
absolute-deformation and fixed focused-audit-reference checks are also retained.
They reject nonfinite values, invalid support, nonpositive reference-normal dot
products, near-zero area (<=1e-12), area compression below 5%, edge ratios outside
0.25–4, landmark spacing below 0.0002 source units, curve crossings and reversed
mouth ordering. Source-support triangles are checked separately. The stricter
area/collision gates are explicit additions, not hidden changes to old metrics.

Normal reversals here mean reversals against the declared geometric reference;
they are conservative warnings, not a general proof of 3D self-intersection or
anatomical invalidity. Passing them cannot confer anatomical approval.

A candidate must also have exact semantic evidence. Region names, favorable
curvature, source skinning weights and source attachment do not meet that standard.
The previously supported 84/314 pair therefore remains a geometric hypothesis,
not a reliably established anatomical correction. Its lack of new regressions
is explicitly distinguished from full-domain structural safety and semantic proof.

## Tests and remaining failures

Eight identities (base, round, narrow, chin-height, temple, wide, tall, lower-lip
volume) × five states (neutral, mouth-open, corner-pull, eye-close, combined
open/pull) give **40 configurations**. They include all previous 20 and additional
stress cases. Identity amplitudes are 0.5 except lip volume 0.4; individual
expression units use 0.35; the combined state uses 0.25 open + 0.25 pull. These
are raw core deltas, not a certified full facial rig.

- The original 20-case baseline remains free of the checked failures.
- The unchanged baseline passes **38/40** expanded cases. `narrow/open_pull` and
  `tall/open_pull` reverse skin triangles **168 = [61,76,184]** and
  **807 = [306,291,408]**. In the tall case both also compress below the new 5%
  area threshold. These are inherited domain failures, not newly applied changes.
- The retained result introduces **zero new failures**, but does not repair those
  two cases. No retained mouth-order failures, severe edge ratios or collisions
  were found. No claim is made that the visible baseline anatomy is correct.
- The previous full proposal is replayed against the true baseline. Its neutral
  comparison still reproduces 23 new normal reversals and six extreme edge ratios.
  Against the separate fixed audit reference it reports 25 reversals/eight extreme
  edges; these reference scopes must not be mixed.
- Rig-constrained curves also fail: e.g. the combined trial has 59 fixed-reference
  normal warnings, three extreme edges and four near-collision pairs. It is
  rejected, not allowed to replace the baseline because it looks more like a lip.
- The 84/314 pair and several isolated rig trials introduce no new posed-baseline
  regressions. They still do not establish canonical anatomical equivalence, and
  the expanded absolute-deformation domain remains unsafe.
- Camera checks: **120** baseline pose/view comparisons plus **189** neutral
  candidate/view comparisons; maximum independent Blender/analytical disagreement
  is below **0.000495 px**. Nine actual diagnostic renders include the two failing
  baseline stress cases. Camera agreement is not anatomical agreement.
- **51 relevant unit tests passed**: 19 new guard/interface tests, 7 earlier
  proposal tests, 13 correspondence tests, 6 annotation tests, 3 audit tests and
  3 hygiene tests. No real training or optimizer work was run for this study.

The six historical canonical mouth-cap normal warnings remain in the original
focused audit, separately scoped from Veyra skin. They were not removed or
relocated. No landmark was shifted just to eliminate a warning.

## Is a simpler mapping more reliable?

A native source topology removes the cross-topology semantic inference problem.
The evaluation freezes a neutral, outward-facing head ROI, excludes the oral
cavity, and retains **3,198 native vertices / 5,818 actual source triangles**.
Selection is fixed before morph/view testing; no failed faces are removed later.
The ROI is a geometric window including parts of ears/upper neck, not an
anatomically certified face oval or hairline. There is no Delaunay replacement.
The [official basemesh documentation](https://static.makehumancommunity.org/mpfb/docs/assets/basemesh_and_helpers.html)
confirms that source topology/vertex identities persist when shape changes.

The native patch passes the conservative screen in **24/40** requests: all eight
neutral, corner-pull and combined open/pull cases. It passes **both** combinations
that break the 468-point baseline. The 16 separate mouth-open/eye-close requests
flag two small mouth-edge distortions or eyelid normal/edge warnings in the actual
source geometry. They are explicitly excluded from the screened configuration
list; changing landmark mapping cannot repair the generator's source deformation.
Those flags call for configuration rejection or proper rig/corrective-shape work,
not moving labels. Neither native nor MediaPipe screen alone proves anatomy.

A smaller evaluation specifies **62 unique native rig-support vertices**. All are
finite and collision-free across these 40 requests. They are not 62 standard
anatomical landmarks, and no artificial sparse face triangulation is supplied.
This demonstrates a simpler indexing contract, not a complete face reconstruction
system. The full patch or a generator-parameter decoder can provide dense geometry.

The native 24/40 and coarse-map 38/40 pass rates are not accuracy scores to compare:
they test different resolution/connectivity. The native advantage is certainty of
source identity, not demonstrated superior anatomical or real-face performance.

**Disposition:** automatic, anatomically valid 468-point correspondence is not
currently justified by these data. Native source geometry is a viable automation
path, with invalid generator requests screened and a new explicit target contract.
It does not demonstrate real-face accuracy or FaceUnlock readiness.

**Recommended next technical step:** build a separate native-geometry or
morph-coefficient target adapter with automatic source-geometry validation and
known camera/crop transforms. Keep the current 468-point model/checkpoint intact;
its decoder, loader and topology losses are explicitly 468-specific and cannot
silently consume native IDs. Use the screened core configurations for a future
small diagnostic pilot, and improve raw expression composition with proper
rig/corrective deformation before expanding it. This route needs no manual
landmark dragging. No dataset or training should begin until that new contract
and diagnostic adapter have been checked.

## Artifacts and reproduction

In `runs/veyra3d/strict-correspondence-002/`:

- `summary.md`, `report.json`, `point-evidence.json`: outcomes, source supports,
  failure cases, generation failures and per-point retention reasons.
- `semantic-inventory.json`, `regional-paths.json`, `tested-candidates.json`:
  author references, hashes, actual quad-edge paths and rejected hypotheses.
- `regional-sheet.png`, `regions/*.png`, `comparison-front/profile/oblique.png`:
  baseline versus rejected rig proposal versus unchanged retained result.
- `inspection-renders/*-comparison.png`: nine actual deformation/view comparisons.
- `weight-interface-evidence.json`, `weight-interfaces-*.png`: quantitative source
  skinning-field sensitivity and all three rejected surface interfaces.
- `native-topology-evaluation.json`, `native-topology-*.png`,
  `native-rig-support-evaluation.json`, `native-configuration-screen.json`:
  evaluation-only IDs/connectivity and configuration outcomes, not training labels.
- `mapping-retained.json`: exact original mapping bytes.
- `preserved-inputs.json`: 303 protected file hashes, including previous results
  and inspected workspace source. Outputs are fresh and prior runs are preserved.

Reproduce into a new directory from the repository root:

```powershell
.\.venv-research\Scripts\python.exe scripts/run_strict_correspondence.py --output runs/veyra3d/strict-correspondence-next --render-checks
```

Override `--project`, `--prototype`, `--proposal`, `--reference`, `--audit` and
`--blender` if needed. The run is bounded, offline, uses existing dependencies and
refuses existing result directories. No original source/model/checkpoint is
rewritten. The renderer/visualizer append diagnostic files to the fresh result;
existing review boards and native studies are not silently overwritten.

No assets were downloaded or installed, no anatomical approval or labels were
fabricated, no full dataset or AI training was started, and no staging, commit,
push or tag was performed. The GitHub checkpoint remains unchanged.

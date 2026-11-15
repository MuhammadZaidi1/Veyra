# First facial-geometry pilot (prepared 2026-11-15; not started)

Recommendation: a small, licensed FaceScape multi-view subset for an internal
noncommercial research pilot, conditional on access eligibility and a reviewed
468-point registration. This is not a distribution-cleared training source.
For a product or distributable weights, obtain explicit written permission covering
training, derived annotations, trained-weight redistribution and commercial use,
or use separately licensed/consented scans. Do not assume research permission
covers product development. No data or weights were downloaded for this plan.

## Dataset comparison and primary sources

- FaceScape: multi-view reconstructed measured surfaces, calibrated photographs,
  and registered models. A registered base mesh is an approximation of the scan,
  not a direct measurement at every vertex. Strongest technical fit for paired
  RGB/geometry and expression diversity. Standard access is academic/research
  restricted, noncommercial; no dataset redistribution. Public images/renders
  require the publishable-subject list. Special licensing must be negotiated.
  https://nju-3dv.github.io/projects/FaceScape/
  https://nju-3dv.github.io/projects/FaceScape/static/license/LicenseAgreement_FaceScape.pdf
- FaceVerse: 2,688 photogrammetric head scans, plus fitted common-topology models
  (fitted meshes are scan-derived, not independently measured vertices).
  Noncommercial research only; academic staff request; derived-data sharing
  needs permission and published portraits/models require blurred eyes. Less
  convenient for paired captured RGB than FaceScape's calibrated views.
  https://github.com/LizhenWangT/FaceVerse-Dataset
- Multiface: captured multi-view sequences with tracked/reconstructed geometry;
  not monocular pseudo-labels, but tracking/registration still introduces error.
  Only 13 identities, much temporal redundancy and substantial storage. Useful
  later for expression/temporal tests, less attractive as the first identity-
  generalization pilot. CC BY-NC 4.0: attribution and noncommercial restrictions;
  commercial deployment needs separate rights. Do not assume weights are cleared.
  https://github.com/facebookresearch/multiface
  https://arxiv.org/abs/2207.11243
- Headspace/LYHM: 3dMD captured head surfaces; the statistical LYHM is learned
  from scans, not the raw measurements. Useful shape reference, less suitable
  for this first expression-rich paired-RGB pilot. Academic application,
  noncommercial research/education; not a product-distribution license.
  https://www-users.york.ac.uk/~np7/research/LYHM/
  https://www-users.york.ac.uk/~np7/research/LYHM/LYHMagreement07.pdf
- 300W-LP / AFLW2000-3D: fitted 3D morphable-model annotations and, for 300W-LP,
  synthesized pose expansion. These are estimated/pseudo-3D, not measured depth.
  Convenient size does not solve topology or rights. Source-photo and morphable-
  model terms require separate review; an MIT implementation license does not
  license its training images/labels. Do not use for measured-geometry evaluation.
  https://arxiv.org/abs/1511.07212
  https://github.com/cleardusk/3DDFA_V2

None of these was verified as an unrestricted commercial/distributable solution.
Publicly requestable does not mean open for all uses. Keep licenses, versions,
approval emails and required paper citations with the private experiment records.
FaceScape's current project page and PDF give different contact addresses: follow
the current project request instructions and clarify special rights with the owner.

## Existing model/data contract

The existing loader reads JSONL records with exactly image, geometry, group:

    {"image":"crops/000001.png","geometry":"labels/000001.npz","group":"opaque-person-001"}

All paths stay under the configured local root. NPZ requires exactly `vertices`
(float32 [468,3]) and `valid` (bool [468], at least one true). Complete labels must
have zero centroid; supervised coordinates must be finite, magnitude <=10.
Indices must match Veyra's MediaPipe-derived 468-vertex connectivity, excluding
iris points. Arbitrary 468 mesh samples and 68 sparse landmarks are incompatible.
RGB crops have sides <=4096 and resolved EXIF orientation. Loader letterboxes to
128x128, normalizes RGB to [-1,1], and multiplies XYZ by width/max(width,height).
Labels entering the loader use original-crop-width units, +x right, +y up,
+z toward camera, in input-view pose. Outputs are centroid-centered relative
geometry, not calibrated metric depth or screen location. Missing vertices must
retain a consistent full-mesh origin; never center just the visible subset.

## Upstream registration and coordinate conversion required

1. Obtain the authorized paired RGB/mesh/camera files and their version metadata.
   Prefer near-frontal views initially. Reject unreliable camera/view flags.
   FaceScape requires image undistortion using supplied distortion coefficients:
   https://github.com/zhuhao-nju/facescape/blob/master/doc/doc_mview_model.md
2. Establish a reviewed mapping from the source mesh surface to the exact 468
   anatomical slots. For a common registered topology this can be a versioned
   triangle-index/barycentric-weight map applied to each mesh; raw scans need
   per-scan registration. Do not assume the same indices across raw meshes.
   Registration is annotation work: no such validated mapping is supplied here.
   Review lip boundaries, eyelids, nose, silhouette, winding and degenerate faces.
   A monocular predictor may propose correspondence for manual review but must
   not supply target depth or be reported as measured ground truth.
3. Transform sampled measured geometry into each view using the supplied camera
   extrinsics. CV-camera axes are x right, y down, z away; flip y and z. Verify
   with projected scan overlays on the undistorted image, not by visual guesswork.
4. Define one scale convention for this relative geometry target. Recommended
   first pilot: weak-perspective camera-space XYZ, with all axes multiplied by
   f/Z0 in pixels per scan unit, then divided by crop width in pixels; use the
   full registered face centroid for Z0 and centering. This assumes near-frontal,
   distant faces and approximately equal fx/fy; reject incompatible calibration.
   Record residuals between perspective projection and this approximation.
   Do not combine perspective x/y with metric z or normalize axes independently.
   Crop translation does not change centered geometry; crop width changes scale.
   If perspective residuals are unacceptable, revise the contract explicitly
   before training rather than silently changing label meaning.
5. Derive validity from scan coverage/registration quality, not predictor confidence.
   The pilot should prefer complete, high-quality faces. Do not fill absent scan
   geometry with guessed depths. Store masks for holes/occlusions as appropriate.
6. Save approved crops and converted NPZs locally; inspect overlays and depth-sign
   checks across subjects, expressions and views. Retain private provenance for
   every sample: source version, subject/session/view, calibration, mapping version,
   scale, crop rectangle and quality review. Crops/meshes remain identifying data.

The prepared importer is deliberately a final packaging/validation stage, not a
raw FaceScape parser or registration tool. Implementing a raw adapter before
seeing the authorized release and a verified correspondence map would guess
label semantics. This remains required work before real training can start.

## Import procedure (only after obtaining and reviewing labels)

Put approved crops/NPZs under a private source directory. Create `index.jsonl`
with image, geometry, group, split fields; split is train, val or test. Create
`review.json` with these fields:

    {
      "contract":"centered-crop-width-xyz-468-v1",
      "label_source":"scan-derived",
      "reviewed":true,
      "license_reference":"private approval identifier and terms version",
      "registration_reference":"reviewed mapping version and quality report",
      "coordinate_conversion_reference":"camera/crop/scale conversion version"
    }

This declaration records a human assertion; software cannot prove provenance,
licensing, correct anatomical correspondence or whether depth was measured.
The importer copies only supplied files, checks all labels/images and split
leakage, preserves review metadata, and refuses existing output paths. It does
not estimate, recenter, pad, interpolate or fabricate target arrays.

    .\.venv-research\Scripts\python -m scripts.import_registered_geometry --source data/staging/facescape-reviewed --output data/approved/facescape-pilot

## Frozen split and pilot size

Select 50 authorized, usable people BEFORE choosing samples. Shuffle sorted
opaque person IDs with seed 42 and freeze 35 train / 7 validation / 8 test people.
Keep every person's sessions, expressions, cameras, augmented variants and
near-duplicate frames in their assigned split. The importer consumes explicit
splits; it does not randomly split images or choose people for you. Retain the
original identity-to-opaque-key mapping privately with restricted access.

Target up to 20 expressions x 3 selected near-frontal views per person:
2,100 train / 420 validation / 480 test images, subject to availability and QC.
Missing examples reduce actual counts; do not invent them to meet the budget.
No augmentations in the first pilot. Check duplicates/near-duplicates and
expression/pose coverage manually; the loader catches groups and byte-identical
images, not all leakage. Fit registration/template choices on training people,
then freeze them before val/test. Validation selects best.pt; use test only after
hyperparameters are fixed. Report per-person and pose/expression coverage and
relative vertex error, including a separate depth error in later analysis.
Do not claim millimeter or in-the-wild accuracy from this normalized pilot.

## Proposed run and measured compute budget

Config: configs/veyra3d/facescape_pilot.example.json. 128x128, latent 128, batch 8,
float32, no AMP, strict determinism, AdamW lr 0.001, five epochs, no accumulation,
workers=0. Existing vertex/edge/normal losses stay 1 / 0.1 / 0.01.
Existing training writes atomic last.pt every epoch, best.pt when validation
improves, experiment.json and metrics.jsonl. RNG/optimizer/scaler are saved.
Resume is epoch-boundary only and rejects changed source/data/config fingerprints;
freeze preparation/code before starting. Failed epochs resume from last complete
checkpoint. Use a fresh run directory for a changed experiment.

Benchmark on 2026-11-15: Windows 11, GTX 1660 SUPER, PyTorch 2.9.1+cu126,
128x128, batch 8, float32, deterministic=true, 2 CPU threads. Five warmups plus
30 timed forward/objective/backward/clip/AdamW steps, GPU synchronized. Median
17.16 ms/step, mean throughput 428.17 samples/sec, peak PyTorch allocated
82.93 MiB / reserved 98 MiB. This excludes driver/context allocations and other
apps; it is not total VRAM. Reproduce with:

    .\.venv-research\Scripts\python -m scripts.benchmark_veyra3d_cuda

At 2,100 train examples: 263 steps/epoch, 1,315 over five epochs; compute-only
training estimate about 25 seconds using measured mean throughput. Validation,
PNG/JPEG decode, transfers, content hashing on every train/evaluate invocation,
checkpoint writes and startup are excluded. Provision 1-5 minutes for the first
pilot as an unverified planning allowance; slow disk/decoding can exceed that.
After approval, first run ONE epoch and use recorded samples_per_second and
wall time to refine the estimate before extending to five. Six GB should have
ample room at this measured batch, but verify actual data-loading memory and
peak VRAM. This pilot tests learning/data quality, not convergence.

Commands below are instructions only; no real run was executed:

    .\.venv-research\Scripts\python -m veyra3d train --config configs/veyra3d/facescape_pilot.example.json
    # To extend a completed five-epoch run, increase epochs (e.g. to 10), then:
    .\.venv-research\Scripts\python -m veyra3d train --config configs/veyra3d/facescape_pilot.example.json --resume runs/veyra3d/facescape-pilot-v1/last.pt
    .\.venv-research\Scripts\python -m veyra3d evaluate --config configs/veyra3d/facescape_pilot.example.json --checkpoint runs/veyra3d/facescape-pilot-v1/best.pt --split test

## Before starting

Obtain eligible dataset access (or negotiated special access if not an academic
researcher), written rights for the intended research/product/weight-release
scope, the permitted RGB/mesh/camera subset, and the reviewed 468-point mapping
and conversion/QC report. Confirm subject-disjoint splits and local storage
capacity before transfer; the full dataset is large, request only permitted
components/subsets where possible. Save immutable license/approval records.

Keep source faces, masks, labels, subject keys, calibration and checkpoints
restricted to approved users, outside public repositories/cloud sync unless the
agreement permits it. Existing Git ignores protect common data paths but are not
access control. Opaque IDs do not anonymize faces. Do not publish example crops,
renders or meshes without portrait permission. Treat weights as private until
release rights and memorization/privacy risks have been reviewed. Follow the
provider's retention/deletion obligations and applicable consent/privacy rules;
record any jurisdiction/institution-specific requirements before collection/use.

# Automated single-face Blender prototype

Tested locally with Windows, Blender 5.3.0 and MPFB 2.0.18. No downloads,
full dataset generation or AI training are performed by these scripts.

## Run from the project root in PowerShell

    .\.venv-research\Scripts\python.exe -m scripts.run_blender_prototype --check-only
    .\.venv-research\Scripts\python.exe -m scripts.run_blender_prototype --output runs/veyra3d/blender-prototype-next

The launcher finds Blender on PATH or under Program Files/Blender Foundation/
Blender 5.3. Override with --blender 'C:\path with spaces\blender.exe'. Arguments
are passed as a subprocess list, not a shell string. Existing output directories
are refused. The Blender subprocess has a five-minute timeout; partial artifacts
on failure are not success and should be inspected before selecting a fresh path.
A repeated run renders the same neutral face; there is no sample-count option.

Equivalent direct Blender call (does not create the Pillow overlay):

    & 'C:\Program Files\Blender Foundation\Blender 5.3\blender.exe' --background --python-exit-code 1 --python scripts/blender_face_prototype.py -- --output runs/veyra3d/blender-direct-next

After a direct call, run make_overlay(output) from scripts.run_blender_prototype
using the research Python environment for the independent raster check/overlay.
The report's initial status remains pending until that check passes.

## Installation checks

MPFB is already installed/enabled on this PC as bl_ext.blender_org.mpfb. No
installation is currently required. If missing elsewhere: install Blender,
open Preferences > Get Extensions, search MPFB, and install/enable the official
extension. This manual setup may require allowing extension-platform network
access. The scripts do not install packages, modify preferences or fetch assets.
If --check-only reports an ambiguous module, resolve duplicate MPFB installations
in Blender Preferences first. Missing core base.obj or its explicit CC0 header
causes a hard error; do not substitute an arbitrary downloaded model.

Official setup: https://static.makehumancommunity.org/mpfb/docs/getting_started.html

The launcher/overlay uses NumPy and Pillow already in .venv-research. Blender
uses its own bundled Python/NumPy, so do not run blender_face_prototype.py with
ordinary Python or try installing bpy into the research environment.

## Asset boundary and licensing

The only character asset read is the installed MPFB data/3dobjs/base.obj and its
bundled vertex-group metadata. Its OBJ header explicitly declares CC0. No user
asset library, skin photograph, morph pack, hair, HDRI or community model is
loaded. Eye spheres, iris spheres, clay shader, lights and world are created by
our script. This is a geometry/projection prototype, not a photorealistic face.
Blender may initialize enabled extensions and MPFB may write its normal logs;
the script never saves preferences or opens an existing scene. It deletes only
objects in its disposable subprocess scene.

Official asset/output license:
https://github.com/makehumancommunity/mpfb2/blob/master/LICENSE.md
https://www.makehumancommunity.org/about/license.html

MPFB program code is GPL; bundled assets/output are CC0. The Blender automation
script is marked GPL-3.0-or-later. This does not impose GPL on rendered output or
trained weights. Record/check separate terms before adding any non-core assets.
The report includes base and metadata SHA256, the extension manifest/version and
license source so the prototype can be traced to the exact local asset version.

## Artifacts (ignored runs directory)

- face.png: one neutral head render, 512x512, Cycles CPU with 16 samples. CPU is
  intentional for a small reference render, not a GTX performance benchmark.
- overlay.png: numbered diagnostic surface points over the RGB render, with an
  explicit NOT 468 LANDMARKS / NOT TRAINING READY banner.
- coordinate-probes.png: isolated cyan emissive spheres rendered at those same
  positions through the same camera for a raster-space consistency check.
- inspection.blend: inspectable single scene, with the face mesh retaining an
  integer point attribute makehuman_source_vertex_id.
- surface-review.npz: evaluated world_xyz, source_vertex_ids and triangles. This
  is a review mesh, NOT a Veyra vertices/valid training NPZ.
- mapping-draft.json: 468 numbered slots, all triangle/weight entries null,
  status=unmapped, reviewer=null, asset and topology fingerprints.
- report.json: environment/provenance, camera transform, diagnostic positions,
  projection/raster errors, training_ready=false.

No manifest, trainable target NPZ, or automatically accepted mapping is produced.
No MediaPipe weights or canonical model are downloaded or used by this prototype.

## What the coordinate checks establish

The square orthographic view has world-space width W. Transform each evaluated
point into Blender camera coordinates (x right, y up, local z toward camera).
Diagnostic centered coordinates are (p - diagnostic_centroid)/W. Retain that
centroid to undo centering before comparing image pixels: the model output's
centroid-zero coordinates alone do not encode screen position.

For 512 pixels, u=(x/W+0.5)*512-0.5 and
v=(0.5-y/W)*512-0.5, with image y pointing down and integer pixel centers.
The script compares this independently to Blender world_to_camera_view; maximum
axis error must be below 0.001 pixel. It ray-tests candidate surface vertices
against the actual mesh to exclude hidden/occluded diagnostic locations. The
Pillow check independently detects cyan pixels in the marker render and requires
each predicted location to be within 2 pixels of cyan. A deliberately shifted
render fails the regression test. These tolerances reflect finite raster sampling.

The 12 points are chosen only to test projection. Their IDs are MPFB source
vertex IDs, NOT Veyra landmark indices. Projection checks establish consistent
axes, scale and image origin; they do not establish anatomical correspondence,
face realism, physical-depth accuracy or future pose/expression correctness.

## Preparing the 468-point correspondence (manual review required)

Veyra retains the 468 ordering/connectivity in veyra/topology_data.py. MPFB's
hm08 mesh has different connectivity. There is no validated mapping in this repo
or this prototype. A count of 468 alone cannot establish semantic compatibility.

1. Pin the MPFB base/version and review exported surface-review.npz/inspection.blend.
   Use source_vertex_ids plus triangle indices; helper/joint cube vertices are
   excluded from the rendered surface. Existing MPFB joints are useful context,
   but their centers are not automatically anatomical skin landmarks.
2. Establish correspondence against Veyra's documented ordering, optionally using
   the official MediaPipe canonical mesh as an anatomical reference only after
   separately obtaining it and retaining its Apache license/notices. Do not use
   a monocular predictor's depth as synthetic ground truth.
   Reference: https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/modules/face_geometry/data
3. For each slot, record source_triangle (index into the exported triangle array)
   and barycentric [w0,w1,w2], finite, nonnegative, sum=1. Anchor position is
   w0*p0 + w1*p1 + w2*p2. Check eyelid/lip boundaries and symmetry carefully;
   nearest-surface fitting alone can put upper/lower lips or lids on wrong sides.
4. Review all slots using numbered overlays from front, left, right and above;
   preserve Veyra's anatomical side conventions. Decide/document fixed anatomical
   contours versus apparent view-dependent silhouette semantics.
5. Verify on several identity shapes, blinks, open mouths and head rotations.
   This prototype does not implement those variations yet. Avoid remeshing or
   topology-changing modifiers; if enabled later, update anchors onto the exact
   render surface and prove index stability.
6. Record a reviewer and mapping version, residual/quality checks, asset/topology
   hashes and review images. Only after explicit anatomical review should a
   separate exporter write centered float32 [468,3] and bool validity arrays.

An edited status field does not prove a mapping is correct. These scripts do not
consume arbitrary draft mappings or export training labels. Implementing and
validating that exporter is later work, after the correspondence review.

## Current evidence and next approval

Corrected inspection output: runs/veyra3d/blender-prototype-002. The initial
prototype-001 is retained as an overexposed lighting trial of the same face.
The corrected render was visually inspected; its analytical-vs-Blender projection
error is 0.0000133 pixel and maximum distance to rendered cyan pixels is 0.570
pixel. The neutral face is intentionally low-sample and simplified, with no skin
textures/hair. This is not anatomical validation or a training-quality approval.

No installation is needed here. You need to review the render/framing and,
separately, approve a completed anatomical mapping after the work above. There is
nothing valid to approve in the empty mapping today. Expanding to multiple faces,
expressions, glasses or any full dataset is outside this single-face prototype.

# Canonical 468 rendering control

This is an executable $0 coordinate/training control using one public synthetic
template. It supplies exact **template vertex identity**, not independently
verified human anatomy. No MakeHuman file or prior experiment was modified.
All 468 original MakeHuman candidates remain anatomically unvalidated and are
excluded. The model, its 468 outputs, and its existing topology are unchanged.

## Source and licensing

The existing `third_party/mediapipe-reference/canonical_face_model.obj` is pinned
to revision `02ca022d573c004c463d8c61eb3f8ce17e95a7fc`, SHA256
`8bac80443397e113f41a8b565ea72c59390bc031d9defab289dba7bc0c54e618`.
The renderer verifies the local provenance and license hashes. No downloads,
photographs, pretrained detectors, textures, MakeHuman assets, or MPFB are used.

MediaPipe's [upstream license](https://github.com/google-ai-edge/mediapipe/blob/02ca022d573c004c463d8c61eb3f8ce17e95a7fc/LICENSE)
and the checked local copy are Apache-2.0. Its copyright grant permits use,
derivatives and redistribution; its terms include no research-only or
noncommercial restriction. AI training/weight ownership is not separately
specified. This provides a permissive basis for the reference control, without
claiming a special upstream endorsement or a blanket license for other assets.
Preserve the license, relevant upstream notices/provenance, and modification
descriptions when distributing derived artifacts. Conservatively carry those
materials alongside any future weights derived from this control. See
[Apache sections 2–4](https://www.apache.org/licenses/LICENSE-2.0).

All clay materials, area lights, background and camera poses are procedural
project code. [Blender's license statement](https://www.blender.org/about/license/)
allows commercial use of artwork produced with Blender; rights in the imported
canonical reference still apply. The Blender script is GPL-3.0-or-later, as
marked in its header. Output includes the original OBJ, LICENSE, provenance,
reference README and a new `SOURCE_NOTICE.txt` describing transformations.

## Geometry contract

OBJ vertex lines retain their exact 0–467 ordering. No welding, remeshing,
correspondence fitting or anatomical adjustment occurs. The original OBJ is
copied byte-for-byte. Its original **898 triangles** are rendered. Veyra's
existing **852 triangles** are a different connectivity list and remain
unchanged. The first control uses vertex-only loss (edge/normal weights zero),
so these surface lists are not assumed interchangeable. Existing objective code
still computes auxiliary diagnostics, but they do not contribute to this loss.

Canonical coordinates are in cm, x right/y up/z forward, as described in
[MediaPipe's canonical model documentation](https://chuoling.github.io/mediapipe/solutions/face_mesh.html#canonical-face-model).
The explicit source-to-world transform converts to meters, rotates to Blender
axes `(x,-z,y)`, and places the known 468-vertex centroid at the origin. Only the
camera moves. No mesh deformation is implemented; the loader rejects one marked
as deformed. A future deformation experiment needs a separate schema and must be
described as a synthetic approximation, not verified human anatomy.

For homogeneous source vertices `v`, source-to-world matrix `S`, and Blender
camera-to-world matrix `C`, absolute camera coordinates are `q = inverse(C) S v`.
Veyra targets are `(q - mean(q)) / 0.23`: x right, y up, z toward camera, all axes
in orthographic image-width units. Absolute visible z is negative before
centering. Labels are float32; source and absolute camera coordinates are
retained at float64 precision. Metric scale, absolute position and depth are
deliberately removed by Veyra's existing contract.

For square resolution `R`, PNG pixel-center coordinates are
`((q.x / .23 + .5)*R - .5, (.5 - q.y / .23)*R - .5)`.
Overlays reconstruct `q` from the actual float32 targets and the saved centroid.
No visibility mask is inferred from anatomy: all vertices of this known template
are supervised, including any that are self-hidden. Visibility is not anatomical
evidence.

NPZ arrays contain `vertices[468,3]`, `valid[468]`, exact `vertex_ids[468]`,
`camera_vertices[468,3]`, and `pixel_xy[468,2]`. Validation recomputes all 468
labels/projections, checks source/order, finite values, centering, frustum,
camera transforms, content hashes, local paths, duplicate images and conditions.
Blender's projection API independently checks every vertex. Twelve cyan vertex
probes in a separate 512px raster check the PNG axis/origin convention.
The overlays are inspection products, never training inputs.

## Split and reproducibility

| Split | Camera yaw | Key-light energies | Images |
| --- | --- | --- | ---: |
| train | -12, 0, +12 degrees | 3, 5 | 6 |
| validation | -6 degrees | 3, 5 | 2 |
| test | +6 degrees | 3, 5 | 2 |

Camera yaw groups are disjoint, so both lighting variants of a held-out pose
stay outside training. Lighting levels are shared intentionally. **One identical
template spans all splits.** This is condition interpolation, not independent
identities or new facial shapes. The production manifest loader's identity
leakage guards were not weakened; this explicitly marked control has its own
loader. Validation selects no anatomy, and the test split was not evaluated.

Factory-startup Blender 5.3, CPU Cycles, 8 samples, no denoising, 128px RGB and
seed 42 were used. Blender version and renderer hash are saved in `control.json`.
Source/renderer/RNG pins make the scene reproducible; bitwise image equality
across different Blender versions or platforms is not claimed. Version the code
and keep the generated provenance when reproducing elsewhere.

From the repository root in PowerShell, use a **fresh** render destination:

```powershell
.\.venv-research\Scripts\python.exe -m scripts.canonical_control render --output runs/veyra3d/canonical-control-002
.\.venv-research\Scripts\python.exe -m scripts.canonical_control check --output runs/veyra3d/canonical-control-002
$env:VEYRA_CANONICAL_CONTROL = 'runs/veyra3d/canonical-control-002'
.\.venv-research\Scripts\python.exe -m unittest discover -s tests_research -p test_canonical.py -v
```

Blender is located from PATH or the normal Windows installation directory;
`--blender <path-to-executable>` overrides discovery. Subprocess arguments use a
list, including paths with spaces. All reference paths derive from the repo
location. No manual Blender operation, installation or paid service was needed
on this PC. NumPy/Pillow/PyTorch are already in the research venv.

## Bounded CUDA experiment and measured result

`configs/veyra3d/canonical.json` is a dedicated wrapper configuration, not an input
to the general Veyra CLI. It specifies 64px inputs, latent dimension 64, batch 3,
float32, AdamW at 0.001, seed 42, deterministic algorithms enabled, and two epochs
(four optimizer steps). No `warn_only` or global determinism relaxation is used.
The runner refuses more than two epochs. Epoch checkpoints are atomically saved
with model/optimizer/scaler/RNG, configuration, data and code signatures, and
`synthetic_only: true`. Resume rejects changed data/code/configuration.

An optional later execution that retains its research checkpoint is:

```powershell
.\.venv-research\Scripts\python.exe -m scripts.canonical_training --data runs/veyra3d/canonical-control-002 --output runs/veyra3d/canonical-training-002 --epochs 1
.\.venv-research\Scripts\python.exe -m scripts.canonical_training --data runs/veyra3d/canonical-control-002 --output runs/veyra3d/canonical-training-002 --epochs 2 --resume runs/veyra3d/canonical-training-002/last.pt
```

Those optional commands were not launched. The minimal integration test did run
eight CUDA optimizer steps total: an uninterrupted two-epoch run, then one epoch
plus resumed second epoch. Model, optimizer, scaler, RNG, epoch/global step and
best validation loss matched **bit-for-bit** on the GTX 1660 SUPER with PyTorch
2.9.1+cu126. Temporary verification checkpoints were removed; small metric
reports were retained. This verifies epoch-boundary resume on this environment,
not bitwise determinism across different devices or library releases.

Results in `runs/veyra3d/canonical-control-001`:

| Check | Measured result |
| --- | ---: |
| Focused unit/integration tests | 11 passed |
| Known template vertices per image | 468 |
| Float32 target maximum absolute rounding error | 1.49e-8 image widths |
| Blender projection maximum disagreement | 8.27e-6 px |
| Independent raster maximum disagreement | 0.5361 px |
| Synthetic validation XYZ error, epoch 1 | 0.23719 image widths |
| Synthetic validation XYZ error, epoch 2 | 0.22662 image widths |
| Constant mean-training-template validation XYZ error | 0.01471 image widths |
| Peak CUDA tensor allocation | 68.4 MiB |

XYZ error is mean Euclidean per-vertex distance, not mm or biometric accuracy.
The constant baseline averages the six training target arrays and repeats that
same array for both validation images; it ignores RGB. It is reproducible as
`mean(norm(val_targets - mean(train_targets, axis=0), axis=-1))`. Its lower error
shows why a falling loss from four SGD steps is not a reconstruction result.

The first training epoch took 0.625s and the second 0.058s; validation took 0.021s
and 0.013s. The complete focused tests took 3.6s excluding Python startup; the
render/check/overlay command took about 4.2s. A fresh two-epoch execution should
therefore take seconds on this PC. These tiny warmup-dominated timings and
PyTorch tensor allocation are not a large-dataset throughput estimate or total
GPU memory usage. No full dataset or prolonged training was launched.

## What this establishes and the next step

The complete path from lawful pinned geometry to rendered RGB, 468 labels,
preprocessing, deterministic backward and checkpoint/resume is executable.
The targets are exact for the defined generated mesh and camera. It can measure
template/pose recovery under these artificial rendering conditions.

It does **not** validate MakeHuman correspondence, human anatomy, shape recovery
across people, real camera perspective/metric depth, expressions, skin/hair,
glasses or lenses, webcam performance, identity recognition, liveness or unlock
security. There are no identity classes, identity embeddings, real scans or
real-world evaluation here. A predictor can succeed by learning one template
plus pose, without learning human geometry.

Next, run a separately authorized, bounded learning-curve control and compare it
with the constant-template baseline before expanding scope. If that succeeds,
add a small versioned experiment with explicitly synthetic procedural shape
parameters and exact topology-preserving labels, then procedural frame
occlusion. Keep those results separate from any claim of anatomical/identity
validity. Real-person performance will still require lawful independent data and
evaluation. No label is promoted from the unvalidated MakeHuman mapping.

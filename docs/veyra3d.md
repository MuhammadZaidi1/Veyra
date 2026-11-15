# Veyra3D v0.3 research foundation

Veyra3D is an independent, randomly initialized PyTorch encoder-decoder for future single-image facial geometry research. **There is no validated custom facial reconstruction model yet.** The current smoke checkpoints learn only invented non-face patterns and grids. A falling synthetic loss, successful backend probe, or a plausible mesh does not establish facial accuracy, measured depth, identity recognition or authentication.

The working MediaPipe/OpenCV/Qt v0.2 prototype remains unchanged in `veyra/`. It is the initial comparison baseline, not the final custom network. Veyra3D is not connected to the webcam viewer or FaceUnlock. No facial dataset was downloaded and no webcam images were collected for this milestone. Only short synthetic diagnostics/tests were run; full model training has not started.

## Environment and exact commands on this Mac

The optional `.venv-research` is separate from the working `.venv`. Research pins PyTorch 2.9.1, NumPy 1.26.4 and Pillow 13.0.0; it does not install torchvision, fetch pretrained weights, or change the baseline dependencies.

```bash
cd /Users/muhammadabbas/Documents/Veyra

# Already installed here; use this on a fresh research setup.
./setup-research.sh

# Architecture, numerical CPU comparison and one backward/optimizer probe.
./research.sh doctor --device cpu
./research.sh doctor --device mps

# Each smoke command performs only four optimizer steps on invented data.
# It saves, reloads, resumes and evaluates temporary checkpoints, then deletes them.
./research.sh smoke --device cpu
./research.sh smoke --device mps

# Research unit tests: generated fixtures only, no camera or facial datasets.
.venv-research/bin/python -m unittest discover -s tests_research -v

# Existing v0.2 regressions, including native rendering and cleanup.
VEYRA_NATIVE_UI_TEST=1 .venv/bin/python -m unittest discover -s tests -v

# The working webcam prototype still launches exactly as before.
./launch.sh
```

`research.sh` applies the same macOS process network-denial policy as the baseline launcher and disables MPS-to-CPU operation fallback. Setup has separate network access to fetch Python packages. Research code contains no network, dataset download, camera acquisition, recording or geometry-export path. CUDA is not available on this Mac; selecting an unavailable backend fails with an explanation instead of silently switching devices.

CPU is the default. MPS forward/backward, optimizer update, CPU numerical comparison, checkpoint roundtrip, resume and evaluation have been exercised on this M1 Pro. A small batch-one probe was faster on CPU; MPS availability is not a speed guarantee.

## Model architecture

Default model: **258,644 trainable parameters**, approximately **0.99 MiB of float32 parameter storage**. Optimizer state, activations, gradients, framework memory and device workspaces are additional. The smaller smoke model has 131,796 parameters.

Input is `float32 [B, 3, 128, 128]`, RGB normalized from byte range to `[-1, 1]`. The configuration also supports square sizes 64, 192 and 256.

1. A 3×3 stride-2 stem maps RGB to 16 channels.
2. Four depthwise-separable stride-2 blocks produce 24, 32, 64 and 96 channels. Each combines a 3×3 depthwise convolution with a 1×1 projection, GroupNorm and SiLU.
3. Adaptive 2×2 spatial pooling retains coarse spatial layout, producing 384 features.
4. The point decoder maps `384 → 128 → 128 → 1404` with SiLU hidden activations.
5. The output reshapes to `[B, 468, 3]` and subtracts its per-mesh centroid.

GroupNorm supports small batches without relying on running BatchNorm statistics. The implementation uses standard eager PyTorch operations without custom CUDA kernels, compilation, an external backbone or pretrained weights. Model size is a design starting point, not evidence of adequate reconstruction capacity.

The output ordering matches the existing 468-vertex surface topology. Its 852 triangles are reused **as index connectivity only** from `veyra/topology_data.py`, under the existing MediaPipe license. Neither MediaPipe network weights nor its predictions are used to initialize/train Veyra3D. Iris vertices 468–477 are outside this first custom output contract.

There is no confidence/presence head, crop detector, pose head, texture decoder, full-head completion, camera calibration or physical depth recovery. The existing live system retains those responsibilities it already implements; the custom model cannot replace it yet.

## Coordinate and preprocessing contract

Contract identifier: `centered-crop-width-xyz-468-v1`.

- Coordinates are centered, relative units. +x points right, +y up, +z toward the camera. The pose is aligned to the input image; the model does not return a canonical pose automatically.
- Local labels must be prepared in **original face-crop-width units**, with a consistent origin established before missing vertices are masked. Complete 468-vertex targets must have zero centroid.
- Non-square crops are letterboxed into the square network canvas without stretching. All XYZ labels multiply by `original_width / max(original_width, original_height)` so output units become square-canvas-width units. Integer image-resize rounding can introduce subpixel quantization.
- Padding uses neutral RGB 128. The loader rejects unresolved EXIF rotation rather than rotating pixels while leaving labels misaligned. Crop extraction/registration must happen in an approved upstream preparation process; this module does not collect or prepare facial datasets automatically.
- Missing target positions may contain NaN only where the validity mask is false. They are zero-filled before tensor arithmetic and do not contribute supervision. There must be at least one valid target per sample. Do not recenter using only the visible subset: that would change the coordinate frame across occlusions.

These units are neither millimeters nor physically measured depth. A registered scan dataset will require an explicit projection, pose, crop and scale conversion before it fits this contract. A single RGB image alone does not remove the underlying depth/scale ambiguity.

## Training objectives and evaluation

Training minimizes:

`1.0 × masked vertex Smooth L1 + 0.1 × masked edge-length Smooth L1 + 0.01 × masked normal consistency`

The Smooth L1 transition defaults to 0.02 relative units.

- **Vertex loss:** supervised XYZ error for valid registered vertices.
- **Edge loss:** difference in edge lengths when both endpoints have supervision, discouraging local distortion.
- **Normal loss:** one minus the cosine between predicted and target triangle normals, using fully supervised, nondegenerate target triangles. This encourages orientation and surface shape; it is not an independent depth measurement.

Targets are sanitized before arithmetic so masked NaNs cannot contaminate gradients. Degenerate or unsupervised normal terms are omitted. Empty terms produce finite zero contributions; samples with no valid vertices are rejected by the data loader. Loss weights are configurable.

Evaluation reports the component losses and **mean relative vertex Euclidean error**, weighted by valid-vertex counts across batches. It does not report millimeter accuracy, apply hidden Procrustes alignment, infer missing ground truth, or calculate identity metrics. Validation and standalone evaluation use float32, including when CUDA training uses optional AMP. Test-set metrics are separate from validation and do not select checkpoints.

The synthetic source encodes three coefficients into colored spatial patterns and a deformed 18×26 grid of 468 nodes. It is **not an anatomical face simulator**. Independent seed sequences separate training, validation and test samples. These signals verify tensor shapes, optimization, masking and state management only.

## Configurable datasets — no facial data included

`DataConfig.kind` selects `synthetic` or `manifest`. The latter reads only explicitly configured local files inside one dataset root. Paths that escape that root, including external symlinks, are rejected. There is no URL adapter or download fallback.

After explicit data approval, an example private layout is:

```text
 data/approved/
   train.jsonl
   val.jsonl
   test.jsonl
   crops/...
   labels/...
```

Each JSONL record has exactly these keys:

```json
{"image":"crops/sample.png","geometry":"labels/sample.npz","group":"opaque-person-group-001"}
```

Use one consistent, de-identified person/group key across all that person's sessions. This is for dataset partitioning, not identity matching. Training/validation/test groups must be disjoint. The loader also rejects duplicate image records and byte-identical image content crossing splits. It cannot verify that group labels are truthful or detect every near-duplicate frame; dataset curation remains necessary.

Each NPZ contains exactly:

- `vertices`: float32 array `[468,3]`, registered to the documented topology, pose, origin and relative-unit contract.
- `valid`: boolean array `[468]`, indicating supervised positions, not model confidence.

All NPZ loading uses `allow_pickle=False`. Dataset records, label arrays, crops and manifests are private artifacts excluded from Git. The checked-in example configurations contain paths only; no private manifests, images, reference target arrays or dataset installers are included.

The loader fingerprints manifest bytes and image/label contents. Keep the data immutable during an experiment. Hashing is streaming, but the initial scan can take time for a large dataset. Evaluation checks the configured split metadata for leakage too, so the configured train/validation manifests remain required.

Data requirements before useful training include approved/licensed RGB crops with compatible registered 3D supervision, calibrated/defined coordinate conversion, consistent masks, and person-disjoint splits spanning expressions, glasses, poses, lighting and occlusions. MediaPipe labels could later be studied as **pseudo-labels** after approval; they would carry its errors and would not become measured 3D ground truth.

## Experiments, resume and checkpoints

`configs/veyra3d/synthetic_smoke.json` is a tiny procedural configuration. `local_data.example.json` and `windows_cuda.example.json` are future approved-data templates; do not launch real/full training before the data and experiment are approved.

For a **persistent tiny synthetic** experiment, explicitly run:

```bash
./research.sh train --config configs/veyra3d/synthetic_smoke.json
./research.sh evaluate --config configs/veyra3d/synthetic_smoke.json \
  --checkpoint runs/veyra3d/synthetic-smoke/last.pt --split val
./research.sh evaluate --config configs/veyra3d/synthetic_smoke.json \
  --checkpoint runs/veyra3d/synthetic-smoke/last.pt --split test
```

The default `smoke` command is preferable for a check because it removes its temporary artifacts. The persistent commands above are documented for use, not full training started as part of this milestone.

To resume, increase `epochs` in a copied config to the desired **total** completed epoch count, retain the training/data/loss settings, then run:

```bash
./research.sh train --config path/to/resume-config.json \
  --resume runs/veyra3d/synthetic-smoke/last.pt
```

`last.pt` is the most recently completed epoch; `best.pt` is selected by validation loss only. Resume is epoch-boundary only; interrupted partial epochs are discarded. Checkpoints contain state dictionaries for the model, AdamW optimizer and AMP scaler, RNG states, completed epoch, optimizer step count, config, content fingerprints and source signature. They do not contain training images, target vertex arrays, dataset objects or serialized `nn.Module` instances.

Writes use a temporary file and atomic replacement; a failed save preserves the previous checkpoint. Loading explicitly uses `weights_only=True` and CPU mapping before device transfer. Load only checkpoints you created or trust; this is not a general untrusted-model importer.

Runs contain `experiment.json` with configuration/environment/source/data fingerprints, `metrics.jsonl` with scalar measurements, and the two checkpoints. A run-local `.gitignore` protects even custom output folders. Nonempty unrelated run folders are not overwritten. `synthetic_only` identifies the procedural source; a false flag for a local manifest is not an accuracy certificate. All weights remain marked `unvalidated research weights`.

The pipeline seeds Python, NumPy, Torch and epoch-specific sampler generators, seeds loader workers, and enables deterministic algorithms by default. CPU tests verify an interrupted/resumed run has exactly the same final weights as an uninterrupted run. Resume rejects changed source, data contents or training hyperparameters. Total epochs, output directory, device, CPU thread count and loader workers may change; moving devices does not guarantee identical numerical results. PyTorch also does not promise bitwise reproducibility across releases, hardware or operating systems.

Gradient accumulation includes a final incomplete group and averages microbatch losses within each optimizer step. Gradients are clipped and checked for nonfinite values. Per-epoch samples/second are synchronized with the device; CUDA runs additionally report peak allocated GPU memory.

## Memory-only inference API

`Predictor` accepts a PIL RGB crop and returns a float32 `[468,3]` array in RAM:

```python
from veyra3d.inference import Predictor
predictor = Predictor("runs/veyra3d/synthetic-smoke/last.pt", device="cpu")
# vertices = predictor.predict(already_approved_rgb_crop)
# predictor.synthetic_only remains True for procedural smoke weights.
```

This does not open a camera, save a mesh or replace MediaPipe. Synthetic predictions must not be presented as a facial reconstruction demonstration. A real-image UI adapter, presence handling and quality evaluation remain future work after suitable training and validation.

## Windows / GTX 1660 Super preparation

The NVIDIA path is designed but **not tested on the GTX 1660 Super yet**. Recreate the environment on Windows; do not copy Mac virtual environments or the Mac lockfile. From the project root in PowerShell, with Python 3.12 and a compatible NVIDIA driver installed:

```powershell
py -3.12 -m venv .venv-research
.\.venv-research\Scripts\python -m pip install --upgrade pip
.\.venv-research\Scripts\python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv-research\Scripts\python -m pip install -r requirements-research.txt
.\.venv-research\Scripts\python -m veyra3d doctor --device cuda
.\.venv-research\Scripts\python -m veyra3d smoke --device cuda
.\.venv-research\Scripts\python -m unittest discover -s tests_research -v
```

The CUDA 12.6 wheel route is published for PyTorch 2.9.1 by the official installer archive. Verify the installed wheel, driver, card capability and compiled architectures at that time. `doctor` runs real forward/backward/optimizer operations rather than relying only on `cuda.is_available()`. CPU/MPS probes on this Mac cannot validate CUDA kernels or VRAM use.

The Windows template starts with 128×128 input, batch 8, two accumulated microbatches, zero loader workers and float32 training. This targets a modest footprint for 6 GB VRAM; it is not a measured memory guarantee. After the synthetic CUDA check, profile a small approved-data pilot and inspect peak allocation. Reduce the batch size first if needed. Optional CUDA float16 AMP is implemented but disabled until correctness, memory and speed are verified on the actual card; do not assume it is faster.

Before moving beyond that pilot, we need the dataset approval and registration/conversion work above, a fixed validation/test protocol, a recorded Windows dependency/driver environment, and comparison against MediaPipe on the same approved inputs. Full training and switching the live viewer require a separate decision based on those results. The Windows direct Python commands do not inherit the Mac `sandbox-exec` policy; the research implementation still has no network/collection code.

## Files and verification

- `veyra3d/model.py`: encoder and point decoder.
- `data.py`: deterministic non-face fixtures, local manifests, preprocessing, masks, fingerprints and split checks.
- `objectives.py`: masked vertex, edge and normal losses plus exact valid-count aggregation.
- `config.py`: strict JSON configuration and resume fingerprints.
- `runtime.py`: device selection, seeding, source/environment metadata and RNG restore.
- `checkpoint.py`: atomic tensor/primitive state serialization.
- `experiment.py`: training, float32 evaluation, accumulation and resume.
- `diagnostics.py`: small CPU/MPS/CUDA probes and four-step temporary smoke.
- `inference.py`, `__main__.py`: memory-only prediction API and explicit research commands.
- `configs/veyra3d/`: tiny synthetic config and unrun approved-data/Windows templates.
- `tests_research/`: generated fixtures, model/loss/data/experiment/privacy checks.
- `requirements-research*.txt`, `setup-research.sh`, `research.sh`: isolated optional environment and offline runtime wrapper.

On this Mac, the original 51 baseline tests passed before changes and **all 51 passed again in 10.113 seconds** after implementation. The final research run passed **32 tests in 3.752 seconds**. CPU and MPS forward/backward/optimizer probes and both four-step synthetic smoke runs passed, including temporary save/load/resume and separate held-out evaluation. Results are also summarized in the root README. No facial reconstruction accuracy, security performance or Windows GPU performance has been established.

References: [PyTorch 2.9.1 installation archive](https://pytorch.org/get-started/previous-versions/), [reproducibility](https://docs.pytorch.org/docs/2.9/notes/randomness.html), [state-dict serialization](https://docs.pytorch.org/docs/2.9/notes/serialization.html), [Apple MPS setup](https://developer.apple.com/metal/pytorch/).

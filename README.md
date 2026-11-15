# Veyra v0.3 — Veyra3D research foundation

This milestone adds a separate, trainable PyTorch module in `veyra3d/`, with a lightweight encoder-decoder, local dataset contracts, masked geometry objectives, reproducible training/evaluation, and resumable checkpoints. **There is no validated custom facial reconstruction model yet.** Synthetic training checks pipeline mechanics only.

The working MediaPipe webcam prototype remains the v0.2 baseline below. It is preserved while the custom model is developed, and has not been replaced or connected to Veyra3D. FaceUnlock is not modified. No facial dataset download, webcam collection, full model training, commit or push is part of this milestone.

## Veyra3D quick start

The optional `.venv-research` has been installed separately from `.venv`:

```bash
cd /Users/muhammadabbas/Documents/Veyra
./research.sh doctor --device cpu
./research.sh smoke --device cpu
./research.sh smoke --device mps
.venv-research/bin/python -m unittest discover -s tests_research -v
```

On a fresh research setup, run `./setup-research.sh` first. The `smoke` command performs only four optimizer steps on colored patterns and deformed grids, verifies temporary checkpoint save/load/resume and held-out evaluation, then removes its artifacts. It never opens a camera or downloads data.

The default network has **258,644 parameters**, accepts a normalized **128×128 RGB crop**, and outputs **468 centered relative XYZ vertices**. CPU is the default; Apple MPS has also been probed. NVIDIA CUDA support is designed for a future Windows/GTX 1660 Super environment and remains unverified on that hardware.

See [the Veyra3D research guide](docs/veyra3d.md) for architecture, geometry/data contracts, losses, exact test commands, checkpoint/resume behavior, limitations and Windows preparation. Private manifests, datasets, model weights and experiment artifacts remain excluded from Git.

Verified on this M1 Pro Mac on **2026-11-14**:

- **32 research tests passed**, including exact CPU resume, local-manifest integration on generated fixtures, a spawned loader worker, masking, split/content leakage checks, atomic checkpoint failure handling and Git exclusions.
- All **51 existing v0.2 tests passed again** after implementation, including native GPU and camera-cleanup regressions. The baseline source, launcher and dependency files were unchanged.
- CPU and MPS forward/backward/optimizer probes passed; the MPS output matched the CPU reference within the configured numerical tolerance.
- CPU and MPS four-step synthetic smoke runs passed checkpoint roundtrip, resume, validation and a separate held-out test split. Temporary weights were removed afterward. Synthetic losses are not facial reconstruction accuracy.
- No full training or facial data collection/download was performed. The Windows CUDA path remains unverified until tested on that hardware.

## Working MediaPipe prototype — v0.2 baseline

A local research prototype with a live OpenCV webcam preview and a separate interactive **shaded, triangulated facial surface**, driven by MediaPipe Face Landmarker. Wireframe and combined views remain available, with optional adaptive temporal smoothing and adjustable lighting.

**This is estimated relative 3D geometry, not physically measured depth.** Shading and smoothing improve presentation and stability; they do not improve the underlying depth measurement. Veyra has no calibrated physical scale, TrueDepth equivalence, identity matching, authentication, or Face ID-level security.

## Launch

From Terminal or VS Code:

```bash
cd /Users/muhammadabbas/Documents/Veyra
./launch.sh
```

The existing v0.1 environment works without installing new packages. Native Python 3.12, MediaPipe 0.10.21, NumPy 1.26.4, OpenCV 4.11.0.86, and Qt 6.12.0 remain pinned. Qt's installed Essentials package already supplies OpenGL rendering.

On a fresh checkout:

```bash
./setup.sh
./launch.sh
```

Setup downloads the project-local Python runtime, dependencies, and checksum-verified public model. It does not change system Python or FaceUnlock. In VS Code, use **Python: Select Interpreter** and select `.venv/bin/python`.

Allow camera permission for the app hosting Python when macOS asks. If denied, enable Terminal or Visual Studio Code in **System Settings → Privacy & Security → Camera**, quit that app fully, and reopen it. Close competing camera apps or select another index with `--camera 1` if necessary.

## Controls

- **Wireframe / Solid / Solid + Wireframe:** choose an estimated landmark wireframe, smooth neutral shaded surface, or overlay.
- **Smoothing:** toggle the additional viewer filter. The camera overlay stays at the MediaPipe baseline. MediaPipe's existing single-face video smoothing remains active even when this checkbox is off.
- **Response:** default **4 Hz**. Higher values respond faster with less smoothing; lower values reduce stationary jitter more but can delay blinking or expressions. Motion adaptation raises responsiveness during faster movements.
- **Light direction / Intensity:** move the key light and adjust its strength. Ambient light keeps the mask visible at zero key-light intensity.
- **Drag / scroll:** rotate and zoom without interrupting tracking. **R / Reset view** restores the original orientation and zoom.
- **Q / Escape / close either window / Ctrl+C:** stop the worker, release the camera and GPU resources, and close both windows.

The preview is mirrored by default; the 3D view uses unmirrored camera coordinates. The mesh is centered for inspection, with the initial display fit retained across updates. Moving closer changes its displayed size; rotation and expressions still follow the estimated geometry.

Examples:

```bash
./launch.sh --mode combined
./launch.sh --mode wireframe --no-smoothing
./launch.sh --smoothing-cutoff 6 --smoothing-beta 20
./launch.sh --camera 1 --width 640 --height 480 --fps 30
./launch.sh --no-mirror --duration 15
./launch.sh --renderer software
./launch.sh --help
```

`--renderer software` retains the v0.1 wireframe renderer for diagnostics. It cannot render a solid surface; surface and lighting controls are visibly disabled. If OpenGL initialization fails, the viewer falls back to this wireframe mode with an explanation. Camera resolution/FPS requests are advisory; the driver may choose different values.

## Surface and tracking architecture

The v0.1 process layout is preserved:

1. A spawned worker owns the OpenCV camera and the MediaPipe Face Landmarker, explicitly using the **CPU inference delegate** in single-face VIDEO mode.
2. One bounded, volatile POSIX shared-memory slot carries the latest preview, raw estimated coordinates, topology edges, and timing. Nonblocking reads prevent a stalled or crashed writer from blocking Qt. There is no frame queue that can grow with lag.
3. The Qt main process validates the geometry and optionally applies an adaptive, One Euro-style per-vertex low-pass filter. It uses actual capture intervals, raises the cutoff with movement speed, and does not average neighboring facial vertices. Only current filter state and scalar motion statistics remain in RAM.
4. The viewer draws an indexed mesh through a verified **OpenGL 4.1 core** context on the Apple M1 Pro. An immutable triangle index buffer and reusable vertex storage avoid rebuilding geometry objects every frame. Positions and area-weighted vertex normals update for each new tracked face. Fragment shading interpolates the normals for a smooth, neutral material. Rotating or relighting uses shader uniforms without rebuilding normals.

The fixed topology contains **852 triangles over landmarks 0–467**, derived from the ordered closed triangle cycles in MediaPipe 0.10.21's `FACE_LANDMARKS_TESSELATION`. Tests check the indices against the installed upstream definition, distinct triangle vertices, duplicate faces, winding, manifold edges, vertex coverage, and all four boundary loops. It is not a Delaunay triangulation over changing image coordinates. The ten additional iris landmarks remain wireframe points/edges, outside the facial surface.

The outer face boundary, both eye openings, and the mouth opening remain open. **No rear skull, teeth, eyeballs, or hidden-side fill geometry is added.** Looking behind the mask shows the reverse of the same estimated sheet, with darker shading. Occluded landmarks are still model predictions and can be wrong; rotating the viewer does not reveal measured hidden anatomy.

Coordinate conversion remains `(x, -y × image_height/image_width, -z)`, followed by centering, with equal relative scale on all axes. This corrects image aspect ratio without independently stretching the facial axes. The display's window aspect changes only its camera projection. Coordinates have no millimeter scale.

No face, partial/nonfinite landmarks, or tracking loss immediately clears the surface on receipt of that result. Reacquisition starts with fresh filter state, avoiding a blend with an old face. Capture gaps over 250 ms reset smoothing. Stalled camera input clears the display after two seconds; capture failures provide actionable errors. Closing during startup or a blocked camera read has a bounded worker-termination path.

## Performance and comparison

The UI distinguishes these measurements:

- **FPS / inference / frame age** in the preview: worker processing throughput; Face Landmarker call time alone; elapsed software time from capture request to presentation. Frame age is not sensor-to-screen latency.
- **Render FPS:** completed widget frame swaps while a face is present, including redraws for interaction/UI changes. It may exceed tracking FPS. Startup and no-face waiting are excluded.
- **Render CPU:** OpenGL submission time, excluding GPU completion and Qt window compositing. **GPU:** asynchronous elapsed GPU draw time when supported. **Normals:** CPU normal-update cost. Live rendering does not call `glFinish` or read pixels back.
- **Motion RMS raw → shown:** frame-to-frame vertex displacement on the *same received frames*, before and after additional smoothing, in relative units/frame. It includes intentional head/expression movement; it is not a pure jitter score or a physical accuracy measurement. The filter CPU cost is shown alongside it.

To compare smoothing, hold still and toggle it under the same lighting and distance, then repeat blinking, smiling and small head movements. Watch raw/shown motion, filter cost, responsiveness and FPS. Compare the displayed values without recording images or landmark arrays.

For a repeatable camera-free benchmark:

```bash
./launch.sh --benchmark
```

This reads the public generic reference mesh already inside the downloaded model into RAM. It measures v0.1 software wireframe and v0.2 shaded rendering at matching pixel sizes, plus invented stationary noise and a step movement at 30 Hz. It never opens the camera or saves benchmark geometry/images. Only this diagnostic uses `glFinish` to time completed GPU work; window compositing is excluded.

Observed on **2026-11-14**, Apple M1 Pro, macOS 27.0.1, native Python 3.12.16:

- Completed 600×600 logical-pixel rendering at DPR 2: v0.1 software wireframe **17.68 ms median / 32.46 ms p95**; v0.2 shaded surface **1.23 ms median / 2.19 ms p95**. This demonstrates a rendering benefit on this Mac; it is not an end-to-end camera FPS comparison.
- Synthetic stationary-noise RMS per coordinate: **0.001000 → 0.000555**, approximately **44.6% reduction**. A 0.05-relative-unit step per coordinate reached 90% within **67 ms** at 30 Hz. Median filter cost was approximately **0.022 ms**. These results describe the invented signal, not human tracking accuracy.
- Live smoothing-off smoke session: 40 previews, 37 surface updates; final snapshot **24.5 tracking FPS / 16.9 ms inference**.
- Live default-smoothing session: 199 previews, 192 surface updates; final snapshot **29.0 tracking FPS / 8.0 ms inference**, **33.2 render FPS**, **0.43 ms render CPU / 0.02 ms GPU / 0.18 ms normals / 0.056 ms filter**. Both sessions stopped cleanly. They had uncontrolled face/scene conditions; their FPS difference does not establish a benefit from smoothing.

Performance varies with capture rate, tracking conditions, display scale and other workloads. Inference acceleration is not enabled; the verified GPU path is for rendering only.

## Automated checks

```bash
# Numerical, mocked-camera, shared-memory and offscreen UI checks
.venv/bin/python -m unittest discover -s tests -v

# Also run native macOS GPU and shutdown regressions, without a camera
VEYRA_NATIVE_UI_TEST=1 .venv/bin/python -m unittest discover -s tests -v

# Real CPU model, synthetic blank frames, no camera
./launch.sh --self-test

# Offline model integrity
.venv/bin/python scripts/download_model.py --check
```

The native surface check renders only the public generic reference template into memory. It verifies shaded pixels, display modes, lighting response, rotation, clearing, recovery and visible fallback. Other tests cover topology, coordinate conversion, smoothing noise/step behavior, invalid/missing data, tracking loss and recovery, camera release on errors, bounded shared memory, and existing v0.1 controls. Native subprocess tests exercise the timer, both window-close paths and terminal interruption.

The original **27-test v0.1 baseline passed** before changes. The final native-enabled v0.2 run passed **all 51 tests in 10.939 seconds** on this Mac. The real CPU model self-test, model checksum, shell syntax, Python compilation and `git diff --check` also passed. Human tracking quality with glasses, without glasses, and specific expressions remains a manual observation.

## Manual test

Use even indoor lighting and a comfortable webcam distance. Do not record the session or save screenshots or scans.

1. **Without glasses:** track for 30 seconds, blink, smile, raise eyebrows, and open your mouth slightly. Check eye/lip alignment in the preview and matching movement in the surface. Look for jitter, clipped features, lag and unexpected spikes.
2. **With glasses:** repeat the same actions for 30 seconds. Check eye landmarks and surface behavior around reflections or occluded features. Adjust lighting/screen angle if reflections obscure the eyes; smoothing cannot correct bad landmarks.
3. **Smoothing comparison:** hold still for ten seconds with smoothing off, then on, at the same distance. Compare raw/shown motion RMS and timing. Repeat expressions and head movements; increase Response toward 6–8 Hz or disable smoothing if motion feels delayed. Lower it cautiously for more stationary stability.
4. **Head movement / inspection:** turn left/right about 15–20 degrees, tilt, and nod slowly. Inspect all three display modes while tracking continues. Change light direction/intensity, drag/zoom, and reset with R. Rear/occluded geometry must remain understood as a model estimate on an open mask.
5. **Loss / recovery:** leave the camera view or cover it. The old surface must disappear. Return with a different expression or pose and check immediate fresh acquisition rather than a sweep from the previous face.
6. **Shutdown:** close either window and confirm both exit and the camera indicator turns off. Repeat using Q, Escape, Ctrl+C, and `./launch.sh --duration 5`; then relaunch successfully.
7. **Errors / privacy:** try `./launch.sh --camera 99` and confirm an actionable error; close to exit. After setup, disconnect networking and launch normally. Check that no captured images, videos, landmark arrays or mesh exports appear in the project folder.

Report qualitative observations and approximate timing only. Neither successful tracking nor a smoother-looking surface establishes recognition, authentication or measured depth accuracy.

## Files and privacy

- `veyra/core.py`, `tracking.py`, `transport.py`: existing configuration/conversion/timing, CPU webcam worker, and bounded volatile transport. Configuration and invalid-detection handling were extended; shared-memory architecture remains intact.
- `veyra/mesh.py`, `topology_data.py`: validated static triangle connectivity and area-weighted normals. The data file contains only index connectivity, not a reference face or user scan.
- `veyra/smoothing.py`: adaptive filter, reset behavior, and RAM-only same-frame motion comparison.
- `veyra/render_gl.py`: persistent GPU buffers, smooth shading, lighting and asynchronous timing.
- `veyra/render_common.py`, `render_software.py`: shared orbit controls and the retained v0.1 wireframe fallback.
- `veyra/app.py`, `view3d.py`, `__main__.py`: two-window UI, controls, lifecycle and launch flags.
- `veyra/diagnostics.py`: public-reference renderer benchmark and deterministic synthetic smoothing comparison.
- `tests/`: automated tests; no recorded webcam fixtures or personal scans.
- `third_party/mediapipe-LICENSE.txt`: Apache 2.0 license for the upstream connectivity definition.
- `requirements.txt`, `requirements.lock.txt`, `setup.sh`: unchanged compatible pinned dependencies and project-local setup.
- `launch.sh`: preserved macOS network-denial policy. `scripts/download_model.py`: explicit setup-only HTTPS download with pinned checksum; startup never downloads assets.
- `.gitignore`: excludes environments, caches, models/checkpoints, datasets, recordings, images, arrays and geometry exports.

All processing remains local. The application keeps bounded working frames, geometry and filter history in volatile RAM/shared memory; it does not save webcam frames, screenshots, facial scans, landmarks, embeddings, recordings or biometric records. There is no export, remote service or identity matching feature. Shared memory is closed and unlinked during shutdown. Public pretrained software and code indices are separate from captured user data.

On macOS, always use `launch.sh` so its network-denial sandbox also applies to native dependencies and the worker. Setup has separate network access to download public software/model assets. If the macOS sandbox is unavailable, launch fails with an explanation. Other operating systems are unverified and lack this launcher network policy.

The separate FaceUnlock macOS application is not connected or modified. No commit or push is performed by setup, launch or development scripts.

## Limitations and references

The 468-vertex facial surface is coarse. It has no camera calibration, measured depth, full head reconstruction, teeth, eyeballs or physically based skin material. Smoothing can suppress small expressions or introduce lag. Glasses, glare, profile poses, lighting and occlusion can destabilize the pretrained estimator; added smoothing does not establish correctness. Only one face is tracked. A visible wireframe fallback preserves tracking if the shaded renderer is unavailable.

`pip check` can print an upstream MediaPipe wheel-tag warning: its internal tag says x86_64 although the native library contains both arm64 and x86_64. Native arm64 CPU inference is exercised successfully. Qt offscreen tests may print benign size-hint/font notices. These do not replace the native GPU tests.

- [MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker?tab=Python)
- [Pinned MediaPipe triangle-cycle source](https://github.com/google-ai-edge/mediapipe/blob/v0.10.21/mediapipe/tasks/python/vision/face_landmarker.py)
- [Qt OpenGLWidget](https://doc.qt.io/qtforpython-6/PySide6/QtOpenGLWidgets/QOpenGLWidget.html)
- [Qt OpenGL example](https://doc.qt.io/qtforpython-6/examples/example_opengl_contextinfo.html)
- [OpenCV camera API](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html)

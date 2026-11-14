# Veyra v0.1

A local research prototype that follows one face from a webcam using MediaPipe Face Landmarker. It shows a camera preview with landmarks and mesh connections, plus a separate interactive view of the estimated facial geometry.

**The 3D view contains relative, model-estimated landmarks, not physically measured depth.** It has no millimeter scale, calibrated depth camera, or TrueDepth equivalence. Veyra v0.1 does not recognize people or provide authentication.

## Launch

From Terminal or the VS Code integrated terminal:

```bash
cd /Users/muhammadabbas/Documents/Veyra
./launch.sh
```

The installed `.venv` uses native Apple Silicon Python 3.12. On a new checkout, run `./setup.sh` first. Setup downloads the Python runtime, pinned dependencies, and the public pretrained model. Everything is installed inside this folder; system Python and FaceUnlock are not modified. In VS Code, select `${workspaceFolder}/.venv/bin/python` with **Python: Select Interpreter**.

```bash
./setup.sh
./launch.sh
```

Accept macOS camera permission for the application hosting Python, such as Terminal or Visual Studio Code. If access is denied, open **System Settings → Privacy & Security → Camera**, enable the relevant application, quit it fully, and reopen it. A camera error can also mean another application has the camera, the selected index is wrong, or the device disconnected.

## Use

- The camera preview is mirrored by default. Face landmarks and mesh connections follow the visible face.
- Drag inside the separate 3D window to rotate the view. Scroll to zoom. Press **R** to reset the view.
- Press **Q** or **Escape**, close either window, or press **Ctrl+C** in the terminal to quit and release the camera.
- If tracking is lost, the geometry clears and the application displays **No face detected**. Move into view with even lighting and look toward the camera.
- **FPS** measures processed-frame throughput. **Inference latency** measures the Face Landmarker call alone, excluding camera capture and rendering. Frame age is displayed separately to expose delay between capture and display. These are software measurements, not sensor-to-screen timing.

Examples:

```bash
./launch.sh --camera 1
./launch.sh --width 640 --height 480 --fps 30
./launch.sh --no-mirror
./launch.sh --duration 15
./launch.sh --help
```

Camera resolution and FPS are requests; the camera driver may choose different values. `--duration` ends a bounded session automatically.

## Local data handling

The application retains only current working frames and landmarks in RAM. It does not save webcam frames, screenshots, videos, landmarks, face scans, embeddings, or biometric records. There is no recording or export feature, no remote server, and no FaceUnlock integration.

On macOS, `launch.sh` starts the application and its worker inside a process sandbox that denies network access. Setup has network access separately to fetch public software and the pretrained baseline. Launch does not download a missing model. Use the launcher for normal operation so that network isolation applies to native dependencies as well as Python code. If the macOS sandbox is unavailable, the launcher fails with an explanation. Other operating systems are unverified and do not receive this launcher network sandbox.

The public pretrained model is software, not a scan of the webcam user. It lives in `models/face_landmarker.task`, is ignored by Git, and is checked against a pinned SHA-256 checksum. The downloader rejects an unexpected or corrupt file without replacing it silently. To repair one, move or remove that model file and run `./setup.sh` again.

`.gitignore` excludes environments, downloaded runtimes, caches, datasets, recordings, checkpoints, images, videos, geometry exports, and common biometric array formats. No commit or push is part of setup or launch.

## Implementation

Inference explicitly uses the **CPU** delegate. MediaPipe's single-face video mode provides tracking and temporal smoothing with monotonically increasing timestamps. OpenCV supplies webcam frames. Camera capture and inference run in a worker process. One volatile POSIX shared-memory slot holds the latest frame and geometry; nonblocking reads keep the UI responsive if the worker stalls. The slot is closed and unlinked at shutdown. Qt runs the two windows on the main process. The interactive wireframe is rendered in software. MediaPipe itself may initialize a graphics context; the inference delegate remains explicitly CPU.

MediaPipe returns image-normalized x/y coordinates and relative z scaled like x. The 3D view corrects for the image aspect ratio, centers the estimated face, and keeps equal relative scaling across the axes. Rotating the viewer changes the inspection angle; it does not produce observations of the hidden side of the face. Head movement and expression still come from live estimates, with expected jitter and occlusion errors.

The baseline is deliberately pinned to MediaPipe `0.10.21` and Python `3.12`, with compatible NumPy, OpenCV, and Qt packages. Newer MediaPipe Python wheels have an [upstream telemetry report](https://github.com/google-ai-edge/mediapipe/issues/6291); the macOS launcher denies network access independently of package behavior. Acceleration has not been verified and is not enabled.

Files:

- `veyra/` — application entry point, camera/inference lifecycle, geometry conversion, performance measurements, and the two-window UI.
- `veyra/core.py` — configuration, model validation, coordinate conversion, and timing.
- `veyra/tracking.py` — OpenCV capture, CPU Face Landmarker inference, and overlay drawing.
- `veyra/transport.py` — bounded volatile shared memory between camera and UI processes.
- `veyra/app.py` and `veyra/view3d.py` — live preview, interactive software 3D rendering, and guarded shutdown.
- `tests/` — automated checks using synthetic data and mocked devices; no webcam frames or face scans are fixtures.
- `requirements.txt` — pinned compatible Python dependencies.
- `requirements.lock.txt` — complete installed dependency snapshot used by setup.
- `setup.sh` — local tool bootstrap, Python 3.12 environment, dependency install, and model download.
- `launch.sh` — exact application launcher, including macOS network isolation.
- `scripts/download_model.py` — explicit HTTPS model download with pinned size/checksum and atomic installation.
- `.gitignore` — excludes generated files and sensitive data formats.

## Automated checks

```bash
.venv/bin/python -m unittest discover -s tests -v
VEYRA_NATIVE_UI_TEST=1 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/download_model.py --check
./launch.sh --self-test
```

The first test command runs the portable synthetic checks; the second also opens brief native macOS windows to check process and window shutdown. Neither command opens a camera. The self-test checks the real CPU inference baseline on three blank images without opening the webcam.

Verified on **2026-11-14**, on this Apple M1 Pro Mac with macOS 27.0.1, native Python 3.12.16, MediaPipe 0.10.21, and Qt 6.12.0:

- All **27 tests passed**, including native window shutdown, a real spawned shared-memory worker, mocked camera/inference failures, coordinate scaling, timing, and interactive rendering.
- Real CPU model self-test passed on three blank images with no face detected; the downloaded model's checksum passed.
- A bounded live webcam run displayed **134 previews at 640 × 480**, then stopped cleanly. Its final snapshot was **28.9 FPS**, **7.5 ms inference latency**, and **19 ms frame age**. These values vary with workload and capture conditions.
- A connection attempt under the launcher's macOS policy was denied, verifying network isolation.
- `pip check` found no dependency conflicts. It prints an upstream MediaPipe wheel-tag warning: the wheel's internal tag says x86_64 although its native library contains both arm64 and x86_64. The running interpreter and successfully exercised inference are arm64.

Automated checks and the bounded webcam run do not establish tracking quality with glasses or specific facial movements. Those human observations remain pending until the following manual tests are performed.

## Manual webcam test

Do these tests in even indoor lighting at a comfortable webcam distance. Do not take screenshots or record the sessions.

1. **Without glasses:** launch Veyra and stay in view for 30 seconds. Blink, smile, open your mouth slightly, and raise your eyebrows. Check alignment of the lips, eyes, nose, and face outline. Observe FPS and inference latency while looking for visible lag or jitter.
2. **With glasses:** repeat the same actions for 30 seconds. Look for unstable eye landmarks, reflection problems, or lost tracking. Try a small lighting or screen-angle change if reflections obscure the eyes. A tracking success here is an observation, not recognition or identity verification.
3. **Slight head movement:** slowly turn left and right about 15–20 degrees, tilt slightly, and nod. Check that the preview overlay and estimated mesh follow the motion. Rotate the 3D view while tracking continues and use **R** to restore its view. Large profile rotations and occluded features are outside this initial test.
4. **No face:** step out of view or cover the camera. The status should change to **No face detected**, and the previous mesh should disappear. Return and confirm it reacquires the face.
5. **Shutdown:** close either window. Both windows should exit, the process should finish, and the webcam indicator should turn off. Relaunch and repeat with **Q**, **Escape**, and **Ctrl+C**. Also check `./launch.sh --duration 5`.
6. **Camera errors:** try an unavailable index, for example `./launch.sh --camera 99`. Confirm an actionable error and a clean exit. Denying camera permission should also explain how to enable it.
7. **Offline operation and privacy:** after setup, disconnect networking and launch again. Tracking should still work. Check the project folder after quitting: no new captured images, videos, landmark arrays, or geometry exports should appear. The downloaded model and environment are the expected setup artifacts.

If sharing test feedback, report only qualitative tracking observations and approximate FPS/latency; do not create biometric records for this prototype.

## References

- [MediaPipe Face Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker?tab=Python)
- [MediaPipe 0.10.21 package and platform wheels](https://pypi.org/project/mediapipe/0.10.21/)
- [Official version-1 Face Landmarker model](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)
- [OpenCV camera capture API](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html)

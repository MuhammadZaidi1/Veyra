# Included notices and license scope

- `mediapipe-LICENSE.txt` covers the existing MediaPipe-derived face topology.
- `mediapipe-reference/` contains the pinned public canonical geometry,
  connectivity, unmodified Apache-2.0 LICENSE, and source/hash provenance.
  These are reference assets, not private images, scans, model weights or data.
- `blender-scripts-GPL-3.0-or-later.txt` supplies the full GPL text for scripts
  explicitly marked `SPDX-License-Identifier: GPL-3.0-or-later`: Blender prototype,
  correspondence, audit rendering, annotation preparation/UI/integration check.
  The text was copied unchanged from the installed Blender license/spdx directory;
  no Blender or MPFB executable/add-on code is vendored here.
- MakeHuman core assets are loaded from the user's MPFB installation, not bundled.
  The loader checks the base OBJ's explicit CC0 declaration and records its hash.
  Named core morph inputs and their hashes are recorded in ignored local reports.
  See the Blender prototype/correspondence docs for upstream sources and limits.

These notices do not assign a blanket license to all Veyra application/research
code. The repository owner must choose that policy before offering a generally
licensed public software release. Do not infer commercial rights for an external
dataset from the license of these tools or reference files.

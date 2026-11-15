# MediaPipe canonical reference for correspondence review

Official upstream: https://github.com/google-ai-edge/mediapipe
Pinned revision: 02ca022d573c004c463d8c61eb3f8ce17e95a7fc

Files fetched for this workflow:
- canonical_face_model.obj: unmodified reference geometry, exactly 468 vertices.
- face_mesh_connections.py: unmodified region connectivity; retains upstream
  copyright header (The MediaPipe Authors) and Apache 2.0 notice.
- LICENSE: upstream Apache License, Version 2.0, unmodified.
- provenance.json: source URLs and SHA256 for every fetched file.

The source connectivity module retains its copyright notice. No neural-network
weights or training datasets are included. The canonical mesh is used only as a
geometric ordering/anatomical reference; generated MakeHuman anchors are derived
candidates, not official MediaPipe correspondences or endorsed annotations.

The public canonical OBJ is tracked through a narrow .gitignore exception.
Private/generated meshes remain ignored. To explicitly restore a missing reference:

    .\.venv-research\Scripts\python.exe -m scripts.run_correspondence_review --fetch-reference

This explicit fetch is the only network-enabled operation in the workflow. It
checks embedded SHA256 values before writing. Normal fitting runs stay offline.
Preserve this file, LICENSE, original source notices and provenance when sharing
the reference/derived mapping. Other repository code is not relicensed by this
reference. Do not confuse the Apache reference with CC0 MakeHuman core assets.

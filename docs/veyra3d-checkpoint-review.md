# GitHub checkpoint review — 2026-11-16

Base commit: `73a0bcc` (`v0.3.0: Add Veyra3D neural network and training pipeline`).
Source checkpoint only. No commit, push, tag, dataset download or real training.

## Reviewed scope since that commit

- Model: replaces adaptive 2x2 pooling with fixed average pooling for supported
  input sizes; no parameter/checkpoint-shape change. Strict determinism remains.
  Forward/backward equivalence, repeated CUDA gradients and exact CUDA resume are
  covered by the research tests.
- Import/experiment preparation: scan-derived-label packaging, explicit human
  provenance assertions, path/split/shape checks, example pilot config and bounded
  benchmark script. Research-restricted FaceScape is explicitly NOT cleared for
  commercial product development or distribution. No dataset is included.
- Blender prototype/correspondence/audit: CC0 core mesh plus procedural assets,
  source ID/hash preservation, candidate fitting, authored deformation proxies,
  independent projection/raster checks and scoped topology diagnostics. Existing
  anatomical defects and unverified source rings remain documented.
- Manual annotation: 56 editable surface markers, point decisions and versioned
  export, accepted-point-only rebuild preserving 468 slots, reference cards and
  before/after panels. No human approval or corrected anatomy is fabricated.
- Tests, documentation and public pinned MediaPipe reference/license provenance.
  Generated reports and images remain local under ignored `runs/`.

## Checkpoint hygiene fixes

- Ignore Blender scenes/backups, EXR/HDR output, scratch directories/files and the
  accidental PowerShell ModuleAnalysisCache. Image/video/mesh extension ignores
  now cover uppercase extensions too. Existing dataset, checkpoint, environment,
  private-manifest and run-folder exclusions remain.
- Narrow exception for the public Apache-2.0 canonical reference OBJ. Nothing
  else in the ignored scan/mesh/image categories is included. Reference hashes
  match provenance; `.gitattributes` preserves upstream bytes across checkouts.
- Removed absolute Windows font paths and the hardcoded Program Files fallback.
  Blender supports PATH or explicit `--blender`; Windows discovery uses the
  environment. Audit `--asset-root` is explicit outside the Windows default.
  No drive-qualified absolute Windows paths remain in portable research/scripts.
  Windows command examples in docs are illustrative; local generated provenance
  may contain resolved absolute paths and is intentionally excluded.
- Annotation preparation accepts configurable mapping/prototype inputs; rebuild
  accepts reference geometry, and card generation accepts an audit directory.
  Blender preparation now reports Python errors through a nonzero exit status.
- Added complete GPL-3.0-or-later text for the already marked Blender-side scripts
  and a scoped third-party notices inventory; preserved Apache notices/provenance.
- Moved annotation tests into `tests_research` so full research discovery runs them.
- Cleaned up annotation UI registration on integration-test/preparation exit;
  disabled unnecessary thumbnail writes during background annotation preparation.

## Verification

- Full research suite: **70 tests pass**, including CUDA strict deterministic
  backward and exact checkpoint/resume. All training-like work is limited to
  tiny procedural test fixtures; no real facial training was started.
- `python -m veyra3d smoke --device cuda`: passed, four synthetic optimizer steps,
  temporary checkpoint roundtrip/resume passed and temporary files removed.
- Fresh Blender/MPFB one-face prototype: passed. Analytical projection error
  0.0000133 px; independent raster error 0.5695 px.
- Fresh candidate geometry integration: 468 anchors, zero numerical attachment
  failures; 16 identity/expression cases, all still unvalidated. Heuristic review
  flags remain visible (312 flagged candidates); these are not anatomical approval.
- Fresh configurable annotation project: 56 markers and three test renders;
  Blender integration passed selection/snap, free-move and source-edit rejection,
  versioned export and UI cleanup. No point acceptance was simulated.
- Annotation draft rebuild: preserves 468 slots, zero applied corrections,
  displacement/duplicates/attachment failures all zero; six canonical mouth-cap
  reversals remain separately reported. Veyra skin reversals remain zero.
- Ignore-rule, public-reference checksum, and absolute-path tests pass; Git
  whitespace checks pass. Staged files checked for generated/private artifacts
  and common secret patterns; none found.

Commands from project root:

```text
python -m unittest discover -s tests_research -q
python -m veyra3d smoke --device cuda
python -m scripts.run_blender_prototype --output runs/veyra3d/checkpoint-prototype-001
python -m scripts.run_correspondence_review --prototype runs/veyra3d/checkpoint-prototype-001 --output runs/veyra3d/checkpoint-correspondence-001 --geometry-only
python scripts/run_anatomical_annotation.py prepare --project runs/veyra3d/checkpoint-annotation-002 --mapping runs/veyra3d/checkpoint-correspondence-001/mapping-candidates.json --prototype runs/veyra3d/checkpoint-prototype-001
```

Use the research environment Python, and fresh output names when reproducing.
The integration script `scripts/check_blender_annotation.py` runs inside Blender
against the generated annotation blend with `--background --factory-startup
--python-exit-code 1`. Mouse interaction/anatomical interpretation still require
human inspection. OS portability was checked statically on Windows, not executed
on macOS/Linux. PyTorch pin-memory and Blender use_nodes deprecation warnings are
nonblocking upstream compatibility follow-ups, not test failures.

## Remaining decisions and limitations

No technical blocker remains for an explicitly unvalidated research-source
checkpoint. Generated render/annotation/candidate files are not restored by Git;
recreate them using the prototype, correspondence and audit instructions.
All 468 anatomical correspondences remain unvalidated; complete human review
before creating any training labels or presenting facial accuracy claims.

There is no blanket repository LICENSE yet. The owner must choose the licensing
policy for the remaining original application/research code before advertising a
licensed public software release. Do not infer it from the scoped Apache/GPL
notices. No dataset/commercial-weight rights are conferred by this checkpoint.

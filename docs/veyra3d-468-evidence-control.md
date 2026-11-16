# 468-point evidence policy and diagnostic control

The 468-point decoder, existing 852-face loss topology and geometry contract are
unchanged. **No native MakeHuman representation was adopted.** The result is in
`runs/veyra3d/468-evidence-001`.

No correction has new defensible anatomical support. The original mapping is
retained byte-for-byte; all 56 disputed points remain unresolved. Their numerical
improvement scores are not enough. Crucially, the remaining 412 are still
candidates too. No MakeHuman ground-truth mask is manufactured from that exclusion.

## Stronger sources and remaining gaps

- MediaPipe's pinned Apache-2.0 canonical reference establishes exact vertex
  identity on its own 468-point template. It does not give a MakeHuman triangle
  binding. The documented template semantics support a reference-only diagnostic.
- The installed CC0 MakeHuman mesh, rigs, morphs and skinning weights provide
  source geometry and regional cues, not a verified MediaPipe binding. The prior
  surface, ordering and collision tests still stand. No old trial was upgraded.
- **ICT FaceKit Light** is stronger: the authors provide 68-point and extended-jaw
  definitions, under MIT terms. Its topology differs from MakeHuman and the
  published indices cannot be pasted across. It lacks a complete, independently
  verified dense transfer. Only its documentation was inspected: no code or assets
  were copied. Its separately licensed full version is excluded.
- A detector on a synthetic render supplies estimated correspondences. It may be
  useful for a future explicitly named pseudo-label study; it is not a substitute
  for independent anatomical truth. Repeated reprojection error alone is inadequate.

Primary sources: [ICT README](https://github.com/USC-ICT/ICT-FaceKit),
[MIT license](https://raw.githubusercontent.com/USC-ICT/ICT-FaceKit/master/LICENSE),
[MediaPipe description](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md),
[MPFB asset license](https://raw.githubusercontent.com/makehumancommunity/mpfb2/master/LICENSE.md).

The per-region requirements are executable in `mediapipe_evidence_policy.py`:

- Lips: an explicit external vermilion boundary, distinguished from inner rim and
  labiomental crease, then a dense binding of its ordered samples to the reference.
- Corners: surface commissure sites, not the mean of two rig supports.
- Forehead/temples: the canonical patch boundary, not skull or hairline estimates.
- Jaw/chin: source sites with matching semantics and expression stability.

If semantic evidence is missing, fitting stops. No global shifting was attempted.
If it becomes available, source attachment, joint topology and expression tests
must still pass before labels are used. References on another topology do not
become a verified source binding merely because their index lists are available.

## Partial-label experiment

The existing loss excludes vertices and also edges/triangles touching invalid
points. Tests show corrupted masked targets have no effect. A future sparse
experiment can therefore keep the architecture intact. The current MakeHuman mask
has zero eligible points, so a proper dataset loader must reject it. A mask alone
cannot justify a training run.

The model centers all outputs. A partial dataset needs a trusted origin too;
centering the trusted subset or a mixture of verified/speculative anchors would
silently corrupt the target frame. There is no such reference supplied yet.

No experiment based on these MakeHuman candidates is ready. The strongest result
is a canonical-only consistency control, not a biometric benchmark.

## Control outcome

Three poses, with/without frame-only glasses, were rendered. Source indices stayed
fixed; projection error is about 0.000046 px. Raster error is about 0.69 px. No
paired target changed. This cannot establish identity performance, and no real
prescription lenses were tested. No datasets were generated, no training started.

Reproduce with a fresh output:

    python scripts/run_468_control.py --output runs/veyra3d/468-next

Future work needs a verified transfer plus more than one lawful identity. It must
not split the same canonical identity across partitions, nor turn "not disputed"
into "verified." Ten new tests passed; originals, including the checkpoint, are
preserved.

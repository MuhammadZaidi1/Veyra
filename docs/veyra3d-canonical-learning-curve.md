# Bounded canonical learning-curve result

Veyra can learn image-dependent coordinates of the canonical template. The best
validation error is **0.00164174**, versus the unchanged constant-template
baseline **0.01471475** (88.8% lower). This is pose-conditioned recovery of one
memorized synthetic shape, not evidence of reconstructing new human geometry.

All results are in `runs/veyra3d/canonical-learning-curve-001`, with a separate
directory per experiment. The 468-output architecture, original control,
MakeHuman research, old results, and Git checkpoint were preserved. No images or
labels were generated. The existing ten renders and their original coordinate
contract were used. Every experiment stopped at 200 optimizer steps.

| Experiment | Best step | Fit XYZ error at best | Validation XYZ error | Update time / total run time |
| --- | ---: | ---: | ---: | ---: |
| Original random initialization, longer | 200 | 0.00280293 | 0.00549065 | 6.61s / 7.89s |
| Image-blind control | 110 | 0.02039408 | 0.01377089 | 6.26s / 7.55s |
| Training-template initialization | 200 | 0.00011744 | **0.00164174** | 6.19s / 7.29s |
| Hold out light level 5 at training yaws | 180 | 0.00132047 | 0.01048221 | 6.17s / 7.34s |

All errors are mean Euclidean per-vertex XYZ distances in image-width units.
Total run time includes evaluation and checkpoint writes. Four runs used **800
updates total**, about 30 seconds of combined run time. Peak PyTorch tensor
allocation was approximately 69 MiB, excluding driver/context memory.

## Diagnosis

**Coordinates:** the pinned-source, 468-vertex order, full label recomputation
and raster checks passed. Training target axes span approximately [-0.386,
0.389]; maximum target-centroid rounding error is 2.37e-8. All three coordinates
are divided by the 0.23m orthographic width. Input RGB is normalized to [-1,1].
Square 128-to-64px resampling does not change image-width target units. There is
no evidence of a coordinate or unit mismatch.

**Duration and initialization:** the original random model starts at 0.25622
validation error. Its four-step result reproduces the previous 0.22662; 200 steps
reach 0.00549. Four steps were insufficient to learn even the template. The
template-initialized run zeros only the final linear layer's weights and sets
its bias to the mean of the **six training targets**. It adds no model component
or validation information. Its initial output is the same constant baseline.

| Steps | Original validation XYZ error | Template-initialized validation XYZ error |
| ---: | ---: | ---: |
| 0 | 0.256220 | 0.014715 |
| 4 | 0.226616 | 0.007548 |
| 20 | 0.089942 | 0.003602 |
| 50 | 0.019772 | 0.001789 |
| 100 | 0.009686 | 0.001658 |
| 150 | 0.009627 | 0.001653 |
| 200 | 0.005491 | 0.001642 |

**Loss scaling:** training uses SmoothL1, beta 0.02, averaged over B*468*3
coordinates. Reported XYZ error is a different statistic. The baseline's
SmoothL1 validation loss is 0.00236942; the selected model's is 0.00003228.
All runs keep the same vertex-only objective and AdamW learning rate 0.001.
No loss-rescaling change was needed. All gradients were finite; the largest
observed norm before clipping was 0.3874, below the clip threshold of 1.

**Image dependence:** the blind model receives the same mean training RGB image
for every example, including validation. It cannot distinguish camera yaws.
Its slight improvement over the arithmetic-mean baseline is insufficient
evidence of image use: objective choice and validation checkpoint selection can
favor a slightly different constant. The selected RGB model is substantially
better. Replacing its validation RGBs with the same mean training image worsens
error from 0.001642 to 0.012426. That substitution is an input-ablation diagnostic
and may be outside the training-image distribution; the separately trained
blind control provides additional evidence.

## Generalization and overfitting

Original training yaws are -12/0/+12 degrees, validation is -6, and test is +6.
All share one shape. Both validation images have the same yaw, so many checkpoint
comparisons can favor that single pose. No statistical confidence or real-world
performance claim is justified.

Selection was frozen using only validation, excluding the image-blind control.
The selected checkpoint was then evaluated **once** on the reserved +6-degree
test pair: **0.00746996**, versus its baseline **0.01471475** (49.2% lower).
No training or tuning followed this test result. Test SmoothL1 loss is 0.00067962.

The selected model fits training yaws within 0.005 degrees on average, according
to a fixed-scale geometric best fit to the known template. At the -6-degree
validation yaw it fits -5.23/-6.47 degrees across lights. At the +6-degree test
yaw it fits only +3.13/+2.64 degrees. These are geometric diagnostics, not
independent anatomical or learned yaw labels. The asymmetry shows an unresolved
generalization limitation.

Training fit is almost exact while validation nearly plateaus after 100 steps;
test error is about 4.55 times validation error. The lighting experiment fits
only the three key-energy-3 training images. At its best validation step,
unseen light at seen yaws gives 0.008575 error, while unseen yaw gives 0.010482.
After step 180, training error improves from 0.001320 to 0.001168 while validation
worsens to 0.011017: mild late overfitting. That experiment changes sample count
and light coverage together, so it does not isolate a pure causal yaw effect.

The network therefore learns more than a constant output: it learns
image-dependent pose-conditioned coordinates around a memorized template.
These data contain no independent shape or identity variation. There is no
evidence yet of unseen-person reconstruction, webcam or glasses robustness,
identity recognition, liveness, or unlock security.

## Artifacts and reproduction

New source files are `scripts/canonical_learning_curve.py`,
`scripts/summarize_canonical_learning_curve.py`, the matching plan in
`configs/veyra3d/canonical_learning_curve.json`, and four diagnostic unit tests.
The original two-epoch control's restrictions remain intact. Use the new
step-bounded runner, rather than extending the original control.

Each experiment directory contains its exact configuration, every pre-update
batch loss and gradient norm, frozen training/validation metrics at step 0,
1/2/4 and every ten updates, best checkpoint, final diagnostic state, environment
and results. `best.pt` is the complete research checkpoint; `last.pt` is a final
diagnostic state dump. The suite contains data/code signatures, the fixed
baseline, selection policy, runner source snapshot, machine-readable summary,
diagnosis and combined `learning_curves.csv`. Test data never participate in
optimization or selection.

```powershell
# Create the ten canonical inputs once if absent; no MPFB or extra assets needed.
if (!(Test-Path runs/veyra3d/canonical-control-001)) {
    .\.venv-research\Scripts\python.exe -m scripts.canonical_control render --output runs/veyra3d/canonical-control-001
}
.\.venv-research\Scripts\python.exe -m scripts.canonical_learning_curve --output runs/veyra3d/canonical-learning-curve-002
.\.venv-research\Scripts\python.exe -m scripts.summarize_canonical_learning_curve --suite runs/veyra3d/canonical-learning-curve-002
.\.venv-research\Scripts\python.exe -m unittest discover -s tests_research -p test_canonical_learning_curve.py -v
```

The suite refuses an existing output directory. Every experiment is capped at
200 steps. A nonfinite value, loss above max(1,10*initial), gradient norm above
100, or parameter magnitude above 100 stops the entire suite. Strict
deterministic CUDA remains enabled, with warn-only disabled, on the GTX 1660
SUPER and PyTorch 2.9.1+cu126. The four original diagnostic tests passed; no
instability occurred. Checkpoint preparation added a regression test for fresh
renders without a separately created baseline JSON.
This suite was not rerun for a variance estimate or cross-device guarantee.

During checkpoint preparation, the baseline-file dependency was made optional:
every run still recomputes the same mean-training-template baseline, and checks
the historical JSON if one exists. The recorded runs and source snapshot remain
unchanged; this fix does not alter the optimization or measurements above.

The next useful control is known procedural shape variation with exact vertex
identity, labeled explicitly as synthetic approximations, and validation split
by shape. Add camera and lighting coverage under a separate small protocol.
Further fitting of this single shape cannot demonstrate identity geometry.
No MakeHuman labels should be introduced without new independent evidence.

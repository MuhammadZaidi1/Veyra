"""Export the existing bounded learning curves; never trains or renders."""

import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.canonical_control_geometry import ROOT


def summarize(suite):
    suite = Path(suite)
    if (suite / "comparison.md").exists():
        raise ValueError("Summary exists; preserve earlier results")
    result = json.loads((suite / "summary.json").read_text())
    plan = json.loads((suite / "plan.json").read_text())
    root = ROOT / plan["dataset_root"]
    rows = json.loads((root / "control.json").read_text())["rows"]
    targets = []
    for row in rows:
        if row["split"] == "train":
            with np.load(root / row["geometry"], allow_pickle=False) as source:
                targets.append(source["vertices"])
    targets = np.stack(targets)
    diagnosis = {"training_target_min_xyz": targets.min((0, 1)).tolist(),
                 "training_target_max_xyz": targets.max((0, 1)).tolist(),
                 "max_abs_target_centroid": float(abs(targets.mean(1)).max()),
                 "normalization": "center all 468 known vertices; divide XYZ by 0.23m orthographic width",
                 "training_loss": "sum SmoothL1(beta=.02)/(B*468*3); edge/normal weights zero",
                 "report_metric": "mean Euclidean per-vertex XYZ, in image-width units",
                 "step_cap_per_experiment": 200, "identity_accuracy_established": False}
    csv_rows = []
    lines = ["# Canonical CUDA comparison", "",
             "| Run | Best step | Best fit XYZ | Best validation XYZ | Optimization / wall seconds |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for run in result["experiments"]:
        lines.append(f"| {run['name']} | {run['best_step']} | {run['best']['fit']['relative_vertex_error']:.7f} | "
                     f"{run['best']['val']['relative_vertex_error']:.7f} | {run['optimization_seconds']:.2f} / "
                     f"{run['wall_seconds_including_eval_and_checkpoints']:.2f} |")
        updates = [json.loads(line) for line in (suite / run["name"] / "updates.jsonl").read_text().splitlines()]
        values = np.array([[r["pre_update_batch_loss"], r["gradient_norm_before_clip"]] for r in updates])
        if len(updates) != run["steps"] or run["steps"] > 200 or not np.isfinite(values).all():
            raise ValueError("Invalid bounded-run evidence")
        diagnosis[run["name"]] = {"optimizer_steps": len(updates),
            "maximum_gradient_norm_before_clip": float(values[:, 1].max()), "finite_updates": True}
        for line in (suite / run["name"] / "learning_curve.jsonl").read_text().splitlines():
            record = json.loads(line)
            csv_rows.append({"experiment": run["name"], "step": record["step"],
                "fit_loss": record["fit"]["loss"], "validation_loss": record["val"]["loss"],
                "fit_xyz_error": record["fit"]["relative_vertex_error"],
                "validation_xyz_error": record["val"]["relative_vertex_error"],
                "fixed_baseline_validation_xyz_error": result["baseline"]["val"]["relative_vertex_error"]})
    lines += ["", f"Same fixed baseline in all runs: {result['baseline']['val']['relative_vertex_error']:.9f}.",
              f"Selected from validation: {result['selected_experiment']}.",
              f"Reserved test XYZ error: {result['selected_test']['relative_vertex_error']:.7f}; "
              f"baseline: {result['test_baseline']['relative_vertex_error']:.7f}.",
              "Test evaluated only after selection; no test-guided retraining. One shape and identity only."]
    with (suite / "learning_curves.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    (suite / "diagnosis.json").write_text(json.dumps(diagnosis, indent=2, allow_nan=False) + "\n")
    shutil.copyfile(Path(__file__).with_name("canonical_learning_curve.py"), suite / "runner-source.py.txt")
    (suite / "comparison.md").write_text("\n".join(lines) + "\n")
    print(str((suite / "comparison.md").resolve()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    summarize(parser.parse_args().suite)

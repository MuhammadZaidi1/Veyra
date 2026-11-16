"""Bounded, reproducible canonical learning curves, never new anatomical labels.

Keep the four-step control unchanged. This separate entry point caps every run
at 200 optimizer steps, fails closed on numerical instability, and selects a
single RGB checkpoint using validation before touching the reserved test split.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from scripts.canonical_control import validate
from scripts.canonical_control_geometry import ROOT, FRAME_WIDTH
from scripts.canonical_training import ControlData, code_signature
from veyra3d.checkpoint import cpu_state, save_checkpoint
from veyra3d.config import ExperimentConfig
from veyra3d.experiment import loader_for
from veyra3d.model import Veyra3D
from veyra3d.objectives import GeometryObjective
from veyra3d.runtime import capture_rng, environment, initialize, synchronize

SCHEMA = "veyra-canonical-learning-curve-v1"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def check_plan(plan):
    if plan.get("schema") != SCHEMA:
        raise ValueError("Wrong learning-curve schema")
    config = ExperimentConfig.from_dict(plan["execution"])
    if (not config.deterministic or config.amp or config.workers != 0
            or asdict(config.model) != {"image_size": 64, "latent_dim": 64}
            or config.batch_size != 3 or config.accumulation_steps != 1
            or config.loss.vertex != 1 or config.loss.edge != 0 or config.loss.normal != 0):
        raise ValueError("Preserve the original architecture and strict float32 vertex objective")
    names = set()
    if not 1 <= len(plan["experiments"]) <= 4:
        raise ValueError("At most four experiments per suite")
    for entry in plan["experiments"]:
        name = entry["name"]
        if not name or not all(c in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in name) or name in names:
            raise ValueError("Use unique simple experiment directory names")
        names.add(name)
        if type(entry["max_steps"]) is not int or not 1 <= entry["max_steps"] <= 200:
            raise ValueError("Each experiment is capped at 200 optimizer steps")
        if entry["initialization"] not in ("random", "training_mean_head"):
            raise ValueError("Unknown initialization")
        if entry["input"] not in ("rgb", "mean_training_image") or entry["fit"] not in ("all_train", "key3_only"):
            raise ValueError("Unknown diagnostic experiment")
    return config


def score(predicted, target, beta):
    if predicted.shape != target.shape or target.shape[1:] != (468, 3):
        raise ValueError("Expected matching Bx468x3 geometry")
    if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
        raise ValueError("Nonfinite evaluation geometry")
    delta = predicted - target
    return {"loss": float(F.smooth_l1_loss(predicted, target, beta=beta)),
            "relative_vertex_error": float(delta.norm(dim=-1).mean()),
            "axis_mae": delta.abs().mean((0, 1)).tolist(),
            "max_vertex_error": float(delta.norm(dim=-1).max())}


def check_recorded_baseline(root, value):
    """Compare a historical baseline when present; fresh renders need no extra file.

    The baseline used by every experiment is always recomputed from the same
    six training targets. The optional JSON is an old diagnostic, not an input.
    """
    path = Path(root) / "constant-baseline.json"
    if path.is_file():
        recorded = json.loads(path.read_text())["synthetic_validation_relative_vertex_error"]
        if not math.isfinite(recorded) or abs(value - recorded) > 1e-8:
            raise ValueError("Constant-template baseline changed")


def fitted_yaws(points, front_template):
    """Least-squares yaw of predicted template; a diagnostic, not a label source."""
    points = np.asarray(points, dtype=np.float64)
    template = np.asarray(front_template, dtype=np.float64)
    x, z = template[:, 0], template[:, 2]
    numerator = (x * points[..., 2] - z * points[..., 0]).sum(axis=-1)
    denominator = (x * points[..., 0] + z * points[..., 2]).sum(axis=-1)
    return np.rad2deg(np.arctan2(numerator, denominator))


def diagnose_predictions(model, groups, mean_image, front_template, config, blind=False):
    result = {}
    model.eval()
    with torch.no_grad():
        for name, group in groups.items():
            images = group["images"]
            if blind:
                images = mean_image.expand_as(images)
            predicted = model(images)
            record = score(predicted, group["targets"], config.loss.huber_beta)
            yaws = fitted_yaws(predicted.cpu().numpy(), front_template)
            record.update(fitted_yaws_degrees=yaws.tolist(),
                          expected_yaws_degrees=group["yaws"],
                          yaw_mae_degrees=float(np.abs(yaws - group["yaws"]).mean()))
            result[name] = record
    return result


def pack(samples, rows, device):
    return {"images": torch.stack([item["image"] for item in samples]).to(device),
            "targets": torch.stack([item["vertices"] for item in samples]).to(device),
            "yaws": [row["condition"]["yaw_degrees"] for row in rows]}


def experiment(entry, base_config, cached, rows, baseline, destination, signature, source_stamp):
    """One independent bounded run; scalar loss is logged before every update."""
    device = initialize(base_config)
    if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
        raise ValueError("Strict deterministic execution required")
    destination.mkdir()  # parent is a fresh suite, never overwrite another run
    write_json(destination / "config.json", {"schema": SCHEMA, "experiment": entry,
        "execution": base_config.to_dict(), "data_signature": signature,
        "source_signature": source_stamp, "synthetic_only": True, "identity_separated": False})
    model = Veyra3D(base_config.model).to(device)
    original_train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    mean_template = torch.stack([s["vertices"] for s in cached["train"]]).mean(0).to(device)
    mean_image = torch.stack([s["image"] for s in cached["train"]]).mean(0, keepdim=True).to(device)
    front_template = next(s["vertices"].numpy() for s, r in zip(cached["train"], original_train_rows)
                          if r["condition"]["yaw_degrees"] == 0)
    indices = [i for i, row in enumerate(original_train_rows)
               if entry["fit"] == "all_train" or row["condition"]["key_energy"] == 3]
    fit_samples = [cached["train"][i] for i in indices]
    if entry["initialization"] == "training_mean_head":
        with torch.no_grad():
            model.decoder[-1].weight.zero_()
            model.decoder[-1].bias.copy_(mean_template.flatten())
    groups = {"fit": pack(fit_samples, [original_train_rows[i] for i in indices], device),
              "val": pack(cached["val"], val_rows, device)}
    if entry["fit"] == "key3_only":
        held = [i for i in range(len(cached["train"])) if i not in indices]
        groups["heldout_lighting_at_seen_yaws"] = pack([cached["train"][i] for i in held],
                                                       [original_train_rows[i] for i in held], device)
    blind = entry["input"] == "mean_training_image"
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_config.learning_rate,
                                 weight_decay=base_config.weight_decay)
    objective = GeometryObjective(base_config.loss).to(device)
    machine = environment(device)
    machine["deterministic_warn_only"] = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    synchronize(device)
    started = time.perf_counter()
    best_error, best_step, best_state = math.inf, None, None
    history, step, epoch, optimization_seconds = [], 0, 0, 0.
    initial_loss = None

    def evaluate_and_save():
        nonlocal best_error, best_step, best_state, initial_loss
        evaluated = diagnose_predictions(model, groups, mean_image, front_template, base_config, blind)
        if initial_loss is None:
            initial_loss = evaluated["fit"]["loss"]
        record = {"step": step, **evaluated,
                  "baseline_val_relative_vertex_error": baseline["val"]["relative_vertex_error"],
                  "beats_baseline": evaluated["val"]["relative_vertex_error"] < baseline["val"]["relative_vertex_error"]}
        if evaluated["val"]["relative_vertex_error"] < best_error:
            best_error, best_step = evaluated["val"]["relative_vertex_error"], step
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_checkpoint(destination / "best.pt", {
                "learning_curve_schema": SCHEMA, "model": best_state,
                "optimizer": optimizer.state_dict(), "scaler": {}, "rng": capture_rng(device),
                "config": base_config.to_dict(), "fingerprint": base_config.fingerprint(),
                "data_signatures": {"canonical": signature}, "source_signature": source_stamp,
                "epoch": epoch + 1, "global_step": step, "best_validation": best_error,
                "synthetic_only": True, "quality_status": "one-template pose research only",
                "environment": machine, "experiment": entry,
                "best_validation_metric": "mean Euclidean XYZ image-width error"})
        history.append(record)
        with (destination / "learning_curve.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        return record

    try:
        evaluate_and_save()
        while step < entry["max_steps"]:
            for batch in loader_for(fit_samples, base_config, epoch, True):
                if step >= entry["max_steps"]:
                    break
                model.train()
                images = batch["image"].to(device)
                if blind:
                    images = mean_image.expand_as(images)
                targets, valid = batch["vertices"].to(device), batch["valid"].to(device)
                optimizer.zero_grad(set_to_none=True)
                synchronize(device)
                update_start = time.perf_counter()
                predicted = model(images)
                loss, _ = objective(predicted, targets, valid)
                if not torch.isfinite(loss) or float(loss) > max(1., 10 * initial_loss):
                    raise ValueError("Numerically unstable loss; suite stopped")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), base_config.grad_clip,
                                                          error_if_nonfinite=True, foreach=False)
                if not torch.isfinite(grad_norm) or float(grad_norm) > 100:
                    raise ValueError("Explosive/nonfinite gradients; suite stopped")
                optimizer.step()
                if any(not torch.isfinite(p).all() or p.abs().max() > 100 for p in model.parameters()):
                    raise ValueError("Explosive/nonfinite parameters; suite stopped")
                synchronize(device)
                optimization_seconds += time.perf_counter() - update_start
                step += 1
                with (destination / "updates.jsonl").open("a") as stream:
                    stream.write(json.dumps({"step": step, "pre_update_batch_loss": float(loss),
                        "gradient_norm_before_clip": float(grad_norm)}, allow_nan=False) + "\n")
                if step in (1, 2, 4) or step % 10 == 0 or step == entry["max_steps"]:
                    evaluate_and_save()
            epoch += 1
    except (ValueError, RuntimeError) as error:
        write_json(destination / "stopped.json", {"step": step, "reason": str(error),
                                                "remaining_experiments_cancelled": True})
        raise
    synchronize(device)
    elapsed = time.perf_counter() - started
    final_state = cpu_state(model.state_dict())
    save_checkpoint(destination / "last.pt", {"model": final_state, "experiment": entry,
        "synthetic_only": True, "learning_curve_schema": SCHEMA, "global_step": step,
        "quality_status": "final diagnostic state; use best.pt for selected checkpoint"})
    final_record = history[-1]
    model.load_state_dict(best_state)
    best_record = next(r for r in history if r["step"] == best_step)
    # No image-dependent information: evaluate every validation sample using
    # the same mean TRAINING image. This is separate from the trained blind run.
    mean_input = diagnose_predictions(model, {"val": groups["val"]}, mean_image, front_template,
                                      base_config, blind=True)["val"]
    result = {"name": entry["name"], "experiment": entry, "steps": step,
        "best_step": best_step, "best": best_record, "final": final_record,
        "initial": history[0], "baseline": baseline, "mean_input_at_best": mean_input,
        "optimization_seconds": optimization_seconds, "wall_seconds_including_eval_and_checkpoints": elapsed,
        "max_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None,
        "overfit_diagnostic": {"final_minus_best_val_error": final_record["val"]["relative_vertex_error"] - best_error,
            "best_val_minus_fit_error": best_error - best_record["fit"]["relative_vertex_error"],
            "validation_rebounds_over_10_percent_after_best": sum(r["step"] > best_step and
                r["val"]["relative_vertex_error"] > 1.1 * best_error for r in history)},
        "environment": machine, "test_used_for_selection": False}
    write_json(destination / "result.json", result)
    print(json.dumps({"experiment": entry["name"], "steps": step, "best_step": best_step,
                      "best_val_error": best_error, "seconds": elapsed}), flush=True)
    return result, model, mean_image, front_template


def run_suite(plan_path, output):
    plan = json.loads(Path(plan_path).read_text())
    config = check_plan(plan)
    root = (ROOT / plan["dataset_root"]).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Use a fresh suite directory; existing experiments are preserved")
    integrity = validate(root)
    rows = json.loads((root / "control.json").read_text())["rows"]
    device = initialize(config)
    cached = {s: list(ControlData(root, rows, s, config.model.image_size)) for s in ("train", "val")}
    train_targets = torch.stack([s["vertices"] for s in cached["train"]]).to(device)
    mean_template = train_targets.mean(0)
    baseline = {s: score(mean_template.expand(len(cached[s]), -1, -1),
                        torch.stack([v["vertices"] for v in cached[s]]).to(device), config.loss.huber_beta)
                for s in cached}
    check_recorded_baseline(root, baseline["val"]["relative_vertex_error"])
    output.mkdir(parents=True)
    (output / ".gitignore").write_text("*\n")
    write_json(output / "plan.json", plan)
    write_json(output / "integrity.json", integrity)
    write_json(output / "baseline.json", baseline)
    source_stamp = hashlib.sha256((code_signature() + Path(__file__).read_text()).encode()).hexdigest()
    write_json(output / "provenance.json", {"source_signature": source_stamp,
        "data_signature": integrity["signature"], "test_selection_policy": "choose best RGB validation checkpoint, then evaluate test exactly once",
        "stop_policy": "nonfinite values, loss > max(1,10*initial), grad norm >100 or parameter magnitude >100 stop the entire suite"})
    results, selected = [], None
    for entry in plan["experiments"]:
        result, model, mean_image, front = experiment(entry, config, cached, rows, baseline,
            output / entry["name"], integrity["signature"], source_stamp)
        results.append(result)
        if entry["input"] == "rgb" and (selected is None or result["best"]["val"]["relative_vertex_error"] < selected[0]):
            selected = (result["best"]["val"]["relative_vertex_error"], entry["name"],
                        {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, mean_image, front)
        del model
    if selected is None:
        raise ValueError("Suite needs an RGB model for held-out selection")
    # Selection is now frozen. No further training/hyperparameter search occurs.
    write_json(output / "selection.json", {"experiment": selected[1], "metric": selected[0],
                                          "source": "validation only; excludes image-blind control"})
    model = Veyra3D(config.model).to(device)
    model.load_state_dict(selected[2])
    test_rows = [r for r in rows if r["split"] == "test"]
    test_samples = list(ControlData(root, rows, "test", config.model.image_size))
    test_group = pack(test_samples, test_rows, device)
    test = diagnose_predictions(model, {"test": test_group}, selected[3], selected[4], config)["test"]
    test_baseline = score(mean_template.expand(len(test_samples), -1, -1), test_group["targets"], config.loss.huber_beta)
    summary = {"schema": SCHEMA, "baseline": baseline, "experiments": results,
        "selected_experiment": selected[1], "selected_test": test, "test_baseline": test_baseline,
        "test_evaluated_after_selection_only": True, "data_signature": integrity["signature"],
        "synthetic_only": True, "identity_count": 1, "identity_accuracy_established": False,
        "total_optimizer_steps": sum(r["steps"] for r in results)}
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/veyra3d/canonical_learning_curve.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_suite(args.plan, args.output)


if __name__ == "__main__":
    main()

"""Bounded canonical-only CUDA/CPU control; not a real-face training entry point."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch

from scripts.canonical_control import validate
from scripts.canonical_control_geometry import ROOT, SCHEMA
from veyra3d.checkpoint import load_checkpoint, save_checkpoint
from veyra3d.config import ExperimentConfig
from veyra3d.data import preprocess_rgb
from veyra3d.experiment import loader_for, run_epoch
from veyra3d.model import Veyra3D
from veyra3d.objectives import GeometryObjective
from veyra3d.runtime import capture_rng, environment, initialize, restore_rng, source_signature

import numpy as np
from PIL import Image


def load_config(path=ROOT / "configs/veyra3d/canonical.json"):
    values = json.loads(Path(path).read_text())
    if values.get("control_schema") != SCHEMA:
        raise ValueError("Not a canonical-only experiment configuration")
    config = ExperimentConfig.from_dict(values["experiment"])
    if (not config.deterministic or config.amp or config.workers != 0 or config.epochs > 2
            or config.model.image_size != 64 or config.batch_size > 3
            or config.loss.edge != 0 or config.loss.normal != 0):
        raise ValueError("Control must remain tiny, deterministic, float32 and vertex-only")
    return values, config


def code_signature():
    digest = hashlib.sha256(source_signature().encode())
    for name in ("canonical_training.py", "canonical_control.py", "canonical_control_geometry.py"):
        digest.update(Path(__file__).with_name(name).read_text().encode())
    return digest.hexdigest()


class ControlData:
    """Only construct after validate(root); never accept a MakeHuman manifest."""

    def __init__(self, root, rows, split, size):
        self.root, self.size = Path(root), size
        self.rows = [row for row in rows if row["split"] == split]
        if not self.rows:
            raise ValueError("Empty control split")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["image"]) as image:
            tensor, scale = preprocess_rgb(image, self.size)
        if scale != 1:
            raise ValueError("Control requires square images")
        with np.load(self.root / row["geometry"], allow_pickle=False) as geometry:
            return {"image": tensor, "vertices": torch.from_numpy(geometry["vertices"].copy()),
                    "valid": torch.from_numpy(geometry["valid"].copy())}


def run(root, destination, epochs=None, resume=None, device=None, config_path=None):
    """At most two epochs / four steps per continuous run. Test split is not read by SGD."""
    values, config = load_config(config_path or ROOT / "configs/veyra3d/canonical.json")
    config = replace(config, epochs=config.epochs if epochs is None else epochs, device=device or config.device)
    if type(config.epochs) is not int or not 1 <= config.epochs <= 2:
        raise ValueError("Only one or two control epochs allowed")
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()) and not resume:
        raise ValueError("Run directory is not empty; preserve previous results")
    report = validate(root)
    signature = report["signature"]
    current_code = code_signature()
    execution_device = initialize(config)
    if (not torch.are_deterministic_algorithms_enabled()
            or torch.is_deterministic_algorithms_warn_only_enabled()):
        raise ValueError("Strict deterministic algorithms required")
    model = Veyra3D(config.model).to(execution_device)
    objective = GeometryObjective(config.loss).to(execution_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    epoch_start, steps, history, best = 0, 0, [], float("inf")
    if resume:
        payload = load_checkpoint(resume)
        if payload.get("control_schema") != SCHEMA or not payload["synthetic_only"]:
            raise ValueError("Not a canonical-only checkpoint")
        if payload["fingerprint"] != config.fingerprint() or payload["source_signature"] != current_code:
            raise ValueError("Resume configuration or source changed")
        if payload["data_signatures"] != {"canonical": signature}:
            raise ValueError("Resume dataset contents changed")
        if payload["environment"]["device"] != str(execution_device):
            raise ValueError("Bitwise resume requires the same device backend")
        if payload["epoch"] >= config.epochs:
            raise ValueError("Resume must extend completed epochs")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload["scaler"])
        restore_rng(payload["rng"], execution_device)
        epoch_start, steps = payload["epoch"], payload["global_step"]
        history, best = payload["history"], payload["best_validation"]
        if destination.exists() and any(destination.iterdir()) and Path(resume).resolve().parent != destination:
            raise ValueError("Resume would overwrite a different run")
    rows = json.loads((root / "control.json").read_text())["rows"]
    datasets = {split: ControlData(root, rows, split, config.model.image_size) for split in ("train", "val")}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / ".gitignore").write_text("*\n")
    machine = environment(execution_device)
    for epoch in range(epoch_start, config.epochs):
        if execution_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(execution_device)
        training = run_epoch(model, loader_for(datasets["train"], config, epoch, True, execution_device),
                             objective, execution_device, config, optimizer, scaler)
        validation = run_epoch(model, loader_for(datasets["val"], config, device=execution_device),
                               objective, execution_device, config)
        steps += training["optimizer_steps"]
        best = min(best, validation["loss"])
        record = {"epoch": epoch + 1, "train": training, "synthetic_validation": validation}
        if execution_device.type == "cuda":
            record["peak_cuda_allocated_mb"] = torch.cuda.max_memory_allocated(execution_device) / 1024**2
        history.append(record)
        payload = {"control_schema": SCHEMA, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "scaler": scaler.state_dict(), "config": config.to_dict(), "fingerprint": config.fingerprint(),
                   "source_signature": current_code, "data_signatures": {"canonical": signature},
                   "epoch": epoch + 1, "global_step": steps, "best_validation": best,
                   "rng": capture_rng(execution_device), "history": history, "environment": machine,
                   "synthetic_only": True, "identity_count": 1,
                   "quality_status": "canonical rendering control only; no real-face or identity accuracy"}
        save_checkpoint(destination / "last.pt", payload)
    result = {"control_schema": SCHEMA, "synthetic_only": True, "identity_separated": False,
              "quality_status": payload["quality_status"], "history": history, "global_step": steps,
              "environment": machine, "dataset_signature": signature, "test_evaluated": False,
              "units": "mean Euclidean XYZ error in image-width units, not real-world mm"}
    (destination / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/veyra3d/canonical.json")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, choices=(1, 2))
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    values, _ = load_config(args.config)
    result = run(args.data or ROOT / values["dataset_root"], args.output or ROOT / values["run_dir"],
                 args.epochs, args.resume, args.device, args.config)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

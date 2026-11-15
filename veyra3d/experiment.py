"""Training, evaluation and epoch-boundary resume for explicit local datasets."""

from contextlib import nullcontext
from dataclasses import replace
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint, save_checkpoint
from .config import ExperimentConfig
from .data import make_datasets
from .model import Veyra3D
from .objectives import GeometryObjective, MetricAccumulator
from .runtime import capture_rng, environment, initialize, restore_rng, seed_worker, source_signature, synchronize


def loader_for(dataset, config, epoch=0, training=False, device=None):
    generator = torch.Generator().manual_seed(config.seed + epoch)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=training,
                      num_workers=config.workers, generator=generator, worker_init_fn=seed_worker,
                      pin_memory=device is not None and device.type == "cuda", drop_last=False)


def run_epoch(model, loader, objective, device, config, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    metrics = MetricAccumulator(config.loss)
    if training:
        optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    start, steps, samples = time.perf_counter(), 0, 0
    for index, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=device.type == "cuda")
        targets = batch["vertices"].to(device, non_blocking=device.type == "cuda")
        valid = batch["valid"].to(device, non_blocking=device.type == "cuda")
        precision = torch.autocast("cuda", dtype=torch.float16) if config.amp else nullcontext()
        with torch.set_grad_enabled(training), precision:
            predicted = model(images)
            # Geometry objectives always operate in float32, including CUDA AMP.
        with torch.set_grad_enabled(training):
            loss, stats = objective(predicted.float(), targets.float(), valid)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite geometry loss; no checkpoint will be saved for this epoch")
            if training:
                group_start = (index // config.accumulation_steps) * config.accumulation_steps
                group_size = min(config.accumulation_steps, len(loader) - group_start)
                scaler.scale(loss / group_size).backward()
                if (index + 1) % config.accumulation_steps == 0 or index + 1 == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip,
                                                   error_if_nonfinite=True, foreach=False)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    steps += 1
        metrics.update(stats)
        samples += len(images)
    synchronize(device)
    elapsed = time.perf_counter() - start
    return {**metrics.result(), "samples": samples, "seconds": elapsed,
            "samples_per_second": samples / max(elapsed, 1e-9), "optimizer_steps": steps}


def train(config, resume=None, quiet=False):
    config.validate()
    device = initialize(config)
    datasets = make_datasets(config)
    signatures = {name: dataset.signature for name, dataset in datasets.items()}
    code_signature = source_signature()
    model = Veyra3D(config.model).to(device)
    objective = GeometryObjective(config.loss).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp)
    start_epoch, global_step, best = 0, 0, math.inf
    if resume:
        payload = load_checkpoint(resume)
        if payload["fingerprint"] != config.fingerprint():
            raise ValueError("Resume config changed model/data/loss/optimization settings")
        if payload["source_signature"] != code_signature:
            raise ValueError("Research source changed since the checkpoint; start a new experiment")
        if payload["data_signatures"] != signatures:
            raise ValueError("Dataset contents changed since the checkpoint; start a new experiment")
        if payload["epoch"] >= config.epochs:
            raise ValueError("epochs must exceed the number already completed in the checkpoint")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload["scaler"])
        restore_rng(payload["rng"], device)
        start_epoch, global_step, best = payload["epoch"], payload["global_step"], payload["best_validation"]
    run_dir = Path(config.run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        metadata = run_dir / "experiment.json"
        if not resume or not metadata.is_file() or json.loads(metadata.read_text()).get("fingerprint") != config.fingerprint():
            raise ValueError("Run directory is not empty; select a fresh run_dir or a matching resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    # Protect artifacts even if a user selects a custom run folder inside Git.
    (run_dir / ".gitignore").write_text("*\n")
    metadata = {"fingerprint": config.fingerprint(), "config": config.to_dict(),
                "environment": environment(device), "data_signatures": signatures,
                "synthetic_only": config.data.kind == "synthetic", "parameters": model.parameter_count,
                "quality_status": "unvalidated research weights"}
    (run_dir / "experiment.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    history = []
    for epoch in range(start_epoch, config.epochs):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        training = run_epoch(model, loader_for(datasets["train"], config, epoch, True, device),
                             objective, device, config, optimizer, scaler)
        validation = run_epoch(model, loader_for(datasets["val"], config, device=device), objective, device,
                               replace(config, amp=False))
        global_step += training["optimizer_steps"]
        improved = validation["loss"] < best
        best = min(best, validation["loss"])
        record = {"epoch": epoch + 1, "global_step": global_step, "train": training, "val": validation,
                  "synthetic_only": config.data.kind == "synthetic"}
        if device.type == "cuda":
            record["peak_cuda_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                   "config": config.to_dict(), "fingerprint": config.fingerprint(), "data_signatures": signatures,
                   "source_signature": code_signature,
                   "epoch": epoch + 1, "global_step": global_step, "best_validation": best,
                   "rng": capture_rng(device), "synthetic_only": config.data.kind == "synthetic",
                   "quality_status": "unvalidated research weights",
                   "environment": metadata["environment"]}
        save_checkpoint(run_dir / "last.pt", payload)
        if improved:
            save_checkpoint(run_dir / "best.pt", payload)
        with (run_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        history.append(record)
        if not quiet:
            print(json.dumps(record, allow_nan=False), flush=True)
    return {"run_dir": str(run_dir), "parameters": model.parameter_count,
            "global_step": global_step, "history": history, "synthetic_only": config.data.kind == "synthetic"}


def evaluate(config, checkpoint, split="val"):
    config = replace(config, amp=False)  # Comparable float32 evaluation on every backend.
    config.validate()
    device = initialize(config)
    datasets = make_datasets(config)
    if split not in ("val", "test") or split not in datasets:
        raise ValueError("Select a configured held-out val/test split")
    payload = load_checkpoint(checkpoint)
    original = ExperimentConfig.from_dict(payload["config"])
    if original.model != config.model:
        raise ValueError("Evaluation model configuration does not match the checkpoint")
    model = Veyra3D(config.model).to(device)
    model.load_state_dict(payload["model"], strict=True)
    objective = GeometryObjective(config.loss).to(device)
    metrics = run_epoch(model, loader_for(datasets[split], config, device=device), objective, device, config)
    return {"split": split, "metrics": metrics, "device": str(device),
            "synthetic_checkpoint": bool(payload["synthetic_only"]), "synthetic_evaluation": config.data.kind == "synthetic",
            "dataset_signature": datasets[split].signature, "units": "relative canvas-width; no metric depth"}

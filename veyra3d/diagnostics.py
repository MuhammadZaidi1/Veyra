"""Short synthetic-only diagnostics, never full training or image collection."""

from dataclasses import replace
import tempfile
import time

import torch

from .checkpoint import load_checkpoint
from .config import DataConfig, ExperimentConfig, ModelConfig
from .data import SyntheticDataset
from .experiment import evaluate, train
from .model import Veyra3D
from .objectives import GeometryObjective
from .runtime import environment, initialize, synchronize


def doctor(requested="cpu"):
    config = ExperimentConfig(device=requested)
    device = initialize(config)
    model = Veyra3D(config.model)
    sample = SyntheticDataset(1, config.model.image_size, config.seed)[0]
    image = sample["image"][None]
    model.eval()
    with torch.no_grad():
        expected = model(image)
    model = model.to(device)
    with torch.no_grad():
        actual = model(image.to(device)).cpu()
    if not torch.allclose(actual, expected, rtol=1e-3, atol=1e-4):
        raise ValueError("Backend output differs substantially from the CPU reference")
    timings = []
    for iteration in range(8):
        synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            prediction = model(image.to(device))
        synchronize(device)
        if iteration >= 2:
            timings.append((time.perf_counter() - start) * 1000)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss, _ = GeometryObjective(config.loss).to(device)(
        model(image.to(device)), sample["vertices"][None].to(device), sample["valid"][None].to(device))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True, foreach=False)
    optimizer.step()
    synchronize(device)
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("Nonfinite weights after backend optimizer probe")
    return {**environment(device), "parameters": model.parameter_count, "input": [1, 3, 128, 128],
            "output": list(prediction.shape), "cpu_reference_max_abs_difference": float((actual - expected).abs().max()),
            "forward_ms_median": sorted(timings)[len(timings)//2], "backward_and_optimizer": "passed",
            "data": "procedural non-face signal only", "facial_accuracy": "not established"}


def smoke(requested="cpu"):
    # Fixed small budget; no user dataset can enter this command.
    with tempfile.TemporaryDirectory(prefix="veyra3d-smoke-") as directory:
        config = ExperimentConfig(model=ModelConfig(image_size=64, latent_dim=64),
                                  data=DataConfig(train_samples=8, val_samples=4, test_samples=4),
                                  device=requested, epochs=1, batch_size=4, run_dir=directory)
        first = train(config, quiet=True)
        checkpoint = directory + "/last.pt"
        resumed_config = replace(config, epochs=2)
        resumed = train(resumed_config, resume=checkpoint, quiet=True)
        validation = evaluate(resumed_config, checkpoint, "val")
        held_out = evaluate(resumed_config, checkpoint, "test")
        state = load_checkpoint(checkpoint)
        if state["epoch"] != 2 or resumed["global_step"] != 4:
            raise ValueError("Training smoke did not complete the expected four optimizer steps")
        return {"status": "passed", "device": requested, "parameters": resumed["parameters"],
                "optimizer_steps": resumed["global_step"], "epochs": 2,
                "initial_train_loss": first["history"][0]["train"]["loss"],
                "final_train_loss": resumed["history"][-1]["train"]["loss"],
                "validation": validation["metrics"], "held_out_test": held_out["metrics"],
                "checkpoint_roundtrip_and_resume": "passed; temporary files removed on exit",
                "data": "procedural colored signals and deformed grids, NOT faces",
                "facial_accuracy": "not established"}

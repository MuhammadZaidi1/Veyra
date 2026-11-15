"""Atomic state-dict checkpoints; tensors and primitive metadata only."""

import os
from pathlib import Path
import tempfile

import torch

from . import GEOMETRY_CONTRACT


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_state(item) for item in value)
    return value


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".checkpoint-", suffix=".tmp", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(cpu_state({"schema_version": 1, "geometry_contract": GEOMETRY_CONTRACT, **payload}), stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def load_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Checkpoint missing: {path}")
    # Only load checkpoints you created or trust; never unpickle nn.Module objects.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"schema_version", "geometry_contract", "model", "optimizer", "scaler", "config", "epoch",
                "global_step", "best_validation", "rng", "data_signatures", "fingerprint", "synthetic_only", "source_signature"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("Not a complete Veyra3D state-dict checkpoint")
    if payload["schema_version"] != 1 or payload["geometry_contract"] != GEOMETRY_CONTRACT:
        raise ValueError("Checkpoint schema or geometry contract is incompatible")
    if type(payload["synthetic_only"]) is not bool or type(payload["epoch"]) is not int or payload["epoch"] < 1:
        raise ValueError("Checkpoint provenance or epoch is invalid")
    return payload

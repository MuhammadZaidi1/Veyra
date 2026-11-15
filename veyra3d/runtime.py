"""Explicit device selection and reproducibility; no silent backend fallback."""

import os
import hashlib
import platform
import random
from pathlib import Path

import numpy as np
import PIL
import torch


def select_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable. On Windows install a compatible NVIDIA driver and CUDA-enabled PyTorch wheel; use cpu on this Mac.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable in this environment. Use cpu or verify the native Apple Silicon PyTorch installation.")
    if requested not in ("cpu", "mps", "cuda"):
        raise ValueError("Unknown device")
    return torch.device(requested)


def initialize(config):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(config.threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(config.deterministic)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = config.deterministic
    device = select_device(config.device)
    if config.amp and device.type != "cuda":
        raise ValueError("Optional AMP is CUDA-only; CPU/MPS use float32 in this foundation")
    return device


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def environment(device):
    result = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
              "torch": str(torch.__version__), "numpy": np.__version__, "pillow": PIL.__version__, "device": str(device),
              "source_signature": source_signature(), "deterministic": torch.are_deterministic_algorithms_enabled(),
              "cpu_threads": torch.get_num_threads(),
              "cuda_runtime": torch.version.cuda, "mps_available": torch.backends.mps.is_available()}
    if device.type == "cuda":
        result.update(gpu=torch.cuda.get_device_name(device),
                      capability=list(torch.cuda.get_device_capability(device)),
                      compiled_cuda_architectures=torch.cuda.get_arch_list())
    return result


def source_signature():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")) + [root.parent / "veyra" / "topology_data.py"]:
        digest.update(path.name.encode())
        digest.update(path.read_text().encode())  # Normalize platform line endings.
    return digest.hexdigest()


def capture_rng(device):
    numpy_state = np.random.get_state()
    state = {"python": random.getstate(), "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
             "torch": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    elif device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state, device):
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), *ns[2:]))
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    elif device.type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def seed_worker(_worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

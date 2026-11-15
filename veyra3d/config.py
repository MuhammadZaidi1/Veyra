"""Strict JSON experiment configuration. No datasets or weights are fetched."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path


@dataclass
class ModelConfig:
    image_size: int = 128
    latent_dim: int = 128


@dataclass
class DataConfig:
    kind: str = "synthetic"
    root: str = "data/approved"
    train_manifest: str = "train.jsonl"
    val_manifest: str = "val.jsonl"
    test_manifest: str | None = None
    train_samples: int = 16
    val_samples: int = 8
    test_samples: int = 8


@dataclass
class LossConfig:
    vertex: float = 1.0
    edge: float = 0.1
    normal: float = 0.01
    huber_beta: float = 0.02


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    seed: int = 42
    device: str = "cpu"
    threads: int = 2
    epochs: int = 2
    batch_size: int = 4
    accumulation_steps: int = 1
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    grad_clip: float = 1.0
    workers: int = 0
    deterministic: bool = True
    amp: bool = False
    run_dir: str = "runs/veyra3d/synthetic"

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise ValueError("Experiment config must be a JSON object")
        values = values.copy()
        try:
            for name, kind in (("model", ModelConfig), ("data", DataConfig), ("loss", LossConfig)):
                if name in values:
                    values[name] = kind(**values[name])
            config = cls(**values)
            config.validate()
            return config
        except (TypeError, AttributeError) as error:
            raise ValueError(f"Invalid or unknown configuration field: {error}") from error

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def validate(self):
        def integer(name, value, lower, upper):
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer between {lower} and {upper}")
        if self.device not in ("cpu", "mps", "cuda", "auto"):
            raise ValueError("device must be cpu, mps, cuda or auto")
        if type(self.model.image_size) is not int or self.model.image_size not in (64, 128, 192, 256):
            raise ValueError("image_size must be 64, 128, 192 or 256")
        integer("latent_dim", self.model.latent_dim, 32, 512)
        integer("seed", self.seed, 0, 2**31 - 1)
        for name, lower, upper in (("threads", 1, 32), ("epochs", 1, 100000),
                                  ("batch_size", 1, 512), ("accumulation_steps", 1, 128), ("workers", 0, 16)):
            integer(name, getattr(self, name), lower, upper)
        if type(self.amp) is not bool or type(self.deterministic) is not bool:
            raise ValueError("amp and deterministic must be booleans")
        for name in ("learning_rate", "weight_decay", "grad_clip"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.learning_rate <= 0 or self.grad_clip <= 0:
            raise ValueError("learning_rate and grad_clip must be positive")
        if self.data.kind not in ("synthetic", "manifest"):
            raise ValueError("data.kind must be synthetic or manifest")
        for name in ("train_samples", "val_samples", "test_samples"):
            integer(name, getattr(self.data, name), 1, 10000000)
        for name in ("vertex", "edge", "normal", "huber_beta"):
            value = getattr(self.loss, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"loss.{name} must be finite and nonnegative")
        if self.loss.vertex <= 0 or self.loss.huber_beta <= 0:
            raise ValueError("The supervised vertex loss and Huber beta must be positive")
        for value in (self.run_dir, self.data.root, self.data.train_manifest, self.data.val_manifest):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Dataset and run paths must be nonempty strings")
        if self.data.test_manifest is not None and (not isinstance(self.data.test_manifest, str) or not self.data.test_manifest.strip()):
            raise ValueError("test_manifest must be a path string or null")
        return self

    def to_dict(self):
        return asdict(self)

    def fingerprint(self):
        values = self.to_dict()
        # These execution changes are allowed at an epoch-boundary resume.
        for key in ("epochs", "run_dir", "device", "threads", "workers"):
            values.pop(key)
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()

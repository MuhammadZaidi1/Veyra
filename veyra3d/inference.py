"""Single-crop, memory-only inference API; not wired into the live prototype."""

import torch

from .checkpoint import load_checkpoint
from .config import ExperimentConfig
from .data import preprocess_rgb
from .model import Veyra3D
from .runtime import select_device


class Predictor:
    def __init__(self, checkpoint, device="cpu"):
        payload = load_checkpoint(checkpoint)
        self.config = ExperimentConfig.from_dict(payload["config"])
        self.synthetic_only = bool(payload["synthetic_only"])
        self.device = select_device(device)
        self.model = Veyra3D(self.config.model).to(self.device).eval()
        self.model.load_state_dict(payload["model"], strict=True)

    @torch.inference_mode()
    def predict(self, rgb_crop):
        """PIL image → float32 [468,3] array in relative square-canvas units.

        No crop detector, pose/metric-depth recovery, export or camera capture.
        A synthetic-only checkpoint has no established facial accuracy.
        """
        images, _ = preprocess_rgb(rgb_crop, self.config.model.image_size)
        result = self.model(images[None].to(self.device))[0]
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite custom-model output")
        return result.cpu().numpy().copy()

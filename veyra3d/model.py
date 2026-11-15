"""Small spatial encoder + point decoder, randomly initialized; no downloads."""

import torch
from torch import nn

from .config import ModelConfig

VERTEX_COUNT = 468


class SeparableBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1, groups=in_channels, bias=False),
            nn.GroupNorm(8, in_channels), nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(8, out_channels), nn.SiLU(),
        )


class Veyra3D(nn.Module):
    """B x 3 x S x S float32 RGB [-1,1] → B x 468 x 3 centered XYZ.

    XYZ uses relative crop/canvas-width units; +x right, +y up, +z toward the
    camera. There is no metric depth, identity embedding or confidence claim.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        if (type(self.config.image_size) is not int or self.config.image_size not in (64, 128, 192, 256)
                or type(self.config.latent_dim) is not int or not 32 <= self.config.latent_dim <= 512):
            raise ValueError("Unsupported model image size or latent dimension")
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 16), nn.SiLU(),
            SeparableBlock(16, 24), SeparableBlock(24, 32),
            SeparableBlock(32, 64), SeparableBlock(64, 96),
            # Five stride-2 stages give S/32; supported sizes divide into 2x2 bins.
            # Fixed pooling preserves these bins and supports deterministic CUDA backward.
            nn.AvgPool2d(self.config.image_size // 64), nn.Flatten(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(96 * 2 * 2, self.config.latent_dim), nn.SiLU(),
            nn.Linear(self.config.latent_dim, self.config.latent_dim), nn.SiLU(),
            nn.Linear(self.config.latent_dim, VERTEX_COUNT * 3),
        )

    def forward(self, images):
        size = self.config.image_size
        if images.ndim != 4 or images.shape[1:] != (3, size, size) or not len(images):
            raise ValueError(f"Expected a nonempty B x 3 x {size} x {size} RGB tensor")
        if images.dtype != torch.float32:
            raise ValueError("Input images must be float32, normalized to [-1, 1]")
        vertices = self.decoder(self.encoder(images)).float().reshape(-1, VERTEX_COUNT, 3)
        return vertices - vertices.mean(dim=1, keepdim=True)

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

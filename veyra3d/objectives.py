"""Masked supervised geometry objectives and unitless evaluation statistics."""

import torch
from torch import nn
from torch.nn import functional as F

from veyra.topology_data import TRIANGLE_INDICES
from .config import LossConfig


class GeometryObjective(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or LossConfig()
        triangles = torch.tensor(TRIANGLE_INDICES, dtype=torch.long)
        edges = torch.cat((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
        edges = torch.unique(edges.sort(dim=1).values, dim=0)
        self.register_buffer("triangles", triangles, persistent=False)
        self.register_buffer("edges", edges, persistent=False)

    def forward(self, predicted, target, valid):
        if (predicted.ndim != 3 or predicted.shape[1:] != (468, 3) or target.shape != predicted.shape
                or valid.shape != predicted.shape[:2] or valid.dtype != torch.bool):
            raise ValueError("Expected Bx468x3 predictions/targets and a boolean Bx468 validity mask")
        if not torch.isfinite(predicted).all() or not torch.isfinite(target[valid]).all():
            raise ValueError("Predictions and supervised target vertices must be finite")
        # Replacing masked NaNs before arithmetic avoids NaN*0 leaking into gradients.
        target = torch.where(valid[..., None], target, torch.zeros_like(target))
        vertex_values = F.smooth_l1_loss(predicted, target, beta=self.config.huber_beta, reduction="none")
        vertex_sum = torch.where(valid[..., None], vertex_values, 0).sum()
        vertex_count = valid.sum() * 3
        a, b = self.edges.unbind(dim=1)
        edge_valid = valid[:, a] & valid[:, b]
        predicted_edges = torch.linalg.vector_norm(predicted[:, a] - predicted[:, b], dim=-1)
        target_edges = torch.linalg.vector_norm(target[:, a] - target[:, b], dim=-1)
        edge_values = F.smooth_l1_loss(predicted_edges, target_edges, beta=self.config.huber_beta, reduction="none")
        edge_sum = torch.where(edge_valid, edge_values, 0).sum()
        edge_count = edge_valid.sum()
        p, t = predicted[:, self.triangles], target[:, self.triangles]
        p_normal = torch.linalg.cross(p[:, :, 1] - p[:, :, 0], p[:, :, 2] - p[:, :, 0], dim=-1)
        t_normal = torch.linalg.cross(t[:, :, 1] - t[:, :, 0], t[:, :, 2] - t[:, :, 0], dim=-1)
        normal_valid = valid[:, self.triangles].all(dim=-1) & (torch.linalg.vector_norm(t_normal, dim=-1) > 1e-6)
        cosine = (F.normalize(p_normal, dim=-1, eps=1e-6) * F.normalize(t_normal, dim=-1, eps=1e-6)).sum(dim=-1)
        normal_sum = torch.where(normal_valid, 1 - cosine.clamp(-1, 1), 0).sum()
        normal_count = normal_valid.sum()
        means = (vertex_sum / vertex_count.clamp_min(1), edge_sum / edge_count.clamp_min(1),
                 normal_sum / normal_count.clamp_min(1))
        total = self.config.vertex * means[0] + self.config.edge * means[1] + self.config.normal * means[2]
        distance = torch.linalg.vector_norm(predicted - target, dim=-1)
        stats = {
            "vertex_sum": vertex_sum.detach(), "vertex_count": vertex_count.detach(),
            "edge_sum": edge_sum.detach(), "edge_count": edge_count.detach(),
            "normal_sum": normal_sum.detach(), "normal_count": normal_count.detach(),
            "distance_sum": torch.where(valid, distance, 0).sum().detach(), "valid_vertices": valid.sum().detach(),
        }
        return total, stats


class MetricAccumulator:
    def __init__(self, config):
        self.config = config
        self.sums = {}

    def update(self, stats):
        names = list(stats)
        values = torch.stack([stats[name].float() for name in names]).cpu().tolist()
        for name, value in zip(names, values):
            self.sums[name] = self.sums.get(name, 0.0) + value

    def result(self):
        means = {name: self.sums.get(name + "_sum", 0) / max(1, self.sums.get(name + "_count", 0))
                 for name in ("vertex", "edge", "normal")}
        return {"loss": sum(getattr(self.config, name) * means[name] for name in means),
                **{name + "_loss": value for name, value in means.items()},
                "relative_vertex_error": self.sums.get("distance_sum", 0) / max(1, self.sums.get("valid_vertices", 0)),
                "supervised_vertices": int(self.sums.get("valid_vertices", 0))}

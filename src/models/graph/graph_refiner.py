from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.models.graph.knn_graph import build_exact_knn_graph


class GraphRefiner(nn.Module):
    """One-modality train-mode exact-kNN graph refiner."""

    def __init__(self, d_z: int, graph_k: int, beta_tree_injection: float = 1.0) -> None:
        super().__init__()
        if d_z <= 0:
            raise ValueError("d_z must be positive")
        if graph_k <= 0:
            raise ValueError("graph_k must be positive")
        if beta_tree_injection < 0.0:
            raise ValueError("beta_tree_injection must be non-negative")
        self.d_z = int(d_z)
        self.graph_k = int(graph_k)
        self.beta_tree_injection = float(beta_tree_injection)
        self.layer_norm = nn.LayerNorm(self.d_z)
        self.graph_linear = nn.Linear(self.d_z, self.d_z)
        self.output_norm = nn.LayerNorm(self.d_z)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.graph_linear.weight, gain=0.5)
        nn.init.zeros_(self.graph_linear.bias)

    def forward(self, Z: torch.Tensor, Y: torch.Tensor) -> dict[str, Any]:
        _check_same_shape(Z, Y, self.d_z)
        Fused = self.layer_norm(Z + self.beta_tree_injection * Y)
        _check_2d_finite(Fused, "F")
        graph, diagnostics = build_exact_knn_graph(Fused.detach(), self.graph_k)
        propagated = graph @ Fused
        hidden = self.output_norm(self.graph_linear(propagated))
        _check_2d_finite(hidden, "graph_hidden")
        diagnostics.update(
            {
                "beta_tree_injection": self.beta_tree_injection,
                "graph_hidden_shape": list(hidden.shape),
                "ann_used": False,
            }
        )
        return {
            "F": Fused,
            "graph": graph,
            "graph_hidden": hidden,
            "diagnostics": diagnostics,
        }


def _check_same_shape(Z: torch.Tensor, Y: torch.Tensor, d_z: int) -> None:
    _check_2d_finite(Z, "Z")
    _check_2d_finite(Y, "Y")
    if Z.shape != Y.shape:
        raise RuntimeError(f"Z and Y shapes must match, got {tuple(Z.shape)} and {tuple(Y.shape)}")
    if Z.shape[1] != d_z:
        raise RuntimeError(f"expected d_z={d_z}, got {Z.shape[1]}")


def _check_2d_finite(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 2:
        raise RuntimeError(f"{name} must be 2D, got shape {tuple(tensor.shape)}")
    if tensor.numel() == 0:
        raise RuntimeError(f"{name} must be non-empty")
    if not torch.is_floating_point(tensor):
        raise RuntimeError(f"{name} must be a floating point tensor")
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains NaN or Inf")

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def build_exact_knn_graph(F_in: torch.Tensor, k: int) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build a symmetric normalized exact kNN graph from detached features."""

    _check_2d_finite(F_in, "F")
    sample_count = int(F_in.shape[0])
    if k <= 0:
        raise ValueError("graph k must be positive")
    if k >= sample_count:
        raise ValueError(f"graph k must be smaller than sample count, got k={k}, N={sample_count}")

    features = F.normalize(F_in.detach(), p=2, dim=1, eps=1e-12)
    cosine = features @ features.transpose(0, 1)
    if not torch.isfinite(cosine).all():
        raise RuntimeError("kNN cosine matrix contains NaN or Inf")
    cosine = cosine.masked_fill(torch.eye(sample_count, dtype=torch.bool, device=F_in.device), float("-inf"))
    values, indices = torch.topk(cosine, k=k, dim=1, largest=True, sorted=False)
    if not torch.isfinite(values).all():
        raise RuntimeError("kNN top-k produced NaN or Inf")

    weights = (1.0 + values).mul(0.5)
    if weights.min().item() < -1e-6 or weights.max().item() > 1.0 + 1e-6:
        raise RuntimeError("kNN edge weights are outside [0, 1]")
    adjacency = torch.zeros(sample_count, sample_count, dtype=F_in.dtype, device=F_in.device)
    adjacency.scatter_(1, indices, weights.to(dtype=F_in.dtype))
    adjacency = torch.maximum(adjacency, adjacency.transpose(0, 1))
    adjacency_finite = bool(torch.isfinite(adjacency).all().detach().item())
    if not adjacency_finite:
        raise RuntimeError("graph adjacency contains NaN or Inf")
    adjacency_with_loop = adjacency + torch.eye(sample_count, dtype=F_in.dtype, device=F_in.device)
    degree = adjacency_with_loop.sum(dim=1)
    if not torch.isfinite(degree).all():
        raise RuntimeError("graph degree contains NaN or Inf")
    isolated_node_count = int(torch.sum(degree <= 0.0).detach().item())
    if isolated_node_count:
        raise RuntimeError("graph contains an isolated node after self-loop")
    hubness = _degree_hubness_diagnostics(degree)
    inv_sqrt_degree = torch.rsqrt(degree)
    normalized = inv_sqrt_degree.unsqueeze(1) * adjacency_with_loop * inv_sqrt_degree.unsqueeze(0)
    normalized_graph_finite = bool(torch.isfinite(normalized).all().detach().item())
    if not normalized_graph_finite:
        raise RuntimeError("normalized graph contains NaN or Inf")
    diagnostics = {
        "k": int(k),
        "node_count": sample_count,
        "degree_min": float(degree.min().detach().item()),
        "degree_max": float(degree.max().detach().item()),
        "degree_mean": float(degree.mean().detach().item()),
        "degree_p95": hubness["degree_p95"],
        "degree_p99": hubness["degree_p99"],
        "degree_max_over_mean": hubness["degree_max_over_mean"],
        "degree_gini": hubness["degree_gini"],
        "graph_hubness_risk": hubness["graph_hubness_risk"],
        "edge_weight_min": float(weights.min().detach().item()),
        "edge_weight_max": float(weights.max().detach().item()),
        "adjacency_finite": adjacency_finite,
        "normalized_graph_finite": normalized_graph_finite,
        "isolated_node_count": isolated_node_count,
        "no_isolated_train_node_after_self_loop": isolated_node_count == 0,
        "self_loop_added": True,
        "exact_knn": True,
        "identity_fallback_used": False,
    }
    return normalized, diagnostics


def _degree_hubness_diagnostics(degree: torch.Tensor) -> dict[str, float | str]:
    sorted_degree = torch.sort(degree.detach().to(dtype=torch.float64).flatten()).values
    sample_count = int(sorted_degree.numel())
    if sample_count <= 0:
        raise RuntimeError("degree must be non-empty")
    degree_mean = torch.mean(sorted_degree)
    if degree_mean <= 0.0 or not torch.isfinite(degree_mean):
        raise RuntimeError("degree mean must be positive and finite")
    degree_max_over_mean = float((sorted_degree[-1] / degree_mean).detach().item())
    return {
        "degree_p95": _nearest_rank_percentile(sorted_degree, 0.95),
        "degree_p99": _nearest_rank_percentile(sorted_degree, 0.99),
        "degree_max_over_mean": degree_max_over_mean,
        "degree_gini": _gini(sorted_degree),
        "graph_hubness_risk": _hubness_risk(degree_max_over_mean),
    }


def _nearest_rank_percentile(sorted_values: torch.Tensor, percentile: float) -> float:
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must be in [0, 1]")
    sample_count = int(sorted_values.numel())
    index = int(torch.ceil(torch.tensor(percentile * sample_count, dtype=torch.float64)).item()) - 1
    index = max(0, min(sample_count - 1, index))
    return float(sorted_values[index].detach().item())


def _gini(sorted_values: torch.Tensor) -> float:
    total = torch.sum(sorted_values)
    if total <= 0.0 or not torch.isfinite(total):
        raise RuntimeError("gini requires positive finite total")
    sample_count = int(sorted_values.numel())
    rank = torch.arange(1, sample_count + 1, dtype=sorted_values.dtype, device=sorted_values.device)
    gini = (2.0 * torch.sum(rank * sorted_values) / (sample_count * total)) - ((sample_count + 1.0) / sample_count)
    return float(torch.clamp(gini, min=0.0).detach().item())


def _hubness_risk(degree_max_over_mean: float) -> str:
    if degree_max_over_mean < 10.0:
        return "low"
    if degree_max_over_mean < 30.0:
        return "medium"
    return "high"


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

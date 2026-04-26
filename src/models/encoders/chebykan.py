from __future__ import annotations

import torch
from torch import nn


class ChebyKAN(nn.Module):
    """Chebyshev KAN-style expansion for one modality."""

    def __init__(self, input_dim: int = 512, output_dim: int = 256, order: int = 4) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if order < 0:
            raise ValueError("order must be non-negative")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.order = int(order)
        self.projection = nn.Linear(self.input_dim * (self.order + 1), self.output_dim)
        self.output_norm = nn.LayerNorm(self.output_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.projection.weight)
        if self.order >= 1:
            start = self.input_dim
            end = 2 * self.input_dim
            nn.init.xavier_uniform_(self.projection.weight[:, start:end], gain=1.0)
        for degree in range(2, self.order + 1):
            start = degree * self.input_dim
            end = (degree + 1) * self.input_dim
            nn.init.normal_(self.projection.weight[:, start:end], mean=0.0, std=0.005)
        nn.init.zeros_(self.projection.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        _check_2d_finite(X, "X")
        if X.shape[1] != self.input_dim:
            raise RuntimeError(f"ChebyKAN expected input dim {self.input_dim}, got {X.shape[1]}")
        X_clamped = X.clamp(min=-1.0, max=1.0)
        basis = _chebyshev_basis(X_clamped, self.order)
        Z = self.output_norm(self.projection(torch.cat(basis, dim=1)))
        _check_2d_finite(Z, "Z")
        if Z.shape != (X.shape[0], self.output_dim):
            raise RuntimeError(f"ChebyKAN produced invalid shape {tuple(Z.shape)}")
        return Z


def _chebyshev_basis(X: torch.Tensor, order: int) -> list[torch.Tensor]:
    terms = [torch.ones_like(X)]
    if order == 0:
        return terms
    terms.append(X)
    for degree in range(2, order + 1):
        terms.append(2.0 * X * terms[degree - 1] - terms[degree - 2])
    return terms


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

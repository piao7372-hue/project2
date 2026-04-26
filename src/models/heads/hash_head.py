from __future__ import annotations

from typing import Any

import torch
from torch import nn


class HashHead(nn.Module):
    """Continuous hash projection plus int8 sign binarization."""

    def __init__(self, input_dim: int, bit: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if bit <= 0:
            raise ValueError("bit must be positive")
        self.input_dim = int(input_dim)
        self.bit = int(bit)
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.projection = nn.Linear(self.input_dim, self.bit)
        self.logit_norm = nn.LayerNorm(self.bit)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.projection.weight, gain=0.5)
        nn.init.zeros_(self.projection.bias)

    def forward(self, X: torch.Tensor) -> dict[str, Any]:
        _check_2d_finite(X, "hash_head_input")
        if X.shape[1] != self.input_dim:
            raise RuntimeError(f"HashHead expected input dim {self.input_dim}, got {X.shape[1]}")
        logits = self.logit_norm(self.projection(self.input_norm(X)))
        logits = logits - logits.mean(dim=0, keepdim=True)
        H = torch.tanh(0.5 * logits)
        _check_2d_finite(H, "H")
        if H.shape != (X.shape[0], self.bit):
            raise RuntimeError(f"HashHead produced invalid H shape {tuple(H.shape)}")
        B = torch.where(H >= 0.0, torch.ones_like(H), -torch.ones_like(H)).to(torch.int8)
        if B.shape != H.shape:
            raise RuntimeError("B shape does not match H shape")
        if not torch.equal(B.to(torch.int16), torch.where(H >= 0.0, 1, -1).to(torch.int16)):
            raise RuntimeError("B does not match sign rule")
        return {"H": H, "B": B}


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

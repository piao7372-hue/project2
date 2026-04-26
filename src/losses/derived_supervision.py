from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DerivedSupervision:
    Q_I: torch.Tensor
    Q_T: torch.Tensor
    S_II_star: torch.Tensor
    S_TT_star: torch.Tensor


def row_l2_normalize(matrix: torch.Tensor, eps: float) -> torch.Tensor:
    _ensure_matrix("matrix", matrix)
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    row_norms = torch.linalg.vector_norm(matrix, ord=2, dim=1, keepdim=True)
    zero_rows = row_norms.squeeze(1) <= eps
    if bool(zero_rows.any().item()):
        first_bad = int(torch.nonzero(zero_rows, as_tuple=False)[0].item())
        raise ValueError(f"matrix row L2 norm must be positive; first invalid row={first_bad}")
    return matrix / (row_norms + eps)


def derive_same_modal_targets(S: torch.Tensor, eps: float) -> DerivedSupervision:
    _ensure_square_supervision(S)
    Q_I = row_l2_normalize(S, eps)
    Q_T = row_l2_normalize(S.transpose(0, 1), eps)
    S_II_star = Q_I @ Q_I.transpose(0, 1)
    S_TT_star = Q_T @ Q_T.transpose(0, 1)
    _ensure_derived_target("S_II_star", S_II_star, S.shape[0])
    _ensure_derived_target("S_TT_star", S_TT_star, S.shape[0])
    return DerivedSupervision(Q_I=Q_I, Q_T=Q_T, S_II_star=S_II_star, S_TT_star=S_TT_star)


def _ensure_matrix(name: str, matrix: torch.Tensor) -> None:
    if not isinstance(matrix, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be rank-2, got shape={list(matrix.shape)}")
    if not bool(torch.isfinite(matrix).all().item()):
        raise ValueError(f"{name} must be finite")


def _ensure_square_supervision(S: torch.Tensor) -> None:
    _ensure_matrix("S", S)
    if S.shape[0] != S.shape[1]:
        raise ValueError(f"S must be square [N,N], got shape={list(S.shape)}")
    if S.numel() == 0:
        raise ValueError("S must be non-empty")
    min_value = float(S.amin().item())
    max_value = float(S.amax().item())
    if min_value < -1e-6 or max_value > 1.0 + 1e-6:
        raise ValueError(f"S values must be in [0,1], got min={min_value}, max={max_value}")


def _ensure_derived_target(name: str, target: torch.Tensor, expected_n: int) -> None:
    if target.shape != (expected_n, expected_n):
        raise ValueError(f"{name} must have shape [{expected_n},{expected_n}], got {list(target.shape)}")
    if not bool(torch.isfinite(target).all().item()):
        raise ValueError(f"{name} must be finite")
    min_value = float(target.amin().item())
    max_value = float(target.amax().item())
    if min_value < -1e-5 or max_value > 1.0 + 1e-5:
        raise ValueError(f"{name} values must be in [0,1], got min={min_value}, max={max_value}")

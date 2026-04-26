from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class RelationPredictions:
    P_IT: torch.Tensor
    P_II: torch.Tensor
    P_TT: torch.Tensor


@dataclass(frozen=True)
class RelationLossComponents:
    L_IT: torch.Tensor
    L_II: torch.Tensor
    L_TT: torch.Tensor
    L_sem: torch.Tensor


@dataclass(frozen=True)
class HashLossComponents:
    L_IT: torch.Tensor
    L_II: torch.Tensor
    L_TT: torch.Tensor
    L_sem: torch.Tensor
    L_pair: torch.Tensor
    L_q: torch.Tensor
    L_bal: torch.Tensor
    L_total: torch.Tensor


def normalize_hash_rows(H: torch.Tensor, eps: float) -> torch.Tensor:
    _ensure_hash_matrix("H", H)
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    row_norms = torch.linalg.vector_norm(H, ord=2, dim=1, keepdim=True)
    zero_rows = row_norms.squeeze(1) <= eps
    if bool(zero_rows.any().item()):
        first_bad = int(torch.nonzero(zero_rows, as_tuple=False)[0].item())
        raise ValueError(f"H row L2 norm must be positive; first invalid row={first_bad}")
    return H / (row_norms + eps)


def compute_relation_predictions_dense(H_I: torch.Tensor, H_T: torch.Tensor, eps: float = 1e-8) -> RelationPredictions:
    _ensure_hash_pair(H_I, H_T)
    H_I_hat = normalize_hash_rows(H_I, eps)
    H_T_hat = normalize_hash_rows(H_T, eps)
    P_IT = (1.0 + H_I_hat @ H_T_hat.transpose(0, 1)) / 2.0
    P_II = (1.0 + H_I_hat @ H_I_hat.transpose(0, 1)) / 2.0
    P_TT = (1.0 + H_T_hat @ H_T_hat.transpose(0, 1)) / 2.0
    return RelationPredictions(P_IT=P_IT, P_II=P_II, P_TT=P_TT)


def compute_relation_losses_dense(
    H_I: torch.Tensor,
    H_T: torch.Tensor,
    S: torch.Tensor,
    S_II_star: torch.Tensor,
    S_TT_star: torch.Tensor,
    beta_relation_weight: float,
    alpha_intra_topology: float,
    eps: float,
) -> RelationLossComponents:
    _ensure_hash_pair(H_I, H_T)
    n = H_I.shape[0]
    _ensure_target("S", S, n)
    _ensure_target("S_II_star", S_II_star, n)
    _ensure_target("S_TT_star", S_TT_star, n)
    _ensure_nonnegative("beta_relation_weight", beta_relation_weight)
    _ensure_nonnegative("alpha_intra_topology", alpha_intra_topology)
    predictions = compute_relation_predictions_dense(H_I, H_T, eps=eps)
    L_IT = _weighted_mse_relation(predictions.P_IT, S, beta_relation_weight)
    L_II = _weighted_mse_relation(predictions.P_II, S_II_star, beta_relation_weight)
    L_TT = _weighted_mse_relation(predictions.P_TT, S_TT_star, beta_relation_weight)
    L_sem = L_IT + alpha_intra_topology / 2.0 * (L_II + L_TT)
    return RelationLossComponents(L_IT=L_IT, L_II=L_II, L_TT=L_TT, L_sem=L_sem)


def compute_relation_losses_blockwise(
    H_I: torch.Tensor,
    H_T: torch.Tensor,
    S: torch.Tensor,
    Q_I: torch.Tensor,
    Q_T: torch.Tensor,
    beta_relation_weight: float,
    alpha_intra_topology: float,
    block_size: int,
    eps: float,
) -> RelationLossComponents:
    _ensure_hash_pair(H_I, H_T)
    n = H_I.shape[0]
    _ensure_target("S", S, n)
    _ensure_target("Q_I", Q_I, n)
    _ensure_target("Q_T", Q_T, n)
    _ensure_nonnegative("beta_relation_weight", beta_relation_weight)
    _ensure_nonnegative("alpha_intra_topology", alpha_intra_topology)
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    H_I_hat = normalize_hash_rows(H_I, eps)
    H_T_hat = normalize_hash_rows(H_T, eps)
    num_it = H_I.new_zeros(())
    den_it = H_I.new_zeros(())
    num_ii = H_I.new_zeros(())
    den_ii = H_I.new_zeros(())
    num_tt = H_I.new_zeros(())
    den_tt = H_I.new_zeros(())

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        S_block = S[start:end, :]
        P_IT_block = (1.0 + H_I_hat[start:end, :] @ H_T_hat.transpose(0, 1)) / 2.0
        num_it, den_it = _accumulate_weighted_block(num_it, den_it, P_IT_block, S_block, beta_relation_weight)

        S_II_block = Q_I[start:end, :] @ Q_I.transpose(0, 1)
        P_II_block = (1.0 + H_I_hat[start:end, :] @ H_I_hat.transpose(0, 1)) / 2.0
        num_ii, den_ii = _accumulate_weighted_block(num_ii, den_ii, P_II_block, S_II_block, beta_relation_weight)

        S_TT_block = Q_T[start:end, :] @ Q_T.transpose(0, 1)
        P_TT_block = (1.0 + H_T_hat[start:end, :] @ H_T_hat.transpose(0, 1)) / 2.0
        num_tt, den_tt = _accumulate_weighted_block(num_tt, den_tt, P_TT_block, S_TT_block, beta_relation_weight)

    L_IT = _safe_divide(num_it, den_it, "W_IT")
    L_II = _safe_divide(num_ii, den_ii, "W_II")
    L_TT = _safe_divide(num_tt, den_tt, "W_TT")
    L_sem = L_IT + alpha_intra_topology / 2.0 * (L_II + L_TT)
    return RelationLossComponents(L_IT=L_IT, L_II=L_II, L_TT=L_TT, L_sem=L_sem)


def compute_pair_loss(H_I: torch.Tensor, H_T: torch.Tensor, eps: float) -> torch.Tensor:
    _ensure_hash_pair(H_I, H_T)
    H_I_hat = normalize_hash_rows(H_I, eps)
    H_T_hat = normalize_hash_rows(H_T, eps)
    return torch.sum((H_I_hat - H_T_hat).pow(2)) / H_I.shape[0]


def compute_quantization_loss(H_I: torch.Tensor, H_T: torch.Tensor) -> torch.Tensor:
    _ensure_hash_pair(H_I, H_T)
    n, k = H_I.shape
    return (torch.sum((H_I.pow(2) - 1.0).pow(2)) + torch.sum((H_T.pow(2) - 1.0).pow(2))) / (n * k)


def compute_balance_loss(H_I: torch.Tensor, H_T: torch.Tensor) -> torch.Tensor:
    _ensure_hash_pair(H_I, H_T)
    k = H_I.shape[1]
    return torch.sum(H_I.mean(dim=0).pow(2)) / k + torch.sum(H_T.mean(dim=0).pow(2)) / k


def compute_total_hash_loss(
    H_I: torch.Tensor,
    H_T: torch.Tensor,
    S: torch.Tensor,
    beta_relation_weight: float,
    alpha_intra_topology: float,
    lambda_sem_total: float,
    lambda_pair_total: float,
    lambda_q_total: float,
    lambda_bal_total: float,
    eps: float,
    relation_mode: str,
    block_size: Optional[int] = None,
    S_II_star: Optional[torch.Tensor] = None,
    S_TT_star: Optional[torch.Tensor] = None,
    Q_I: Optional[torch.Tensor] = None,
    Q_T: Optional[torch.Tensor] = None,
) -> HashLossComponents:
    _ensure_nonnegative("lambda_sem_total", lambda_sem_total)
    _ensure_nonnegative("lambda_pair_total", lambda_pair_total)
    _ensure_nonnegative("lambda_q_total", lambda_q_total)
    _ensure_nonnegative("lambda_bal_total", lambda_bal_total)

    if relation_mode == "dense":
        if S_II_star is None or S_TT_star is None:
            raise ValueError("dense relation_mode requires S_II_star and S_TT_star")
        relation = compute_relation_losses_dense(
            H_I=H_I,
            H_T=H_T,
            S=S,
            S_II_star=S_II_star,
            S_TT_star=S_TT_star,
            beta_relation_weight=beta_relation_weight,
            alpha_intra_topology=alpha_intra_topology,
            eps=eps,
        )
    elif relation_mode == "blockwise":
        if Q_I is None or Q_T is None:
            raise ValueError("blockwise relation_mode requires Q_I and Q_T")
        if block_size is None:
            raise ValueError("blockwise relation_mode requires block_size")
        relation = compute_relation_losses_blockwise(
            H_I=H_I,
            H_T=H_T,
            S=S,
            Q_I=Q_I,
            Q_T=Q_T,
            beta_relation_weight=beta_relation_weight,
            alpha_intra_topology=alpha_intra_topology,
            block_size=block_size,
            eps=eps,
        )
    else:
        raise ValueError(f"unknown relation_mode={relation_mode}")

    L_pair = compute_pair_loss(H_I, H_T, eps)
    L_q = compute_quantization_loss(H_I, H_T)
    L_bal = compute_balance_loss(H_I, H_T)
    L_total = (
        lambda_sem_total * relation.L_sem
        + lambda_pair_total * L_pair
        + lambda_q_total * L_q
        + lambda_bal_total * L_bal
    )
    return HashLossComponents(
        L_IT=relation.L_IT,
        L_II=relation.L_II,
        L_TT=relation.L_TT,
        L_sem=relation.L_sem,
        L_pair=L_pair,
        L_q=L_q,
        L_bal=L_bal,
        L_total=L_total,
    )


def _weighted_mse_relation(prediction: torch.Tensor, target: torch.Tensor, beta_relation_weight: float) -> torch.Tensor:
    weights = 1.0 + beta_relation_weight * target
    return _safe_divide(torch.sum(weights * (prediction - target).pow(2)), torch.sum(weights), "relation weights")


def _accumulate_weighted_block(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    prediction_block: torch.Tensor,
    target_block: torch.Tensor,
    beta_relation_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = 1.0 + beta_relation_weight * target_block
    numerator = numerator + torch.sum(weights * (prediction_block - target_block).pow(2))
    denominator = denominator + torch.sum(weights)
    return numerator, denominator


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor, name: str) -> torch.Tensor:
    if not bool(torch.isfinite(denominator).all().item()):
        raise ValueError(f"{name} denominator must be finite")
    if float(denominator.item()) <= 0.0:
        raise ValueError(f"{name} denominator must be positive")
    return numerator / denominator


def _ensure_hash_pair(H_I: torch.Tensor, H_T: torch.Tensor) -> None:
    _ensure_hash_matrix("H_I", H_I)
    _ensure_hash_matrix("H_T", H_T)
    if H_I.shape != H_T.shape:
        raise ValueError(f"H_I and H_T shapes must match, got {list(H_I.shape)} and {list(H_T.shape)}")


def _ensure_hash_matrix(name: str, H: torch.Tensor) -> None:
    if not isinstance(H, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if H.ndim != 2:
        raise ValueError(f"{name} must be rank-2 [N,K], got shape={list(H.shape)}")
    if H.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if not bool(torch.isfinite(H).all().item()):
        raise ValueError(f"{name} must be finite")


def _ensure_target(name: str, target: torch.Tensor, expected_n: int) -> None:
    if not isinstance(target, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if target.shape != (expected_n, expected_n):
        raise ValueError(f"{name} must have shape [{expected_n},{expected_n}], got {list(target.shape)}")
    if not bool(torch.isfinite(target).all().item()):
        raise ValueError(f"{name} must be finite")
    min_value = float(target.amin().item())
    max_value = float(target.amax().item())
    if min_value < -1e-5 or max_value > 1.0 + 1e-5:
        raise ValueError(f"{name} values must be in [0,1], got min={min_value}, max={max_value}")


def _ensure_nonnegative(name: str, value: float) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative, got {value}")

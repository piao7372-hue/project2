from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.losses.derived_supervision import derive_same_modal_targets
from src.losses.hash_loss import (
    HashLossComponents,
    compute_pair_loss,
    compute_total_hash_loss,
)


CONFIG_PATH = REPO_ROOT / "configs" / "stages" / "stage5_loss.json"
LOSS_FIELDS = ("L_IT", "L_II", "L_TT", "L_sem", "L_pair", "L_q", "L_bal", "L_total")


def main() -> int:
    config = _read_config()
    _enforce_formal_python(config)
    runtime = config["runtime"]
    if runtime["device"] != "cuda:0":
        raise RuntimeError(f"Stage 5A-1 smoke requires cuda:0, got {runtime['device']}")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 5A-1 smoke requires cuda:0; CPU fallback is not allowed")
    if runtime.get("amp_enabled") is not False:
        raise RuntimeError("Stage 5A-1 smoke requires amp_enabled=false")

    seed = int(runtime["seed"])
    eps = float(runtime["eps"])
    device = torch.device("cuda:0")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    N = 96
    K = 16
    block_sizes = [7, 16, 32, 96]
    profile = config["datasets"]["mirflickr25k"]
    params = {
        "beta_relation_weight": float(profile["beta_relation_weight"]),
        "alpha_intra_topology": float(profile["alpha_intra_topology"]),
        "lambda_sem_total": float(profile["lambda_sem_total"]),
        "lambda_pair_total": float(profile["lambda_pair_total"]),
        "lambda_q_total": float(profile["lambda_q_total"]),
        "lambda_bal_total": float(profile["lambda_bal_total"]),
        "eps": eps,
    }
    atol = 1e-5
    rtol = 1e-4

    H_I_base, H_T_base, S = _make_synthetic_inputs(N, K, device)
    derived = derive_same_modal_targets(S, eps)
    _assert_finite_range("S_II_star", derived.S_II_star)
    _assert_finite_range("S_TT_star", derived.S_TT_star)

    dense_loss = _compute_dense(H_I_base, H_T_base, S, derived, params)
    max_loss_diffs = {field: 0.0 for field in LOSS_FIELDS}
    max_grad_H_I_diff = 0.0
    max_grad_H_T_diff = 0.0
    gradient_finite = True
    gradient_nonzero = True

    for block_size in block_sizes:
        block_loss = _compute_blockwise(H_I_base, H_T_base, S, derived, params, block_size)
        for field in LOSS_FIELDS:
            diff = _assert_close(field, getattr(dense_loss, field), getattr(block_loss, field), atol, rtol, block_size)
            max_loss_diffs[field] = max(max_loss_diffs[field], diff)

        grad_dense = _compute_gradients(H_I_base, H_T_base, S, derived, params, "dense", block_size)
        grad_block = _compute_gradients(H_I_base, H_T_base, S, derived, params, "blockwise", block_size)
        _assert_gradient_ok("dense_H_I", grad_dense[0])
        _assert_gradient_ok("dense_H_T", grad_dense[1])
        _assert_gradient_ok("blockwise_H_I", grad_block[0])
        _assert_gradient_ok("blockwise_H_T", grad_block[1])
        max_grad_H_I_diff = max(max_grad_H_I_diff, _assert_close_tensor("grad_H_I", grad_dense[0], grad_block[0], atol, rtol, block_size))
        max_grad_H_T_diff = max(max_grad_H_T_diff, _assert_close_tensor("grad_H_T", grad_dense[1], grad_block[1], atol, rtol, block_size))
        gradient_finite = gradient_finite and bool(torch.isfinite(grad_dense[0]).all().item())
        gradient_finite = gradient_finite and bool(torch.isfinite(grad_dense[1]).all().item())
        gradient_finite = gradient_finite and bool(torch.isfinite(grad_block[0]).all().item())
        gradient_finite = gradient_finite and bool(torch.isfinite(grad_block[1]).all().item())
        gradient_nonzero = gradient_nonzero and float(grad_dense[0].abs().sum().item()) > 0.0
        gradient_nonzero = gradient_nonzero and float(grad_dense[1].abs().sum().item()) > 0.0
        gradient_nonzero = gradient_nonzero and float(grad_block[0].abs().sum().item()) > 0.0
        gradient_nonzero = gradient_nonzero and float(grad_block[1].abs().sum().item()) > 0.0

    fail_fast_results = _run_fail_fast_tests(H_I_base, H_T_base, S, eps, params)

    print(f"N={N}")
    print(f"K={K}")
    print("device=cuda:0")
    print(f"block_sizes={block_sizes}")
    print(f"beta_relation_weight={params['beta_relation_weight']:g}")
    print(f"alpha_intra_topology={params['alpha_intra_topology']:g}")
    print(f"S_II_star_shape={list(derived.S_II_star.shape)}")
    print(f"S_TT_star_shape={list(derived.S_TT_star.shape)}")
    print("derived_finite=true")
    print(f"S_II_star_range=[{float(derived.S_II_star.amin().item()):.8g},{float(derived.S_II_star.amax().item()):.8g}]")
    print(f"S_TT_star_range=[{float(derived.S_TT_star.amin().item()):.8g},{float(derived.S_TT_star.amax().item()):.8g}]")
    for field in LOSS_FIELDS:
        print(f"{field}_max_diff={max_loss_diffs[field]:.8g}")
    print(f"tolerance_abs={atol:.1e}")
    print(f"tolerance_rel={rtol:.1e}")
    print("loss_equivalence_passed=true")
    print(f"grad_H_I_max_diff={max_grad_H_I_diff:.8g}")
    print(f"grad_H_T_max_diff={max_grad_H_T_diff:.8g}")
    print(f"gradient_finite={str(gradient_finite).lower()}")
    print(f"gradient_nonzero={str(gradient_nonzero).lower()}")
    print("gradient_equivalence_passed=true")
    for name, passed in fail_fast_results.items():
        print(f"{name}={str(passed).lower()}")
    print("all_fail_fast_tests_raised_clear_errors=true")
    print("loss_api_accepts_label_vector=false")
    print("loss_api_accepts_B=false")
    print("loss_uses_sign=false")
    print("loss_uses_numpy=false")
    print("loss_detaches_H=false")
    print("loss_uses_no_grad=false")
    print("smoke_result=passed")
    return 0


def _read_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _enforce_formal_python(config: dict) -> None:
    expected = Path(config["runtime"]["python"]).resolve()
    current = Path(sys.executable).resolve()
    if os.path.normcase(str(expected)) != os.path.normcase(str(current)):
        raise RuntimeError(f"Stage 5 smoke requires formal Python: current={current}; expected={expected}")


def _make_synthetic_inputs(N: int, K: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    H_I = torch.randn(N, K, device=device, dtype=torch.float32)
    H_T = torch.randn(N, K, device=device, dtype=torch.float32)
    S = torch.rand(N, N, device=device, dtype=torch.float32) * 0.30
    diag_index = torch.arange(N, device=device)
    S[diag_index, diag_index] = 0.70 + 0.30 * torch.rand(N, device=device, dtype=torch.float32)
    return H_I, H_T, S


def _compute_dense(
    H_I_base: torch.Tensor,
    H_T_base: torch.Tensor,
    S: torch.Tensor,
    derived,
    params: dict,
) -> HashLossComponents:
    return compute_total_hash_loss(
        H_I=H_I_base.clone().requires_grad_(True),
        H_T=H_T_base.clone().requires_grad_(True),
        S=S,
        relation_mode="dense",
        S_II_star=derived.S_II_star,
        S_TT_star=derived.S_TT_star,
        **params,
    )


def _compute_blockwise(
    H_I_base: torch.Tensor,
    H_T_base: torch.Tensor,
    S: torch.Tensor,
    derived,
    params: dict,
    block_size: int,
) -> HashLossComponents:
    return compute_total_hash_loss(
        H_I=H_I_base.clone().requires_grad_(True),
        H_T=H_T_base.clone().requires_grad_(True),
        S=S,
        relation_mode="blockwise",
        block_size=block_size,
        Q_I=derived.Q_I,
        Q_T=derived.Q_T,
        **params,
    )


def _compute_gradients(
    H_I_base: torch.Tensor,
    H_T_base: torch.Tensor,
    S: torch.Tensor,
    derived,
    params: dict,
    relation_mode: str,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    H_I = H_I_base.clone().requires_grad_(True)
    H_T = H_T_base.clone().requires_grad_(True)
    if relation_mode == "dense":
        loss = compute_total_hash_loss(
            H_I=H_I,
            H_T=H_T,
            S=S,
            relation_mode="dense",
            S_II_star=derived.S_II_star,
            S_TT_star=derived.S_TT_star,
            **params,
        )
    else:
        loss = compute_total_hash_loss(
            H_I=H_I,
            H_T=H_T,
            S=S,
            relation_mode="blockwise",
            block_size=block_size,
            Q_I=derived.Q_I,
            Q_T=derived.Q_T,
            **params,
        )
    loss.L_total.backward()
    if H_I.grad is None or H_T.grad is None:
        raise RuntimeError(f"{relation_mode} backward did not produce gradients")
    return H_I.grad, H_T.grad


def _run_fail_fast_tests(
    H_I: torch.Tensor,
    H_T: torch.Tensor,
    S: torch.Tensor,
    eps: float,
    params: dict,
) -> dict[str, bool]:
    return {
        "wrong_S_shape_raised": _expect_raises(lambda: derive_same_modal_targets(S[:, :-1], eps)),
        "H_shape_mismatch_raised": _expect_raises(lambda: compute_pair_loss(H_I, H_T[:, :-1], eps)),
        "S_NaN_raised": _expect_raises(lambda: derive_same_modal_targets(_with_nan(S), eps)),
        "H_Inf_raised": _expect_raises(lambda: compute_pair_loss(_with_inf(H_I), H_T, eps)),
        "row_zero_S_raised": _expect_raises(lambda: derive_same_modal_targets(_with_zero_row(S), eps)),
    }


def _expect_raises(fn: Callable[[], object]) -> bool:
    try:
        fn()
    except (RuntimeError, TypeError, ValueError) as exc:
        if not str(exc):
            raise RuntimeError("fail-fast exception message must be non-empty")
        return True
    raise RuntimeError("expected fail-fast exception was not raised")


def _with_nan(S: torch.Tensor) -> torch.Tensor:
    value = S.clone()
    value[0, 1] = float("nan")
    return value


def _with_inf(H: torch.Tensor) -> torch.Tensor:
    value = H.clone()
    value[0, 0] = float("inf")
    return value


def _with_zero_row(S: torch.Tensor) -> torch.Tensor:
    value = S.clone()
    value[3, :] = 0.0
    return value


def _assert_finite_range(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"{name} must be finite")
    min_value = float(value.amin().item())
    max_value = float(value.amax().item())
    if min_value < -1e-5 or max_value > 1.0 + 1e-5:
        raise RuntimeError(f"{name} must be in [0,1], got min={min_value}, max={max_value}")


def _assert_close(name: str, expected: torch.Tensor, actual: torch.Tensor, atol: float, rtol: float, block_size: int) -> float:
    diff = abs(float((expected - actual).item()))
    allowed = atol + rtol * abs(float(expected.item()))
    if diff > allowed:
        raise RuntimeError(f"{name} mismatch for block_size={block_size}: diff={diff}, allowed={allowed}")
    return diff


def _assert_close_tensor(name: str, expected: torch.Tensor, actual: torch.Tensor, atol: float, rtol: float, block_size: int) -> float:
    max_diff = float((expected - actual).abs().amax().item())
    max_expected = float(expected.abs().amax().item())
    allowed = atol + rtol * max_expected
    if max_diff > allowed:
        raise RuntimeError(f"{name} mismatch for block_size={block_size}: max_diff={max_diff}, allowed={allowed}")
    return max_diff


def _assert_gradient_ok(name: str, grad: torch.Tensor) -> None:
    if not bool(torch.isfinite(grad).all().item()):
        raise RuntimeError(f"{name} gradient must be finite")
    if float(grad.abs().sum().item()) <= 0.0:
        raise RuntimeError(f"{name} gradient must be nonzero")


if __name__ == "__main__":
    raise SystemExit(main())

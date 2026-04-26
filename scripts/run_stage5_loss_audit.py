from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.losses.derived_supervision import row_l2_normalize
from src.losses.hash_loss import (
    compute_balance_loss,
    compute_pair_loss,
    compute_quantization_loss,
    normalize_hash_rows,
)
from src.utils.jsonl import read_json, write_json


CONFIG_PATH = REPO_ROOT / "configs" / "stages" / "stage5_loss.json"
RUNNER_VERSION = "stage5_loss_audit_runner_v2"
STAGE5_ALLOWED_DATASETS = {"mirflickr25k", "nuswide", "mscoco"}
RISK_ORDER = {"low": 0, "medium": 1, "high": 2}
RISK_VALUES = {"low", "medium", "high"}
FORBIDDEN_FLAGS = {
    "accepts_label_vector": False,
    "accepts_B": False,
    "uses_sign": False,
    "uses_numpy_loss": False,
    "detaches_H": False,
    "uses_no_grad": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 5 loss-scale audit for authorized datasets.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--all-bits", action="store_true")
    parser.add_argument("--config", default="configs/stages/stage5_loss.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_json((REPO_ROOT / args.config).resolve())
    allowed_datasets = set(config.get("execution_policy", {}).get("stage5c_allowed_datasets", []))
    if allowed_datasets != STAGE5_ALLOWED_DATASETS:
        raise RuntimeError(f"Stage 5 allowed datasets must be {sorted(STAGE5_ALLOWED_DATASETS)}, got {sorted(allowed_datasets)}")
    if args.dataset not in allowed_datasets:
        raise RuntimeError(f"Stage 5 current audit is authorized only for {sorted(allowed_datasets)}; got {args.dataset}.")
    if not args.all_bits:
        raise RuntimeError("Stage 5 audit requires --all-bits")
    _enforce_formal_python(config)
    runtime = config["runtime"]
    _enforce_cuda_runtime(runtime)
    device = torch.device(runtime["device"])
    eps = float(runtime["eps"])
    block_size = int(runtime["relation_block_size"])
    torch.manual_seed(int(runtime["seed"]))
    torch.cuda.manual_seed_all(int(runtime["seed"]))

    dataset = args.dataset
    dataset_profile = config["datasets"][dataset]
    beta_candidates = [int(value) for value in dataset_profile["beta_relation_weight_candidates"]]
    default_beta = int(dataset_profile["beta_relation_weight"])
    if default_beta not in beta_candidates:
        raise RuntimeError(f"default beta {default_beta} must be included in beta candidates {beta_candidates}")

    paths = _build_dataset_paths(dataset)
    allowed_inputs = _allowed_input_paths(paths, config["hash_bits"], dataset)
    input_fingerprints_before = {name: _fingerprint(path) for name, path in allowed_inputs.items()}

    S = _load_npy_tensor(paths["S"], device=device, expected_ndim=2, name="S")
    if S.shape != (5000, 5000):
        raise RuntimeError(f"S must have shape [5000,5000], got {list(S.shape)}")
    Q_I = row_l2_normalize(S, eps)
    Q_T = row_l2_normalize(S.transpose(0, 1), eps)
    derived_blocks = _build_derived_blocks(Q_I, Q_T, block_size)
    derived_summary = _summarize_derived_supervision(S, Q_I, Q_T, derived_blocks, block_size, dataset)

    output_root = REPO_ROOT / config["outputs"]["loss_audit_root"] / dataset
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "derived_supervision_summary.json", derived_summary)
    (output_root / "derived_supervision_summary.md").write_text(_format_derived_markdown(derived_summary), encoding="utf-8")

    bit_summaries: dict[str, Any] = {}
    for bit in [int(value) for value in config["hash_bits"]]:
        bit_dir = paths["stage4_root"] / str(bit)
        H_I = _load_npy_tensor(bit_dir / "H_I.npy", device=device, expected_ndim=2, name=f"H_I_{bit}")
        H_T = _load_npy_tensor(bit_dir / "H_T.npy", device=device, expected_ndim=2, name=f"H_T_{bit}")
        if H_I.shape != (5000, bit) or H_T.shape != (5000, bit):
            raise RuntimeError(f"Stage 4 H shape mismatch for bit={bit}: H_I={list(H_I.shape)}, H_T={list(H_T.shape)}")
        bit_summary = _audit_bit(
            dataset=dataset,
            bit=bit,
            H_I_base=H_I,
            H_T_base=H_T,
            S=S,
            derived_blocks=derived_blocks,
            profile=dataset_profile,
            beta_candidates=beta_candidates,
            default_beta=default_beta,
            block_size=block_size,
            eps=eps,
        )
        output_dir = output_root / str(bit)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "loss_audit_summary.json", bit_summary)
        (output_dir / "loss_audit_summary.md").write_text(_format_bit_markdown(bit_summary), encoding="utf-8")
        bit_summaries[str(bit)] = {
            "passed": bit_summary["passed"],
            "default_beta": default_beta,
            "default_beta_total": bit_summary["beta_audits"][str(default_beta)]["L_total"],
        }

    input_fingerprints_after = {name: _fingerprint(path) for name, path in allowed_inputs.items()}
    input_integrity = {
        "allowed_input_fingerprints_before_after_match": input_fingerprints_before == input_fingerprints_after,
        "stage5_audit_modified_allowed_inputs": input_fingerprints_before != input_fingerprints_after,
        "checked_input_names": sorted(allowed_inputs),
    }
    aggregate = {
        "stage": "stage5",
        "substage": "Stage 5B/5C",
        "runner_version": RUNNER_VERSION,
        "dataset": dataset,
        "hash_bits": [int(value) for value in config["hash_bits"]],
        "default_beta": default_beta,
        "beta_candidates": beta_candidates,
        "block_size": block_size,
        "forbidden_flags": FORBIDDEN_FLAGS,
        "input_integrity": input_integrity,
        "outputs_root": str(output_root.relative_to(REPO_ROOT)),
        "derived_profile_norm_risk": derived_summary["derived_profile_norm_risk"],
        "bits": bit_summaries,
        "passed": all(summary["passed"] for summary in bit_summaries.values()) and not input_integrity["stage5_audit_modified_allowed_inputs"],
        "final_beta_selected": False,
        "stage6_parameters_modified": False,
    }
    write_json(output_root / "stage5_loss_audit_summary.json", aggregate)

    print(f"dataset={dataset}")
    print(f"hash_bits={aggregate['hash_bits']}")
    print(f"beta_candidates={beta_candidates}")
    print(f"default_beta={default_beta}")
    print(f"block_size={block_size}")
    print(f"derived_summary={output_root / 'derived_supervision_summary.json'}")
    for bit, summary in bit_summaries.items():
        print(f"bit={bit}")
        print(f"passed={str(summary['passed']).lower()}")
        print(f"default_beta_L_total={summary['default_beta_total']}")
    print(f"input_integrity_match={str(input_integrity['allowed_input_fingerprints_before_after_match']).lower()}")
    print(f"passed={str(aggregate['passed']).lower()}")
    return 0 if aggregate["passed"] else 1


def _enforce_formal_python(config: dict[str, Any]) -> None:
    expected = Path(config["runtime"]["python"]).resolve()
    current = Path(sys.executable).resolve()
    if os.path.normcase(str(expected)) != os.path.normcase(str(current)):
        raise RuntimeError(f"Stage 5 loss audit requires formal Python: current={current}; expected={expected}")


def _enforce_cuda_runtime(runtime: dict[str, Any]) -> None:
    if runtime["device"] != "cuda:0":
        raise RuntimeError(f"Stage 5 formal audit requires cuda:0, got {runtime['device']}")
    if runtime.get("amp_enabled") is not False:
        raise RuntimeError("Stage 5 formal audit requires amp_enabled=false")
    if runtime.get("dtype") != "float32":
        raise RuntimeError(f"Stage 5 formal audit requires float32, got {runtime.get('dtype')}")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 5 formal audit requires cuda:0; CPU fallback is not allowed")


def _build_dataset_paths(dataset: str) -> dict[str, Path]:
    processed = REPO_ROOT / "data" / "processed" / dataset
    semantic = processed / "semantic_cache" / "se_c_s_formal_v1"
    stage4_root = processed / "model_cache" / "stage4_forward_v1"
    return {
        "S": semantic / "S.npy",
        "semantic_meta": semantic / "meta.json",
        "semantic_diagnostics": semantic / "semantic_diagnostics.json",
        "stage4_root": stage4_root,
    }


def _allowed_input_paths(paths: dict[str, Path], bits: list[int], dataset: str) -> dict[str, Path]:
    inputs = {
        "S": paths["S"],
        "semantic_meta": paths["semantic_meta"],
    }
    if dataset == "mirflickr25k":
        inputs["semantic_diagnostics"] = paths["semantic_diagnostics"]
    for bit in [int(value) for value in bits]:
        bit_dir = paths["stage4_root"] / str(bit)
        inputs[f"H_I_{bit}"] = bit_dir / "H_I.npy"
        inputs[f"H_T_{bit}"] = bit_dir / "H_T.npy"
        inputs[f"stage4_meta_{bit}"] = bit_dir / "meta.json"
    for name, path in inputs.items():
        if not path.exists():
            raise RuntimeError(f"missing allowed Stage 5 input for dataset={dataset}, name={name}: {path}")
    return inputs


def _fingerprint(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _load_npy_tensor(path: Path, device: torch.device, expected_ndim: int, name: str) -> torch.Tensor:
    if not path.exists():
        raise RuntimeError(f"missing {name}: {path}")
    array = np.load(path)
    if array.ndim != expected_ndim:
        raise RuntimeError(f"{name} must be rank-{expected_ndim}, got shape={list(array.shape)}")
    if array.dtype != np.float32:
        array = array.astype(np.float32, copy=False)
    tensor = torch.tensor(array, device=device, dtype=torch.float32)
    if not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError(f"{name} must be finite")
    return tensor


def _build_derived_blocks(Q_I: torch.Tensor, Q_T: torch.Tensor, block_size: int) -> list[dict[str, Any]]:
    n = Q_I.shape[0]
    blocks: list[dict[str, Any]] = []
    try:
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            S_II_block = (Q_I[start:end, :] @ Q_I.transpose(0, 1)).cpu()
            S_TT_block = (Q_T[start:end, :] @ Q_T.transpose(0, 1)).cpu()
            blocks.append({"start": start, "end": end, "S_II": S_II_block, "S_TT": S_TT_block})
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(f"CUDA OOM while deriving same-modal target blocks; reduce relation_block_size={block_size}") from exc
        raise
    return blocks


def _summarize_derived_supervision(
    S: torch.Tensor,
    Q_I: torch.Tensor,
    Q_T: torch.Tensor,
    derived_blocks: list[dict[str, Any]],
    block_size: int,
    dataset: str,
) -> dict[str, Any]:
    n = S.shape[0]
    row_norms_s = torch.linalg.vector_norm(S, ord=2, dim=1)
    col_norms_s = torch.linalg.vector_norm(S, ord=2, dim=0)
    q_i_norms = torch.linalg.vector_norm(Q_I, ord=2, dim=1)
    q_t_norms = torch.linalg.vector_norm(Q_T, ord=2, dim=1)
    S_stats = _square_stats_from_dense(S)
    S_II_stats = _square_stats_from_blocks(derived_blocks, "S_II", n)
    S_TT_stats = _square_stats_from_blocks(derived_blocks, "S_TT", n)
    row_zero_count = int((row_norms_s == 0.0).sum().item())
    col_zero_count = int((col_norms_s == 0.0).sum().item())
    row_near_zero = _near_zero_counts(row_norms_s)
    col_near_zero = _near_zero_counts(col_norms_s)
    derived_profile_norm_risk = _classify_derived_profile_norm_risk(
        row_zero_count=row_zero_count,
        col_zero_count=col_zero_count,
        row_near_zero=row_near_zero,
        col_near_zero=col_near_zero,
        q_i_norms=q_i_norms,
        q_t_norms=q_t_norms,
    )
    return {
        "stage": "stage5",
        "substage": "Stage 5B/5C",
        "dataset": dataset,
        "block_size": block_size,
        "S_shape": [int(S.shape[0]), int(S.shape[1])],
        "S_min": S_stats["min"],
        "S_max": S_stats["max"],
        "S_mean": S_stats["mean"],
        "S_std": S_stats["std"],
        "S_diag_mean": S_stats["diag_mean"],
        "S_offdiag_mean": S_stats["offdiag_mean"],
        "S_row_zero_count": row_zero_count,
        "S_col_zero_count": col_zero_count,
        "S_row_l2_norm_min": _to_float(row_norms_s.amin()),
        "S_row_l2_norm_max": _to_float(row_norms_s.amax()),
        "S_row_l2_norm_mean": _to_float(row_norms_s.mean()),
        "S_row_l2_norm_median": _to_float(torch.median(row_norms_s)),
        "S_col_l2_norm_min": _to_float(col_norms_s.amin()),
        "S_col_l2_norm_max": _to_float(col_norms_s.amax()),
        "S_col_l2_norm_mean": _to_float(col_norms_s.mean()),
        "S_col_l2_norm_median": _to_float(torch.median(col_norms_s)),
        "S_row_near_zero_count_1e-8": row_near_zero["1e-8"],
        "S_row_near_zero_count_1e-7": row_near_zero["1e-7"],
        "S_row_near_zero_count_1e-6": row_near_zero["1e-6"],
        "S_col_near_zero_count_1e-8": col_near_zero["1e-8"],
        "S_col_near_zero_count_1e-7": col_near_zero["1e-7"],
        "S_col_near_zero_count_1e-6": col_near_zero["1e-6"],
        "Q_I_row_norm_min": _to_float(q_i_norms.amin()),
        "Q_I_row_norm_max": _to_float(q_i_norms.amax()),
        "Q_T_row_norm_min": _to_float(q_t_norms.amin()),
        "Q_T_row_norm_max": _to_float(q_t_norms.amax()),
        "derived_profile_norm_risk": derived_profile_norm_risk,
        "S_II_star_shape": [n, n],
        "S_TT_star_shape": [n, n],
        "S_II_star_diag_mean": S_II_stats["diag_mean"],
        "S_II_star_offdiag_mean": S_II_stats["offdiag_mean"],
        "S_II_star_min": S_II_stats["min"],
        "S_II_star_max": S_II_stats["max"],
        "S_II_star_range_tolerance_passed": bool(S_II_stats["min"] >= -1e-5 and S_II_stats["max"] <= 1.0 + 1e-5),
        "S_TT_star_diag_mean": S_TT_stats["diag_mean"],
        "S_TT_star_offdiag_mean": S_TT_stats["offdiag_mean"],
        "S_TT_star_min": S_TT_stats["min"],
        "S_TT_star_max": S_TT_stats["max"],
        "S_TT_star_range_tolerance_passed": bool(S_TT_stats["min"] >= -1e-5 and S_TT_stats["max"] <= 1.0 + 1e-5),
        "derived_matrices_written": False,
        "data_processed_written": False,
    }


def _audit_bit(
    dataset: str,
    bit: int,
    H_I_base: torch.Tensor,
    H_T_base: torch.Tensor,
    S: torch.Tensor,
    derived_blocks: list[dict[str, Any]],
    profile: dict[str, Any],
    beta_candidates: list[int],
    default_beta: int,
    block_size: int,
    eps: float,
) -> dict[str, Any]:
    beta_audits: dict[str, Any] = {}
    for beta in beta_candidates:
        audit = _audit_beta(
            H_I_base=H_I_base,
            H_T_base=H_T_base,
            S=S,
            derived_blocks=derived_blocks,
            profile=profile,
            beta=float(beta),
            block_size=block_size,
            eps=eps,
        )
        beta_audits[str(beta)] = audit
    beta_effectiveness_risk = _classify_beta_effectiveness_risk(beta_audits)
    for audit in beta_audits.values():
        audit["beta_effectiveness_risk"] = beta_effectiveness_risk
        audit["loss_balance_risk"] = _max_risk(
            audit["pair_dominance_risk"],
            audit["semantic_underweight_risk"],
            audit["beta_effectiveness_risk"],
        )
        audit["passed"] = bool(audit["passed"] and _risk_values_valid(audit))
    passed = all(audit["passed"] for audit in beta_audits.values())
    return {
        "stage": "stage5",
        "substage": "Stage 5B/5C",
        "runner_version": RUNNER_VERSION,
        "dataset": dataset,
        "bit": bit,
        "H_I_shape": [int(H_I_base.shape[0]), int(H_I_base.shape[1])],
        "H_T_shape": [int(H_T_base.shape[0]), int(H_T_base.shape[1])],
        "block_size": block_size,
        "default_beta": default_beta,
        "beta_candidates": beta_candidates,
        "beta_audits": beta_audits,
        "forbidden_flags": FORBIDDEN_FLAGS,
        "final_beta_selected": False,
        "stage6_parameters_modified": False,
        "passed": passed,
    }


def _audit_beta(
    H_I_base: torch.Tensor,
    H_T_base: torch.Tensor,
    S: torch.Tensor,
    derived_blocks: list[dict[str, Any]],
    profile: dict[str, Any],
    beta: float,
    block_size: int,
    eps: float,
) -> dict[str, Any]:
    H_I = H_I_base.clone().requires_grad_(True)
    H_T = H_T_base.clone().requires_grad_(True)
    H_I_hat = normalize_hash_rows(H_I, eps)
    H_T_hat = normalize_hash_rows(H_T, eps)
    n = H_I.shape[0]

    num_it = H_I.new_zeros(())
    den_it = H_I.new_zeros(())
    num_ii = H_I.new_zeros(())
    den_ii = H_I.new_zeros(())
    num_tt = H_I.new_zeros(())
    den_tt = H_I.new_zeros(())
    w_it_acc = _new_weight_accumulator()
    w_ii_acc = _new_weight_accumulator()
    w_tt_acc = _new_weight_accumulator()

    try:
        for block in derived_blocks:
            start = int(block["start"])
            end = int(block["end"])
            row_count = end - start
            row_index = torch.arange(row_count, device=H_I.device)
            col_index = torch.arange(start, end, device=H_I.device)

            S_block = S[start:end, :]
            S_II_block = block["S_II"].to(device=H_I.device)
            S_TT_block = block["S_TT"].to(device=H_I.device)

            P_IT_block = (1.0 + H_I_hat[start:end, :] @ H_T_hat.transpose(0, 1)) / 2.0
            P_II_block = (1.0 + H_I_hat[start:end, :] @ H_I_hat.transpose(0, 1)) / 2.0
            P_TT_block = (1.0 + H_T_hat[start:end, :] @ H_T_hat.transpose(0, 1)) / 2.0

            num_it, den_it = _accumulate_relation(num_it, den_it, P_IT_block, S_block, beta)
            num_ii, den_ii = _accumulate_relation(num_ii, den_ii, P_II_block, S_II_block, beta)
            num_tt, den_tt = _accumulate_relation(num_tt, den_tt, P_TT_block, S_TT_block, beta)

            _accumulate_weight_stats(w_it_acc, 1.0 + beta * S_block, row_index, col_index, n)
            _accumulate_weight_stats(w_ii_acc, 1.0 + beta * S_II_block, row_index, col_index, n)
            _accumulate_weight_stats(w_tt_acc, 1.0 + beta * S_TT_block, row_index, col_index, n)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(f"CUDA OOM during Stage 5 blockwise loss audit; reduce relation_block_size={block_size}") from exc
        raise

    L_IT = _safe_divide(num_it, den_it, "W_IT")
    L_II = _safe_divide(num_ii, den_ii, "W_II")
    L_TT = _safe_divide(num_tt, den_tt, "W_TT")
    L_sem = L_IT + float(profile["alpha_intra_topology"]) / 2.0 * (L_II + L_TT)
    L_pair = compute_pair_loss(H_I, H_T, eps)
    L_q = compute_quantization_loss(H_I, H_T)
    L_bal = compute_balance_loss(H_I, H_T)
    weighted_sem = float(profile["lambda_sem_total"]) * L_sem
    weighted_pair = float(profile["lambda_pair_total"]) * L_pair
    weighted_q = float(profile["lambda_q_total"]) * L_q
    weighted_bal = float(profile["lambda_bal_total"]) * L_bal
    L_total = weighted_sem + weighted_pair + weighted_q + weighted_bal
    L_total.backward()
    if H_I.grad is None or H_T.grad is None:
        raise RuntimeError("Stage 5 loss backward did not produce H gradients")
    gradient_norm_H_I = _to_float(torch.linalg.vector_norm(H_I.grad))
    gradient_norm_H_T = _to_float(torch.linalg.vector_norm(H_T.grad))
    gradient_finite = bool(torch.isfinite(H_I.grad).all().item() and torch.isfinite(H_T.grad).all().item())
    gradient_nonzero = bool(gradient_norm_H_I > 0.0 and gradient_norm_H_T > 0.0)
    weighted_sum_check_diff = abs(_to_float(L_total) - (_to_float(weighted_sem) + _to_float(weighted_pair) + _to_float(weighted_q) + _to_float(weighted_bal)))
    total_value = _to_float(L_total)
    ratios = {
        "sem": _safe_ratio(_to_float(weighted_sem), total_value),
        "pair": _safe_ratio(_to_float(weighted_pair), total_value),
        "q": _safe_ratio(_to_float(weighted_q), total_value),
        "bal": _safe_ratio(_to_float(weighted_bal), total_value),
    }
    pair_risk = _classify_pair_dominance_risk(ratios["pair"])
    sem_risk = _classify_semantic_underweight_risk(ratios["sem"])
    losses = {
        "L_IT": _to_float(L_IT),
        "L_II": _to_float(L_II),
        "L_TT": _to_float(L_TT),
        "L_sem": _to_float(L_sem),
        "L_pair": _to_float(L_pair),
        "L_q": _to_float(L_q),
        "L_bal": _to_float(L_bal),
        "L_total": total_value,
    }
    all_losses_ok = all(math.isfinite(value) and value >= 0.0 for value in losses.values())
    return {
        "beta_relation_weight": beta,
        **_prefix_stats("W_IT", _finalize_weight_stats(w_it_acc)),
        **_prefix_stats("W_II", _finalize_weight_stats(w_ii_acc)),
        **_prefix_stats("W_TT", _finalize_weight_stats(w_tt_acc)),
        **losses,
        "weighted_sem_component": _to_float(weighted_sem),
        "weighted_pair_component": _to_float(weighted_pair),
        "weighted_q_component": _to_float(weighted_q),
        "weighted_bal_component": _to_float(weighted_bal),
        "weighted_sem_component_ratio": ratios["sem"],
        "weighted_pair_component_ratio": ratios["pair"],
        "weighted_q_component_ratio": ratios["q"],
        "weighted_bal_component_ratio": ratios["bal"],
        "loss_component_ratios": ratios,
        "pair_dominance_risk": pair_risk,
        "semantic_underweight_risk": sem_risk,
        "beta_effectiveness_risk": "low",
        "loss_balance_risk": _max_risk(pair_risk, sem_risk, "low"),
        "gradient_norm_H_I": gradient_norm_H_I,
        "gradient_norm_H_T": gradient_norm_H_T,
        "gradient_finite": gradient_finite,
        "gradient_nonzero": gradient_nonzero,
        "loss_finite_nonnegative": all_losses_ok,
        "weighted_sum_check_diff": weighted_sum_check_diff,
        "L_total_weighted_sum_match": bool(weighted_sum_check_diff <= 1e-5 + 1e-5 * max(1.0, abs(total_value))),
        "passed": bool(all_losses_ok and gradient_finite and gradient_nonzero and weighted_sum_check_diff <= 1e-5 + 1e-5 * max(1.0, abs(total_value))),
    }


def _classify_pair_dominance_risk(ratio: float) -> str:
    if ratio <= 0.60:
        return "low"
    if ratio <= 0.80:
        return "medium"
    return "high"


def _classify_semantic_underweight_risk(ratio: float) -> str:
    if ratio >= 0.25:
        return "low"
    if ratio >= 0.15:
        return "medium"
    return "high"


def _near_zero_counts(norms: torch.Tensor) -> dict[str, int]:
    return {
        "1e-8": int((norms <= 1e-8).sum().item()),
        "1e-7": int((norms <= 1e-7).sum().item()),
        "1e-6": int((norms <= 1e-6).sum().item()),
    }


def _classify_derived_profile_norm_risk(
    *,
    row_zero_count: int,
    col_zero_count: int,
    row_near_zero: dict[str, int],
    col_near_zero: dict[str, int],
    q_i_norms: torch.Tensor,
    q_t_norms: torch.Tensor,
) -> str:
    if row_zero_count > 0 or col_zero_count > 0:
        return "high"
    if row_near_zero["1e-8"] > 0 or col_near_zero["1e-8"] > 0:
        return "high"
    q_min = min(_to_float(q_i_norms.amin()), _to_float(q_t_norms.amin()))
    if row_near_zero["1e-7"] > 0 or col_near_zero["1e-7"] > 0 or q_min < 0.95:
        return "medium"
    return "low"


def _classify_beta_effectiveness_risk(beta_audits: dict[str, Any]) -> str:
    audits = list(beta_audits.values())
    if any(audit.get("gradient_finite") is not True or audit.get("loss_finite_nonnegative") is not True for audit in audits):
        return "high"
    contrasts = [float(audit["W_IT_diag_offdiag_contrast"]) for audit in audits]
    if max(contrasts) - min(contrasts) < 0.10:
        return "high"
    l_sem_values = [float(audit["L_sem"]) for audit in audits]
    l_total_values = [float(audit["L_total"]) for audit in audits]
    if _relative_range(l_sem_values) <= 0.05 and _relative_range(l_total_values) <= 0.02:
        return "medium"
    return "low"


def _relative_range(values: list[float]) -> float:
    center = max(abs(sum(values) / len(values)), 1e-12)
    return (max(values) - min(values)) / center


def _max_risk(*risks: str) -> str:
    for risk in risks:
        if risk not in RISK_ORDER:
            raise RuntimeError(f"invalid risk value: {risk}")
    return max(risks, key=lambda value: RISK_ORDER[value])


def _risk_values_valid(audit: dict[str, Any]) -> bool:
    return all(
        audit.get(key) in RISK_VALUES
        for key in ["pair_dominance_risk", "semantic_underweight_risk", "beta_effectiveness_risk", "loss_balance_risk"]
    )


def _accumulate_relation(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = 1.0 + beta * target
    return numerator + torch.sum(weights * (prediction - target).pow(2)), denominator + torch.sum(weights)


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor, name: str) -> torch.Tensor:
    if not bool(torch.isfinite(denominator).all().item()):
        raise RuntimeError(f"{name} denominator must be finite")
    if float(denominator.item()) <= 0.0:
        raise RuntimeError(f"{name} denominator must be positive")
    return numerator / denominator


def _new_weight_accumulator() -> dict[str, float]:
    return {
        "count": 0.0,
        "sum": 0.0,
        "sum_sq": 0.0,
        "min": float("inf"),
        "max": float("-inf"),
        "diag_sum": 0.0,
        "diag_count": 0.0,
    }


def _accumulate_weight_stats(acc: dict[str, float], weights: torch.Tensor, row_index: torch.Tensor, col_index: torch.Tensor, n: int) -> None:
    count = float(weights.numel())
    acc["count"] += count
    acc["sum"] += _to_float(weights.sum())
    acc["sum_sq"] += _to_float(weights.pow(2).sum())
    acc["min"] = min(acc["min"], _to_float(weights.amin()))
    acc["max"] = max(acc["max"], _to_float(weights.amax()))
    diag = weights[row_index, col_index]
    acc["diag_sum"] += _to_float(diag.sum())
    acc["diag_count"] += float(diag.numel())
    if acc["diag_count"] > n:
        raise RuntimeError("diag_count exceeded expected matrix size")


def _finalize_weight_stats(acc: dict[str, float]) -> dict[str, float]:
    if acc["count"] <= 0.0 or acc["diag_count"] <= 0.0:
        raise RuntimeError("cannot finalize empty weight statistics")
    mean = acc["sum"] / acc["count"]
    variance = max(acc["sum_sq"] / acc["count"] - mean * mean, 0.0)
    diag_mean = acc["diag_sum"] / acc["diag_count"]
    offdiag_count = acc["count"] - acc["diag_count"]
    offdiag_mean = (acc["sum"] - acc["diag_sum"]) / offdiag_count
    return {
        "min": acc["min"],
        "max": acc["max"],
        "mean": mean,
        "std": math.sqrt(variance),
        "diag_mean": diag_mean,
        "offdiag_mean": offdiag_mean,
        "diag_offdiag_contrast": _safe_ratio(diag_mean, offdiag_mean),
    }


def _prefix_stats(prefix: str, stats: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in stats.items()}


def _square_stats_from_dense(matrix: torch.Tensor) -> dict[str, float]:
    n = matrix.shape[0]
    diag = matrix.diagonal()
    total = _to_float(matrix.sum())
    diag_sum = _to_float(diag.sum())
    offdiag_mean = (total - diag_sum) / float(n * n - n)
    return {
        "min": _to_float(matrix.amin()),
        "max": _to_float(matrix.amax()),
        "mean": _to_float(matrix.mean()),
        "std": _to_float(matrix.std(unbiased=False)),
        "diag_mean": _to_float(diag.mean()),
        "offdiag_mean": offdiag_mean,
    }


def _square_stats_from_blocks(blocks: list[dict[str, Any]], key: str, n: int) -> dict[str, float]:
    count = 0.0
    total = 0.0
    total_sq = 0.0
    min_value = float("inf")
    max_value = float("-inf")
    diag_total = 0.0
    diag_count = 0.0
    for block in blocks:
        value = block[key]
        start = int(block["start"])
        end = int(block["end"])
        row_count = end - start
        count += float(value.numel())
        total += _to_float(value.sum())
        total_sq += _to_float(value.pow(2).sum())
        min_value = min(min_value, _to_float(value.amin()))
        max_value = max(max_value, _to_float(value.amax()))
        diag = value[torch.arange(row_count), torch.arange(start, end)]
        diag_total += _to_float(diag.sum())
        diag_count += float(diag.numel())
    mean = total / count
    variance = max(total_sq / count - mean * mean, 0.0)
    offdiag_mean = (total - diag_total) / float(n * n - n)
    return {
        "min": min_value,
        "max": max_value,
        "mean": mean,
        "std": math.sqrt(variance),
        "diag_mean": diag_total / diag_count,
        "offdiag_mean": offdiag_mean,
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or abs(denominator) <= 1e-12:
        raise RuntimeError(f"invalid ratio numerator={numerator}, denominator={denominator}")
    return numerator / denominator


def _to_float(value: torch.Tensor) -> float:
    result = float(value.item())
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite scalar value: {result}")
    return result


def _format_derived_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage 5B/5C Derived Supervision Summary",
            "",
            f"- dataset: {summary['dataset']}",
            f"- S_shape: {summary['S_shape']}",
            f"- S_diag_mean: {summary['S_diag_mean']}",
            f"- S_offdiag_mean: {summary['S_offdiag_mean']}",
            f"- S_row_zero_count: {summary['S_row_zero_count']}",
            f"- S_col_zero_count: {summary['S_col_zero_count']}",
            f"- S_row_l2_norm_range: [{summary['S_row_l2_norm_min']}, {summary['S_row_l2_norm_max']}]",
            f"- S_col_l2_norm_range: [{summary['S_col_l2_norm_min']}, {summary['S_col_l2_norm_max']}]",
            f"- S_row_near_zero_counts: 1e-8={summary['S_row_near_zero_count_1e-8']}, 1e-7={summary['S_row_near_zero_count_1e-7']}, 1e-6={summary['S_row_near_zero_count_1e-6']}",
            f"- S_col_near_zero_counts: 1e-8={summary['S_col_near_zero_count_1e-8']}, 1e-7={summary['S_col_near_zero_count_1e-7']}, 1e-6={summary['S_col_near_zero_count_1e-6']}",
            f"- Q_I_row_norm_range: [{summary['Q_I_row_norm_min']}, {summary['Q_I_row_norm_max']}]",
            f"- Q_T_row_norm_range: [{summary['Q_T_row_norm_min']}, {summary['Q_T_row_norm_max']}]",
            f"- derived_profile_norm_risk: {summary['derived_profile_norm_risk']}",
            f"- S_II_star_diag_mean: {summary['S_II_star_diag_mean']}",
            f"- S_II_star_offdiag_mean: {summary['S_II_star_offdiag_mean']}",
            f"- S_TT_star_diag_mean: {summary['S_TT_star_diag_mean']}",
            f"- S_TT_star_offdiag_mean: {summary['S_TT_star_offdiag_mean']}",
            f"- range_tolerance_passed: {summary['S_II_star_range_tolerance_passed'] and summary['S_TT_star_range_tolerance_passed']}",
            "",
        ]
    )


def _format_bit_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Stage 5B/5C Loss Audit: {summary['dataset']} {summary['bit']}-bit",
        "",
        f"- default_beta: {summary['default_beta']}",
        f"- beta_candidates: {summary['beta_candidates']}",
        f"- passed: {summary['passed']}",
        "",
        "| beta | L_sem | L_pair | L_total | sem ratio | pair ratio | W_IT contrast | grad H_I | grad H_T | loss balance risk | passed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for beta, audit in summary["beta_audits"].items():
        lines.append(
            f"| {beta} | {audit['L_sem']:.8g} | {audit['L_pair']:.8g} | {audit['L_total']:.8g} | "
            f"{audit['weighted_sem_component_ratio']:.8g} | {audit['weighted_pair_component_ratio']:.8g} | "
            f"{audit['W_IT_diag_offdiag_contrast']:.8g} | {audit['gradient_norm_H_I']:.8g} | "
            f"{audit['gradient_norm_H_T']:.8g} | {audit['loss_balance_risk']} | {audit['passed']} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

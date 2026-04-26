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
from src.losses.hash_loss import compute_total_hash_loss
from src.utils.jsonl import read_json, write_json


CONFIG_PATH = REPO_ROOT / "configs" / "stages" / "stage5_loss.json"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "stage5_loss_weight_sensitivity"
AUDIT_VERSION = "stage5_loss_weight_sensitivity_v1"
DATASETS = ["mirflickr25k", "nuswide", "mscoco"]
RISK_ORDER = {"low": 0, "medium": 1, "high": 2}
FORBIDDEN_FLAGS = {
    "accepts_label_vector": False,
    "accepts_B": False,
    "uses_sign": False,
    "uses_numpy_loss": False,
    "detaches_H": False,
    "uses_no_grad": False,
}


CANDIDATE_PROFILES: dict[str, list[dict[str, Any]]] = {
    "mirflickr25k": [
        {"name": "mir_current", "lambda_sem_total": 1.00, "lambda_pair_total": 0.60, "lambda_q_total": 0.05, "lambda_bal_total": 0.01},
        {"name": "mir_sem1p5_pair0p45", "lambda_sem_total": 1.50, "lambda_pair_total": 0.45, "lambda_q_total": 0.05, "lambda_bal_total": 0.01},
        {"name": "mir_sem2p0_pair0p35", "lambda_sem_total": 2.00, "lambda_pair_total": 0.35, "lambda_q_total": 0.05, "lambda_bal_total": 0.01},
        {"name": "mir_sem1p5_pair0p30", "lambda_sem_total": 1.50, "lambda_pair_total": 0.30, "lambda_q_total": 0.05, "lambda_bal_total": 0.01},
    ],
    "nuswide": [
        {"name": "nus_current", "lambda_sem_total": 1.20, "lambda_pair_total": 0.40, "lambda_q_total": 0.08, "lambda_bal_total": 0.02},
        {"name": "nus_sem1p5_pair0p35", "lambda_sem_total": 1.50, "lambda_pair_total": 0.35, "lambda_q_total": 0.08, "lambda_bal_total": 0.02},
        {"name": "nus_sem1p5_pair0p30", "lambda_sem_total": 1.50, "lambda_pair_total": 0.30, "lambda_q_total": 0.08, "lambda_bal_total": 0.02},
        {"name": "nus_sem2p0_pair0p30", "lambda_sem_total": 2.00, "lambda_pair_total": 0.30, "lambda_q_total": 0.08, "lambda_bal_total": 0.02},
    ],
    "mscoco": [
        {"name": "coco_current", "lambda_sem_total": 1.00, "lambda_pair_total": 0.80, "lambda_q_total": 0.05, "lambda_bal_total": 0.01},
        {"name": "coco_sem1p5_pair0p60", "lambda_sem_total": 1.50, "lambda_pair_total": 0.60, "lambda_q_total": 0.05, "lambda_bal_total": 0.01},
        {"name": "coco_sem2p0_pair0p50", "lambda_sem_total": 2.00, "lambda_pair_total": 0.50, "lambda_q_total": 0.05, "lambda_bal_total": 0.01},
        {"name": "coco_sem2p0_pair0p40", "lambda_sem_total": 2.00, "lambda_pair_total": 0.40, "lambda_q_total": 0.05, "lambda_bal_total": 0.01},
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Stage 5 loss-weight sensitivity for Stage 6 dev candidates.")
    parser.add_argument("--all-datasets", action="store_true")
    bit_group = parser.add_mutually_exclusive_group(required=True)
    bit_group.add_argument("--all-bits", action="store_true")
    bit_group.add_argument("--bit", type=int)
    parser.add_argument("--config", default="configs/stages/stage5_loss.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.all_datasets:
        raise RuntimeError("Stage 5D sensitivity audit requires --all-datasets")
    config = read_json((REPO_ROOT / args.config).resolve())
    _enforce_formal_python(config)
    runtime = config["runtime"]
    _enforce_cuda_runtime(runtime)
    bits = [int(value) for value in config["hash_bits"]] if args.all_bits else [int(args.bit)]
    for bit in bits:
        if bit not in [int(value) for value in config["hash_bits"]]:
            raise RuntimeError(f"unsupported bit={bit}; expected one of {config['hash_bits']}")

    device = torch.device(runtime["device"])
    eps = float(runtime["eps"])
    block_size = int(runtime["relation_block_size"])
    torch.manual_seed(int(runtime["seed"]))
    torch.cuda.manual_seed_all(int(runtime["seed"]))

    dataset_summaries: dict[str, Any] = {}
    for dataset in DATASETS:
        dataset_summary = _audit_dataset(
            dataset=dataset,
            bits=bits,
            config=config,
            device=device,
            eps=eps,
            block_size=block_size,
        )
        dataset_summaries[dataset] = dataset_summary

    global_summary = _build_global_summary(bits, dataset_summaries)
    write_json(OUTPUT_ROOT / "global_loss_weight_sensitivity_summary.json", global_summary)
    (OUTPUT_ROOT / "global_loss_weight_sensitivity_summary.md").write_text(_format_global_markdown(global_summary), encoding="utf-8")

    print(f"datasets={DATASETS}")
    print(f"bits={bits}")
    for dataset, summary in dataset_summaries.items():
        rec = summary["recommended_stage6_dev_candidates"]
        print(f"dataset={dataset}")
        print(f"primary_candidate={rec['primary_candidate']['name']}")
        print(f"backup_candidate={rec['backup_candidate']['name'] if rec['backup_candidate'] else 'none'}")
        print(f"current_profile_risk={summary['current_profile_risk']}")
        print(f"best_profile_risk={rec['primary_candidate']['max_overall_loss_profile_risk']}")
    print(f"final_stage6_profile_selected={str(global_summary['final_stage6_profile_selected']).lower()}")
    print(f"stage6_parameters_modified={str(global_summary['stage6_parameters_modified']).lower()}")
    print(f"passed={str(global_summary['passed']).lower()}")
    return 0 if global_summary["passed"] else 1


def _audit_dataset(
    *,
    dataset: str,
    bits: list[int],
    config: dict[str, Any],
    device: torch.device,
    eps: float,
    block_size: int,
) -> dict[str, Any]:
    profile = config["datasets"][dataset]
    beta = int(profile["beta_relation_weight"])
    S = _load_npy_tensor(_semantic_path(dataset) / "S.npy", device=device, expected_ndim=2, name=f"{dataset}_S")
    if S.shape != (5000, 5000):
        raise RuntimeError(f"{dataset} S must have shape [5000,5000], got {list(S.shape)}")
    Q_I = row_l2_normalize(S, eps)
    Q_T = row_l2_normalize(S.transpose(0, 1), eps)

    bit_summaries: dict[str, Any] = {}
    for bit in bits:
        _validate_existing_stage5_summary(dataset, bit, beta)
        H_I = _load_npy_tensor(_stage4_bit_path(dataset, bit) / "H_I.npy", device=device, expected_ndim=2, name=f"{dataset}_H_I_{bit}")
        H_T = _load_npy_tensor(_stage4_bit_path(dataset, bit) / "H_T.npy", device=device, expected_ndim=2, name=f"{dataset}_H_T_{bit}")
        if H_I.shape != (5000, bit) or H_T.shape != (5000, bit):
            raise RuntimeError(f"{dataset} bit={bit} H shape mismatch: H_I={list(H_I.shape)}, H_T={list(H_T.shape)}")
        bit_summary = _audit_bit(
            dataset=dataset,
            bit=bit,
            H_I_base=H_I,
            H_T_base=H_T,
            S=S,
            Q_I=Q_I,
            Q_T=Q_T,
            alpha_intra_topology=float(profile["alpha_intra_topology"]),
            beta=beta,
            candidates=CANDIDATE_PROFILES[dataset],
            eps=eps,
            block_size=block_size,
        )
        output_dir = OUTPUT_ROOT / dataset / str(bit)
        write_json(output_dir / "loss_weight_sensitivity_summary.json", bit_summary)
        (output_dir / "loss_weight_sensitivity_summary.md").write_text(_format_bit_markdown(bit_summary), encoding="utf-8")
        bit_summaries[str(bit)] = bit_summary

    dataset_candidate_summary = _aggregate_dataset_candidates(bit_summaries, CANDIDATE_PROFILES[dataset])
    recommendations = _recommend_candidates(dataset_candidate_summary)
    dataset_summary = {
        "stage": "stage5",
        "substage": "Stage 5D loss-weight sensitivity",
        "audit_version": AUDIT_VERSION,
        "dataset": dataset,
        "bits": bits,
        "beta_relation_weight": beta,
        "candidate_profiles": CANDIDATE_PROFILES[dataset],
        "bit_summaries": {bit: {"passed": summary["passed"]} for bit, summary in bit_summaries.items()},
        "candidate_aggregate": dataset_candidate_summary,
        "current_profile_risk": dataset_candidate_summary[_current_candidate_name(dataset)]["max_overall_loss_profile_risk"],
        "recommended_stage6_dev_candidates": recommendations,
        "final_stage6_profile_selected": False,
        "stage6_parameters_modified": False,
        "forbidden_flags": FORBIDDEN_FLAGS,
        "passed": all(summary["passed"] for summary in bit_summaries.values()),
    }
    write_json(OUTPUT_ROOT / dataset / "dataset_loss_weight_sensitivity_summary.json", dataset_summary)
    (OUTPUT_ROOT / dataset / "dataset_loss_weight_sensitivity_summary.md").write_text(_format_dataset_markdown(dataset_summary), encoding="utf-8")
    return dataset_summary


def _audit_bit(
    *,
    dataset: str,
    bit: int,
    H_I_base: torch.Tensor,
    H_T_base: torch.Tensor,
    S: torch.Tensor,
    Q_I: torch.Tensor,
    Q_T: torch.Tensor,
    alpha_intra_topology: float,
    beta: int,
    candidates: list[dict[str, Any]],
    eps: float,
    block_size: int,
) -> dict[str, Any]:
    audits: dict[str, Any] = {}
    for candidate in candidates:
        audit = _audit_candidate(
            H_I_base=H_I_base,
            H_T_base=H_T_base,
            S=S,
            Q_I=Q_I,
            Q_T=Q_T,
            alpha_intra_topology=alpha_intra_topology,
            beta=beta,
            candidate=candidate,
            eps=eps,
            block_size=block_size,
        )
        audits[str(candidate["name"])] = audit

    current = audits[str(candidates[0]["name"])]
    baseline_grad_i = float(current["gradient_norm_H_I"])
    baseline_grad_t = float(current["gradient_norm_H_T"])
    if baseline_grad_i <= 0.0 or baseline_grad_t <= 0.0:
        raise RuntimeError(f"{dataset} bit={bit} current profile gradients must be nonzero")

    for audit in audits.values():
        rel_i = _safe_ratio(float(audit["gradient_norm_H_I"]), baseline_grad_i)
        rel_t = _safe_ratio(float(audit["gradient_norm_H_T"]), baseline_grad_t)
        audit["relative_grad_norm_vs_current_H_I"] = rel_i
        audit["relative_grad_norm_vs_current_H_T"] = rel_t
        audit["gradient_scale_risk"] = _classify_gradient_scale_risk(
            gradient_finite=bool(audit["gradient_finite"]),
            gradient_nonzero=bool(audit["gradient_nonzero"]),
            rel_i=rel_i,
            rel_t=rel_t,
        )
        audit["overall_loss_profile_risk"] = _max_risk(
            audit["semantic_underweight_risk"],
            audit["pair_dominance_risk"],
            audit["q_balance_risk"],
            audit["gradient_scale_risk"],
        )
        audit["passed"] = bool(
            audit["gradient_finite"]
            and audit["gradient_nonzero"]
            and audit["loss_finite_nonnegative"]
            and audit["L_total_weighted_sum_match"]
        )

    return {
        "stage": "stage5",
        "substage": "Stage 5D loss-weight sensitivity",
        "audit_version": AUDIT_VERSION,
        "dataset": dataset,
        "bit": bit,
        "beta_relation_weight": beta,
        "candidate_audits": audits,
        "current_candidate": str(candidates[0]["name"]),
        "final_stage6_profile_selected": False,
        "stage6_parameters_modified": False,
        "forbidden_flags": FORBIDDEN_FLAGS,
        "passed": all(audit["passed"] for audit in audits.values()),
    }


def _audit_candidate(
    *,
    H_I_base: torch.Tensor,
    H_T_base: torch.Tensor,
    S: torch.Tensor,
    Q_I: torch.Tensor,
    Q_T: torch.Tensor,
    alpha_intra_topology: float,
    beta: int,
    candidate: dict[str, Any],
    eps: float,
    block_size: int,
) -> dict[str, Any]:
    H_I = H_I_base.clone().requires_grad_(True)
    H_T = H_T_base.clone().requires_grad_(True)
    try:
        components = compute_total_hash_loss(
            H_I=H_I,
            H_T=H_T,
            S=S,
            beta_relation_weight=float(beta),
            alpha_intra_topology=alpha_intra_topology,
            lambda_sem_total=float(candidate["lambda_sem_total"]),
            lambda_pair_total=float(candidate["lambda_pair_total"]),
            lambda_q_total=float(candidate["lambda_q_total"]),
            lambda_bal_total=float(candidate["lambda_bal_total"]),
            eps=eps,
            relation_mode="blockwise",
            block_size=block_size,
            Q_I=Q_I,
            Q_T=Q_T,
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(f"CUDA OOM during loss-weight sensitivity audit; reduce relation_block_size={block_size}") from exc
        raise
    components.L_total.backward()
    if H_I.grad is None or H_T.grad is None:
        raise RuntimeError(f"candidate={candidate['name']} backward did not produce H gradients")

    losses = {
        "L_IT": _to_float(components.L_IT),
        "L_II": _to_float(components.L_II),
        "L_TT": _to_float(components.L_TT),
        "L_sem": _to_float(components.L_sem),
        "L_pair": _to_float(components.L_pair),
        "L_q": _to_float(components.L_q),
        "L_bal": _to_float(components.L_bal),
        "L_total": _to_float(components.L_total),
    }
    weighted_sem = float(candidate["lambda_sem_total"]) * losses["L_sem"]
    weighted_pair = float(candidate["lambda_pair_total"]) * losses["L_pair"]
    weighted_q = float(candidate["lambda_q_total"]) * losses["L_q"]
    weighted_bal = float(candidate["lambda_bal_total"]) * losses["L_bal"]
    weighted_sum = weighted_sem + weighted_pair + weighted_q + weighted_bal
    if abs(losses["L_total"] - weighted_sum) > 1e-5 + 1e-5 * max(1.0, abs(losses["L_total"])):
        weighted_sum_match = False
    else:
        weighted_sum_match = True

    sem_ratio = _safe_ratio(weighted_sem, losses["L_total"])
    pair_ratio = _safe_ratio(weighted_pair, losses["L_total"])
    q_ratio = _safe_ratio(weighted_q, losses["L_total"])
    bal_ratio = _safe_ratio(weighted_bal, losses["L_total"])
    gradient_norm_i = _to_float(torch.linalg.vector_norm(H_I.grad))
    gradient_norm_t = _to_float(torch.linalg.vector_norm(H_T.grad))
    gradient_finite = bool(torch.isfinite(H_I.grad).all().item() and torch.isfinite(H_T.grad).all().item())
    gradient_nonzero = bool(gradient_norm_i > 0.0 and gradient_norm_t > 0.0)
    all_losses_ok = all(math.isfinite(value) and value >= 0.0 for value in losses.values())

    return {
        "candidate_name": str(candidate["name"]),
        "lambda_sem_total": float(candidate["lambda_sem_total"]),
        "lambda_pair_total": float(candidate["lambda_pair_total"]),
        "lambda_q_total": float(candidate["lambda_q_total"]),
        "lambda_bal_total": float(candidate["lambda_bal_total"]),
        "beta_relation_weight": beta,
        **losses,
        "weighted_sem_component": weighted_sem,
        "weighted_pair_component": weighted_pair,
        "weighted_q_component": weighted_q,
        "weighted_bal_component": weighted_bal,
        "weighted_sem_component_ratio": sem_ratio,
        "weighted_pair_component_ratio": pair_ratio,
        "weighted_q_component_ratio": q_ratio,
        "weighted_bal_component_ratio": bal_ratio,
        "gradient_norm_H_I": gradient_norm_i,
        "gradient_norm_H_T": gradient_norm_t,
        "gradient_finite": gradient_finite,
        "gradient_nonzero": gradient_nonzero,
        "relative_grad_norm_vs_current_H_I": None,
        "relative_grad_norm_vs_current_H_T": None,
        "semantic_underweight_risk": _classify_semantic_ratio_risk(sem_ratio),
        "pair_dominance_risk": _classify_pair_ratio_risk(pair_ratio),
        "q_balance_risk": _classify_q_ratio_risk(q_ratio),
        "gradient_scale_risk": None,
        "overall_loss_profile_risk": None,
        "loss_finite_nonnegative": all_losses_ok,
        "L_total_weighted_sum_match": weighted_sum_match,
        "passed": False,
    }


def _aggregate_dataset_candidates(bit_summaries: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for candidate in candidates:
        name = str(candidate["name"])
        audits = [summary["candidate_audits"][name] for summary in bit_summaries.values()]
        risks = [audit["overall_loss_profile_risk"] for audit in audits]
        result[name] = {
            "name": name,
            "lambda_sem_total": float(candidate["lambda_sem_total"]),
            "lambda_pair_total": float(candidate["lambda_pair_total"]),
            "lambda_q_total": float(candidate["lambda_q_total"]),
            "lambda_bal_total": float(candidate["lambda_bal_total"]),
            "max_overall_loss_profile_risk": _max_risk(*risks),
            "mean_sem_ratio": _mean(audit["weighted_sem_component_ratio"] for audit in audits),
            "mean_pair_ratio": _mean(audit["weighted_pair_component_ratio"] for audit in audits),
            "mean_q_ratio": _mean(audit["weighted_q_component_ratio"] for audit in audits),
            "min_relative_grad_norm_H_I": min(float(audit["relative_grad_norm_vs_current_H_I"]) for audit in audits),
            "max_relative_grad_norm_H_I": max(float(audit["relative_grad_norm_vs_current_H_I"]) for audit in audits),
            "min_relative_grad_norm_H_T": min(float(audit["relative_grad_norm_vs_current_H_T"]) for audit in audits),
            "max_relative_grad_norm_H_T": max(float(audit["relative_grad_norm_vs_current_H_T"]) for audit in audits),
            "all_gradients_finite_nonzero": all(audit["gradient_finite"] and audit["gradient_nonzero"] for audit in audits),
            "all_bits_passed": all(audit["passed"] for audit in audits),
            "per_bit_risks": {bit: summary["candidate_audits"][name]["overall_loss_profile_risk"] for bit, summary in bit_summaries.items()},
        }
    return result


def _recommend_candidates(candidate_summary: dict[str, Any]) -> dict[str, Any]:
    ranked = sorted(candidate_summary.values(), key=_recommendation_key)
    primary = ranked[0] if ranked else None
    backup = ranked[1] if len(ranked) > 1 else None
    if primary is None:
        raise RuntimeError("no candidate profiles available for recommendation")
    low_candidates = [candidate for candidate in ranked if candidate["max_overall_loss_profile_risk"] == "low"]
    reason = "selected low-risk candidate" if low_candidates else "no low-risk candidate; selected lowest-risk medium/high candidate for dev-only testing"
    return {
        "primary_candidate": primary,
        "backup_candidate": backup,
        "selection_basis": reason,
        "dev_only": True,
        "final_training_profile_selected": False,
    }


def _recommendation_key(candidate: dict[str, Any]) -> tuple[float, float, float, float, str]:
    risk_score = float(RISK_ORDER[candidate["max_overall_loss_profile_risk"]])
    constraint_penalty = 0.0
    if candidate["mean_sem_ratio"] < 0.30:
        constraint_penalty += 0.30 - candidate["mean_sem_ratio"]
    if candidate["mean_pair_ratio"] > 0.65:
        constraint_penalty += candidate["mean_pair_ratio"] - 0.65
    if not (0.03 <= candidate["mean_q_ratio"] <= 0.15):
        constraint_penalty += min(abs(candidate["mean_q_ratio"] - 0.03), abs(candidate["mean_q_ratio"] - 0.15))
    grad_penalty = max(
        abs(candidate["min_relative_grad_norm_H_I"] - 1.0),
        abs(candidate["max_relative_grad_norm_H_I"] - 1.0),
        abs(candidate["min_relative_grad_norm_H_T"] - 1.0),
        abs(candidate["max_relative_grad_norm_H_T"] - 1.0),
    )
    balance_penalty = abs(candidate["mean_sem_ratio"] - 0.40) + abs(candidate["mean_pair_ratio"] - 0.45) + abs(candidate["mean_q_ratio"] - 0.08)
    return (risk_score, constraint_penalty, grad_penalty, balance_penalty, str(candidate["name"]))


def _build_global_summary(bits: list[int], dataset_summaries: dict[str, Any]) -> dict[str, Any]:
    recommended = {
        dataset: summary["recommended_stage6_dev_candidates"]
        for dataset, summary in dataset_summaries.items()
    }
    any_high = any(
        candidate["max_overall_loss_profile_risk"] == "high"
        for summary in dataset_summaries.values()
        for candidate in summary["candidate_aggregate"].values()
    )
    return {
        "stage": "stage5",
        "substage": "Stage 5D loss-weight sensitivity",
        "audit_version": AUDIT_VERSION,
        "datasets": DATASETS,
        "bits": bits,
        "run_scope": "all_bits" if len(bits) > 1 else "single_bit",
        "recommended_stage6_dev_candidates": recommended,
        "any_high_risk_dataset": any_high,
        "config_automatically_changed": False,
        "final_stage6_profile_selected": False,
        "stage6_parameters_modified": False,
        "training_performed": False,
        "optimizer_step_performed": False,
        "map_evaluation_performed": False,
        "forbidden_flags": FORBIDDEN_FLAGS,
        "dataset_summaries": {
            dataset: {
                "current_profile_risk": summary["current_profile_risk"],
                "primary_candidate": summary["recommended_stage6_dev_candidates"]["primary_candidate"]["name"],
                "backup_candidate": summary["recommended_stage6_dev_candidates"]["backup_candidate"]["name"]
                if summary["recommended_stage6_dev_candidates"]["backup_candidate"]
                else None,
            }
            for dataset, summary in dataset_summaries.items()
        },
        "passed": all(summary["passed"] for summary in dataset_summaries.values()),
    }


def _validate_existing_stage5_summary(dataset: str, bit: int, beta: int) -> None:
    path = REPO_ROOT / "outputs" / "stage5_loss_audit" / dataset / str(bit) / "loss_audit_summary.json"
    if not path.exists():
        raise RuntimeError(f"missing existing Stage 5 loss audit summary: {path}")
    summary = read_json(path)
    if summary.get("passed") is not True:
        raise RuntimeError(f"existing Stage 5 loss audit summary did not pass: {path}")
    if int(summary.get("default_beta", -1)) != beta:
        raise RuntimeError(f"existing Stage 5 summary beta mismatch for {dataset} bit={bit}: summary={summary.get('default_beta')} config={beta}")
    forbidden = summary.get("forbidden_flags", {})
    if forbidden != FORBIDDEN_FLAGS:
        raise RuntimeError(f"existing Stage 5 summary forbidden flags mismatch for {dataset} bit={bit}: {forbidden}")


def _semantic_path(dataset: str) -> Path:
    return REPO_ROOT / "data" / "processed" / dataset / "semantic_cache" / "se_c_s_formal_v1"


def _stage4_bit_path(dataset: str, bit: int) -> Path:
    return REPO_ROOT / "data" / "processed" / dataset / "model_cache" / "stage4_forward_v1" / str(bit)


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


def _enforce_formal_python(config: dict[str, Any]) -> None:
    expected = Path(config["runtime"]["python"]).resolve()
    current = Path(sys.executable).resolve()
    if os.path.normcase(str(expected)) != os.path.normcase(str(current)):
        raise RuntimeError(f"Stage 5D sensitivity audit requires formal Python: current={current}; expected={expected}")


def _enforce_cuda_runtime(runtime: dict[str, Any]) -> None:
    if runtime["device"] != "cuda:0":
        raise RuntimeError(f"Stage 5D sensitivity audit requires cuda:0, got {runtime['device']}")
    if runtime.get("amp_enabled") is not False:
        raise RuntimeError("Stage 5D sensitivity audit requires amp_enabled=false")
    if runtime.get("dtype") != "float32":
        raise RuntimeError(f"Stage 5D sensitivity audit requires float32, got {runtime.get('dtype')}")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 5D sensitivity audit requires cuda:0; CPU fallback is not allowed")


def _classify_semantic_ratio_risk(ratio: float) -> str:
    if 0.30 <= ratio <= 0.55:
        return "low"
    if 0.20 <= ratio < 0.30 or 0.55 < ratio <= 0.70:
        return "medium"
    return "high"


def _classify_pair_ratio_risk(ratio: float) -> str:
    if 0.25 <= ratio <= 0.65:
        return "low"
    if 0.65 < ratio <= 0.80 or 0.15 <= ratio < 0.25:
        return "medium"
    return "high"


def _classify_q_ratio_risk(ratio: float) -> str:
    if 0.03 <= ratio <= 0.15:
        return "low"
    if 0.15 < ratio <= 0.25 or 0.01 <= ratio < 0.03:
        return "medium"
    return "high"


def _classify_gradient_scale_risk(*, gradient_finite: bool, gradient_nonzero: bool, rel_i: float, rel_t: float) -> str:
    if not gradient_finite or not gradient_nonzero:
        return "high"
    if 0.25 <= rel_i <= 3.0 and 0.25 <= rel_t <= 3.0:
        return "low"
    return "high"


def _max_risk(*risks: str) -> str:
    for risk in risks:
        if risk not in RISK_ORDER:
            raise RuntimeError(f"invalid risk value: {risk}")
    return max(risks, key=lambda value: RISK_ORDER[value])


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or abs(denominator) <= 1e-12:
        raise RuntimeError(f"invalid ratio numerator={numerator}, denominator={denominator}")
    return numerator / denominator


def _to_float(value: torch.Tensor) -> float:
    result = float(value.item())
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite scalar value: {result}")
    return result


def _mean(values: Any) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        raise RuntimeError("cannot average empty values")
    result = sum(materialized) / len(materialized)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite mean: {result}")
    return result


def _current_candidate_name(dataset: str) -> str:
    if dataset == "mirflickr25k":
        return "mir_current"
    if dataset == "nuswide":
        return "nus_current"
    if dataset == "mscoco":
        return "coco_current"
    raise RuntimeError(f"unknown dataset={dataset}")


def _format_bit_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Stage 5D Loss-Weight Sensitivity: {summary['dataset']} {summary['bit']}-bit",
        "",
        f"- beta_relation_weight: {summary['beta_relation_weight']}",
        f"- current_candidate: {summary['current_candidate']}",
        f"- passed: {summary['passed']}",
        "",
        "| candidate | L_total | sem ratio | pair ratio | q ratio | grad rel I | grad rel T | overall risk |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for audit in summary["candidate_audits"].values():
        lines.append(
            f"| {audit['candidate_name']} | {audit['L_total']:.8g} | {audit['weighted_sem_component_ratio']:.8g} | "
            f"{audit['weighted_pair_component_ratio']:.8g} | {audit['weighted_q_component_ratio']:.8g} | "
            f"{audit['relative_grad_norm_vs_current_H_I']:.8g} | {audit['relative_grad_norm_vs_current_H_T']:.8g} | "
            f"{audit['overall_loss_profile_risk']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _format_dataset_markdown(summary: dict[str, Any]) -> str:
    rec = summary["recommended_stage6_dev_candidates"]
    lines = [
        f"# Stage 5D Dataset Loss-Weight Sensitivity: {summary['dataset']}",
        "",
        f"- bits: {summary['bits']}",
        f"- current_profile_risk: {summary['current_profile_risk']}",
        f"- primary_candidate: {rec['primary_candidate']['name']}",
        f"- backup_candidate: {rec['backup_candidate']['name'] if rec['backup_candidate'] else 'none'}",
        f"- selection_basis: {rec['selection_basis']}",
        "- final_training_profile_selected: false",
        "",
        "| candidate | max risk | mean sem | mean pair | mean q | grad rel I range | grad rel T range |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate in summary["candidate_aggregate"].values():
        lines.append(
            f"| {candidate['name']} | {candidate['max_overall_loss_profile_risk']} | {candidate['mean_sem_ratio']:.8g} | "
            f"{candidate['mean_pair_ratio']:.8g} | {candidate['mean_q_ratio']:.8g} | "
            f"{candidate['min_relative_grad_norm_H_I']:.8g}-{candidate['max_relative_grad_norm_H_I']:.8g} | "
            f"{candidate['min_relative_grad_norm_H_T']:.8g}-{candidate['max_relative_grad_norm_H_T']:.8g} |"
        )
    lines.append("")
    return "\n".join(lines)


def _format_global_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Stage 5D Global Loss-Weight Sensitivity",
        "",
        f"- datasets: {summary['datasets']}",
        f"- bits: {summary['bits']}",
        f"- config_automatically_changed: {summary['config_automatically_changed']}",
        f"- final_stage6_profile_selected: {summary['final_stage6_profile_selected']}",
        f"- stage6_parameters_modified: {summary['stage6_parameters_modified']}",
        f"- passed: {summary['passed']}",
        "",
        "| dataset | current risk | primary | backup |",
        "| --- | --- | --- | --- |",
    ]
    for dataset, item in summary["dataset_summaries"].items():
        lines.append(f"| {dataset} | {item['current_profile_risk']} | {item['primary_candidate']} | {item['backup_candidate']} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

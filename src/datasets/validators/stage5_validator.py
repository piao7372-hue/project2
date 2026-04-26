from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from src.utils.jsonl import read_json


SUPPORTED_STAGE5_DATASETS = {"mirflickr25k", "nuswide", "mscoco"}
STAGE5_VALIDATOR_VERSION = "stage5_loss_audit_validator_v3"
RISK_VALUES = {"low", "medium", "high"}
REQUIRED_FORBIDDEN_FLAGS = {
    "accepts_label_vector": False,
    "accepts_B": False,
    "uses_sign": False,
    "uses_numpy_loss": False,
    "detaches_H": False,
    "uses_no_grad": False,
}


def validate_stage5_loss_audit(repo_root: Path, config_path: Path, dataset: str, all_bits: bool) -> dict[str, Any]:
    config = read_json((repo_root / config_path).resolve())
    allowed_datasets = set(config.get("execution_policy", {}).get("stage5c_allowed_datasets", []))
    if allowed_datasets != SUPPORTED_STAGE5_DATASETS:
        raise RuntimeError(f"Stage 5 allowed datasets must be {sorted(SUPPORTED_STAGE5_DATASETS)}, got {sorted(allowed_datasets)}")
    if dataset not in allowed_datasets:
        raise RuntimeError(f"Stage 5 validator is authorized only for {sorted(allowed_datasets)}; got {dataset}")
    if not all_bits:
        raise RuntimeError("Stage 5 validator requires --all-bits")

    required_bits = [int(value) for value in config["hash_bits"]]
    dataset_profile = config["datasets"][dataset]
    required_betas = [int(value) for value in dataset_profile["beta_relation_weight_candidates"]]
    default_beta = int(dataset_profile["beta_relation_weight"])
    output_root = repo_root / config["outputs"]["loss_audit_root"] / dataset
    failures: list[str] = []
    if not output_root.exists():
        failures.append(f"missing Stage 5 output root: {output_root}")

    derived_summary = _load_required_json(output_root / "derived_supervision_summary.json", failures)
    _check_required_file(output_root / "derived_supervision_summary.md", failures)
    if derived_summary:
        _validate_derived_summary(derived_summary, dataset, failures)

    bit_results: dict[str, Any] = {}
    for bit in required_bits:
        bit_json = output_root / str(bit) / "loss_audit_summary.json"
        bit_md = output_root / str(bit) / "loss_audit_summary.md"
        summary = _load_required_json(bit_json, failures)
        _check_required_file(bit_md, failures)
        if summary:
            bit_results[str(bit)] = _validate_bit_summary(summary, bit, dataset, required_betas, default_beta, failures)

    aggregate_path = output_root / "stage5_loss_audit_summary.json"
    aggregate = _load_required_json(aggregate_path, failures)
    if aggregate:
        _validate_aggregate(aggregate, dataset, required_bits, required_betas, failures)

    _validate_no_data_processed_outputs(repo_root, dataset, failures)
    passed = not failures and all(result.get("passed", False) for result in bit_results.values())
    return {
        "stage": "stage5",
        "substage": "Stage 5B/5C",
        "validator_version": STAGE5_VALIDATOR_VERSION,
        "dataset": dataset,
        "hash_bits": required_bits,
        "beta_candidates": required_betas,
        "passed": passed,
        "derived_profile_norm_risk": derived_summary.get("derived_profile_norm_risk") if derived_summary else None,
        "failure_count": len(failures),
        "failure_reason": failures,
        "bits": bit_results,
    }


def _load_required_json(path: Path, failures: list[str]) -> dict[str, Any]:
    if not path.exists():
        failures.append(f"missing required JSON output: {path}")
        return {}
    try:
        return read_json(path)
    except Exception as exc:
        failures.append(f"failed to read JSON {path}: {exc}")
        return {}


def _check_required_file(path: Path, failures: list[str]) -> None:
    if not path.exists():
        failures.append(f"missing required output: {path}")
    elif path.stat().st_size <= 0:
        failures.append(f"required output is empty: {path}")


def _validate_derived_summary(summary: dict[str, Any], dataset: str, failures: list[str]) -> None:
    if summary.get("dataset") != dataset:
        failures.append(f"derived summary dataset mismatch: {summary.get('dataset')}")
    if summary.get("S_shape") != [5000, 5000]:
        failures.append(f"derived summary S_shape invalid: {summary.get('S_shape')}")
    if summary.get("S_II_star_shape") != [5000, 5000] or summary.get("S_TT_star_shape") != [5000, 5000]:
        failures.append("derived same-modal target shapes must be [5000,5000]")
    for key in [
        "S_min",
        "S_max",
        "S_mean",
        "S_std",
        "S_diag_mean",
        "S_offdiag_mean",
        "S_row_l2_norm_min",
        "S_row_l2_norm_max",
        "S_row_l2_norm_mean",
        "S_row_l2_norm_median",
        "S_col_l2_norm_min",
        "S_col_l2_norm_max",
        "S_col_l2_norm_mean",
        "S_col_l2_norm_median",
        "Q_I_row_norm_min",
        "Q_I_row_norm_max",
        "Q_T_row_norm_min",
        "Q_T_row_norm_max",
        "S_II_star_diag_mean",
        "S_II_star_offdiag_mean",
        "S_TT_star_diag_mean",
        "S_TT_star_offdiag_mean",
    ]:
        _require_finite(summary, key, failures)
    if summary.get("S_row_zero_count") != 0:
        failures.append(f"S_row_zero_count must be 0, got {summary.get('S_row_zero_count')}")
    if summary.get("S_col_zero_count") != 0:
        failures.append(f"S_col_zero_count must be 0, got {summary.get('S_col_zero_count')}")
    for key in [
        "S_row_near_zero_count_1e-8",
        "S_row_near_zero_count_1e-7",
        "S_row_near_zero_count_1e-6",
        "S_col_near_zero_count_1e-8",
        "S_col_near_zero_count_1e-7",
        "S_col_near_zero_count_1e-6",
    ]:
        _require_nonnegative_int(summary, key, failures)
    risk = summary.get("derived_profile_norm_risk")
    if risk not in RISK_VALUES:
        failures.append(f"derived_profile_norm_risk must be one of {sorted(RISK_VALUES)}, got {risk}")
    elif risk == "high":
        failures.append("derived_profile_norm_risk is high")
    if summary.get("S_II_star_range_tolerance_passed") is not True:
        failures.append("S_II_star range tolerance failed")
    if summary.get("S_TT_star_range_tolerance_passed") is not True:
        failures.append("S_TT_star range tolerance failed")
    if summary.get("derived_matrices_written") is not False:
        failures.append("derived matrices must not be written")
    if summary.get("data_processed_written") is not False:
        failures.append("Stage 5 audit must not write data/processed")


def _validate_bit_summary(
    summary: dict[str, Any],
    bit: int,
    dataset: str,
    required_betas: list[int],
    default_beta: int,
    failures: list[str],
) -> dict[str, Any]:
    local_failures: list[str] = []
    if summary.get("dataset") != dataset:
        local_failures.append(f"unexpected dataset {summary.get('dataset')}")
    if summary.get("bit") != bit:
        local_failures.append(f"unexpected bit {summary.get('bit')}")
    if summary.get("beta_candidates") != required_betas:
        local_failures.append(f"beta candidates mismatch: {summary.get('beta_candidates')}")
    if int(summary.get("default_beta", -1)) != default_beta:
        local_failures.append(f"default beta {default_beta} missing")
    if summary.get("forbidden_flags") != REQUIRED_FORBIDDEN_FLAGS:
        local_failures.append(f"forbidden flags mismatch: {summary.get('forbidden_flags')}")
    if summary.get("final_beta_selected") is not False:
        local_failures.append("Stage 5 audit must not select final beta")
    if summary.get("stage6_parameters_modified") is not False:
        local_failures.append("Stage 5 audit must not modify Stage 6 parameters")

    beta_audits = summary.get("beta_audits", {})
    for beta in required_betas:
        audit = beta_audits.get(str(beta))
        if not audit:
            local_failures.append(f"missing beta audit {beta}")
            continue
        _validate_beta_audit(audit, local_failures)

    failures.extend([f"bit {bit}: {failure}" for failure in local_failures])
    return {
        "passed": not local_failures and summary.get("passed") is True,
        "failure_count": len(local_failures),
        "failure_reason": local_failures,
    }


def _validate_beta_audit(audit: dict[str, Any], failures: list[str]) -> None:
    for key in ["L_IT", "L_II", "L_TT", "L_sem", "L_pair", "L_q", "L_bal", "L_total"]:
        _require_finite(audit, key, failures)
        if isinstance(audit.get(key), (int, float)) and audit[key] < 0.0:
            failures.append(f"{key} must be nonnegative")
    weighted_sum = (
        float(audit["weighted_sem_component"])
        + float(audit["weighted_pair_component"])
        + float(audit["weighted_q_component"])
        + float(audit["weighted_bal_component"])
    )
    if abs(float(audit["L_total"]) - weighted_sum) > 1e-5 + 1e-5 * max(1.0, abs(float(audit["L_total"]))):
        failures.append("L_total does not match weighted sum")
    if audit.get("L_total_weighted_sum_match") is not True:
        failures.append("L_total_weighted_sum_match must be true")
    if audit.get("gradient_finite") is not True:
        failures.append("gradient_finite must be true")
    if audit.get("gradient_nonzero") is not True:
        failures.append("gradient_nonzero must be true")
    for key in ["gradient_norm_H_I", "gradient_norm_H_T"]:
        _require_finite(audit, key, failures)
        if isinstance(audit.get(key), (int, float)) and audit[key] <= 0.0:
            failures.append(f"{key} must be positive")
    for prefix in ["W_IT", "W_II", "W_TT"]:
        for suffix in ["min", "max", "mean", "std", "diag_mean", "offdiag_mean", "diag_offdiag_contrast"]:
            _require_finite(audit, f"{prefix}_{suffix}", failures)
    ratios = audit.get("loss_component_ratios", {})
    for name in ["sem", "pair", "q", "bal"]:
        if name not in ratios or not math.isfinite(float(ratios[name])):
            failures.append(f"loss component ratio {name} must be finite")
    for key in [
        "weighted_sem_component_ratio",
        "weighted_pair_component_ratio",
        "weighted_q_component_ratio",
        "weighted_bal_component_ratio",
    ]:
        _require_finite(audit, key, failures)
    for key in ["pair_dominance_risk", "semantic_underweight_risk", "beta_effectiveness_risk", "loss_balance_risk"]:
        if audit.get(key) not in RISK_VALUES:
            failures.append(f"{key} must be one of {sorted(RISK_VALUES)}, got {audit.get(key)}")
    if audit.get("passed") is not True:
        failures.append("beta audit passed flag must be true")


def _validate_aggregate(
    aggregate: dict[str, Any],
    dataset: str,
    required_bits: list[int],
    required_betas: list[int],
    failures: list[str],
) -> None:
    if aggregate.get("dataset") != dataset:
        failures.append(f"aggregate dataset mismatch: {aggregate.get('dataset')}")
    if aggregate.get("hash_bits") != required_bits:
        failures.append(f"aggregate hash_bits mismatch: {aggregate.get('hash_bits')}")
    if aggregate.get("beta_candidates") != required_betas:
        failures.append(f"aggregate beta candidates mismatch: {aggregate.get('beta_candidates')}")
    if aggregate.get("forbidden_flags") != REQUIRED_FORBIDDEN_FLAGS:
        failures.append("aggregate forbidden flags mismatch")
    input_integrity = aggregate.get("input_integrity", {})
    if input_integrity.get("allowed_input_fingerprints_before_after_match") is not True:
        failures.append("Stage 5 input fingerprints changed during audit")
    if input_integrity.get("stage5_audit_modified_allowed_inputs") is not False:
        failures.append("Stage 5 audit modified allowed inputs")
    if aggregate.get("final_beta_selected") is not False:
        failures.append("aggregate must not select final beta")
    if aggregate.get("stage6_parameters_modified") is not False:
        failures.append("aggregate must not modify Stage 6 parameters")
    if aggregate.get("derived_profile_norm_risk") not in RISK_VALUES:
        failures.append(f"aggregate derived_profile_norm_risk invalid: {aggregate.get('derived_profile_norm_risk')}")


def _validate_no_data_processed_outputs(repo_root: Path, dataset: str, failures: list[str]) -> None:
    disallowed_paths = [
        repo_root / "data" / "processed" / dataset / "stage5_loss_audit",
        repo_root / "data" / "processed" / dataset / "model_cache" / "stage5_loss_audit",
        repo_root / "data" / "processed" / dataset / "semantic_cache" / "stage5_loss_audit",
    ]
    for path in disallowed_paths:
        if path.exists():
            failures.append(f"Stage 5 audit output must not be under data/processed: {path}")


def _require_finite(payload: dict[str, Any], key: str, failures: list[str]) -> None:
    if key not in payload:
        failures.append(f"missing required numeric field {key}")
        return
    try:
        value = float(payload[key])
    except (TypeError, ValueError):
        failures.append(f"{key} must be numeric, got {payload[key]}")
        return
    if not math.isfinite(value):
        failures.append(f"{key} must be finite, got {value}")


def _require_nonnegative_int(payload: dict[str, Any], key: str, failures: list[str]) -> None:
    if key not in payload:
        failures.append(f"missing required integer field {key}")
        return
    value = payload[key]
    if not isinstance(value, int):
        failures.append(f"{key} must be an integer, got {value}")
        return
    if value < 0:
        failures.append(f"{key} must be nonnegative, got {value}")

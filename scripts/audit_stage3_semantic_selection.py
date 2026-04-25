from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.jsonl import iter_jsonl, read_json, write_json
from src.semantic.semantic_relation import (
    _check_feature_cache,
    _check_stage1_stage2_inputs,
    _check_train_features,
    _compute_a_r_with_torch,
)


SUPPORTED_DATASETS = {"mirflickr25k"}
CONFIG_PATH = REPO_ROOT / "configs" / "stages" / "stage3_semantic.json"
LAMBDA_CANDIDATES = [0.60, 0.65, 0.70]
TAU_CANDIDATES = [0.02, 0.015, 0.01, 0.0075]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit MIR Stage 3 lambda x tau candidates without overwriting formal cache.")
    parser.add_argument("--dataset", required=True, choices=sorted(SUPPORTED_DATASETS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset
    config = read_json(CONFIG_PATH)
    dataset_config = config["datasets"][dataset]
    profile = config["profiles"][dataset]
    processed_root = REPO_ROOT / config["inputs"]["processed_root"] / dataset
    feature_dir = processed_root / "feature_cache" / config["outputs"]["feature_cache_dirname"]
    semantic_dir = processed_root / "semantic_cache" / config["outputs"]["semantic_cache_dirname"]
    output_dir = REPO_ROOT / "outputs" / "stage3_semantic_selection" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    formal_paths = {
        "A": semantic_dir / "A.npy",
        "R": semantic_dir / "R.npy",
        "Se": semantic_dir / "Se.npy",
        "C": semantic_dir / "C.npy",
        "S": semantic_dir / "S.npy",
        "meta": semantic_dir / "meta.json",
        "diagnostics": semantic_dir / "semantic_diagnostics.json",
    }
    input_paths = {
        "manifest_filtered": processed_root / "manifest" / "manifest_filtered.jsonl",
        "train_ids": processed_root / "splits" / "train_ids.txt",
        "order_hashes": processed_root / "reports" / "order_hashes.json",
        "stage2_meta": feature_dir / "meta.json",
        "stage2_x_i": feature_dir / "X_I.npy",
        "stage2_x_t": feature_dir / "X_T.npy",
    }
    _require_files(input_paths.values())
    before_hashes = {name: _sha256_file(path) for name, path in formal_paths.items() if path.is_file()}
    rows_manifest = list(iter_jsonl(input_paths["manifest_filtered"]))
    train_ids = _read_lines(input_paths["train_ids"])
    order_hashes = read_json(input_paths["order_hashes"])
    stage2_meta = read_json(input_paths["stage2_meta"])
    _check_stage1_stage2_inputs(rows_manifest, train_ids, order_hashes, stage2_meta, config, dataset_config, dataset)
    train_count = int(dataset_config["expected_train_count"])
    feature_dim = int(dataset_config["feature_dim"])
    topk = int(profile["topk_for_diagnostics"])
    formal_lambda = float(profile["lambda_ar_fusion"])
    formal_tau = float(profile["tau_confidence"])

    sample_ids = [str(row["sample_id"]) for row in rows_manifest]
    id_to_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    train_indices = np.asarray([id_to_index[sample_id] for sample_id in train_ids], dtype=np.int64)
    x_i_all = np.load(input_paths["stage2_x_i"], mmap_mode="r")
    x_t_all = np.load(input_paths["stage2_x_t"], mmap_mode="r")
    _check_feature_cache(x_i_all, "X_I", dataset_config)
    _check_feature_cache(x_t_all, "X_T", dataset_config)
    x_i_train = np.asarray(x_i_all[train_indices], dtype=np.float32)
    x_t_train = np.asarray(x_t_all[train_indices], dtype=np.float32)
    _check_train_features(x_i_train, "X_I_train", train_count, feature_dim, config)
    _check_train_features(x_t_train, "X_T_train", train_count, feature_dim, config)
    A_array, R_array = _compute_a_r_with_torch(x_i_train, x_t_train, str(config["runtime"].get("device", "cuda:0")))
    labels = _load_train_labels(processed_root)

    rows = []
    for lambda_ar_fusion in LAMBDA_CANDIDATES:
        Se = np.asarray(lambda_ar_fusion * A_array + (1.0 - lambda_ar_fusion) * R_array, dtype=np.float32)
        for tau_confidence in TAU_CANDIDATES:
            rows.append(_audit_candidate(Se, lambda_ar_fusion, tau_confidence, topk, labels))

    recommendation = _recommend_candidate(rows, formal_lambda, formal_tau)
    current_vs_best = _current_vs_best(rows, formal_lambda, formal_tau, recommendation)
    top5_candidates = _top_candidates(rows, limit=5)
    after_hashes = {name: _sha256_file(path) for name, path in formal_paths.items() if path.is_file()}
    formal_cache_unchanged = before_hashes == after_hashes

    summary = {
        "stage": "stage3_semantic_selection_audit",
        "dataset": dataset,
        "generated_at_utc": _utc_now(),
        "formal_semantic_cache_dir": _repo_relative(semantic_dir),
        "output_dir": _repo_relative(output_dir),
        "formal_cache_hashes_before": before_hashes,
        "formal_cache_hashes_after": after_hashes,
        "formal_cache_unchanged": formal_cache_unchanged,
        "formal_candidate": {
            "lambda_ar_fusion": formal_lambda,
            "tau_confidence": formal_tau,
            "source": "current_stage3_config_profile_recomputed_from_current_stage1_stage2",
        },
        "lambda_ar_fusion_candidates": LAMBDA_CANDIDATES,
        "tau_confidence_candidates": TAU_CANDIDATES,
        "candidate_count": len(rows),
        "selection_criteria": _selection_criteria(),
        "rows": rows,
        "top5_candidates": top5_candidates,
        "current_vs_best": current_vs_best,
        "recommendation": recommendation,
        "forbidden_transformations_used": {
            "normalize_s": False,
            "minmax_scale_s": False,
            "multiply_s_by_constant": False,
            "add_identity_to_s": False,
            "topk_mask_s": False,
        },
    }
    write_json(output_dir / "semantic_selection_summary.json", summary)
    (output_dir / "semantic_selection_summary.md").write_text(_markdown(summary), encoding="utf-8")

    print(f"dataset={dataset}")
    print(f"output_json={output_dir / 'semantic_selection_summary.json'}")
    print(f"output_md={output_dir / 'semantic_selection_summary.md'}")
    print(f"formal_cache_unchanged={str(formal_cache_unchanged).lower()}")
    print(f"recommended_lambda_ar_fusion={recommendation['recommended_lambda_ar_fusion']}")
    print(f"recommended_tau_confidence={recommendation['recommended_tau_confidence']}")
    print(f"formal_revision_needed={str(recommendation['formal_revision_needed']).lower()}")
    return 0 if formal_cache_unchanged else 1


def _audit_candidate(Se: np.ndarray, lambda_ar_fusion: float, tau_confidence: float, topk: int, labels: np.ndarray) -> dict[str, Any]:
    P_I2T = _stable_softmax(Se, tau_confidence, axis=1)
    P_T2I = _stable_softmax(Se, tau_confidence, axis=0)
    C = np.sqrt(P_I2T * P_T2I).astype(np.float32)
    S = np.asarray(C * Se, dtype=np.float32)
    if not np.isfinite(S).all():
        raise RuntimeError("candidate S contains NaN or Inf")

    diag_mean_s, offdiag_mean_s = _diag_offdiag(S)
    row_quantiles = _diag_quantiles_by_row(S)
    col_quantiles = _diag_quantiles_by_col(S)
    row_topk_cols = _row_topk_indices(S, topk)
    col_topk_rows = _col_topk_indices(S, topk)
    diag = np.arange(S.shape[0])
    diag_in_row_topk = np.any(row_topk_cols == diag[:, None], axis=1)
    diag_in_col_topk = np.any(col_topk_rows == diag[None, :], axis=0)

    row_prob = _row_normalize(S, "candidate S rows")
    col_prob = _row_normalize(S.T, "candidate S columns")
    row_effective = np.exp(_entropy(row_prob, axis=1))
    col_effective = np.exp(_entropy(col_prob, axis=1))
    top50_non_diag_mass = _topk_non_diag_mass(row_prob, 50)

    Q_I = row_prob.astype(np.float32)
    Q_T = col_prob.astype(np.float32)
    S_II = _matmul_torch(Q_I, Q_I.T)
    S_TT = _matmul_torch(Q_T, Q_T.T)
    s_ii_diag, s_ii_offdiag = _diag_offdiag(S_II)
    s_tt_diag, s_tt_offdiag = _diag_offdiag(S_TT)
    s_ii_top50 = _topk_non_diag_value_mean(S_II, 50)
    s_tt_top50 = _topk_non_diag_value_mean(S_TT, 50)
    label_metrics = _label_precision_metrics(S, labels)

    row = {
        "lambda_ar_fusion": float(lambda_ar_fusion),
        "tau_confidence": float(tau_confidence),
        "diag_mean_s": diag_mean_s,
        "offdiag_mean_s": offdiag_mean_s,
        "diag_minus_offdiag_s": float(diag_mean_s - offdiag_mean_s),
        "diag_over_offdiag_ratio": float(diag_mean_s / offdiag_mean_s) if offdiag_mean_s > 0.0 else math.inf,
        "paired_diag_quantile_in_row_median": float(np.median(row_quantiles)),
        "paired_diag_quantile_in_col_median": float(np.median(col_quantiles)),
        "diag_in_row_topk_rate": float(np.mean(diag_in_row_topk)),
        "diag_in_col_topk_rate": float(np.mean(diag_in_col_topk)),
        "row_topk_coverage": float(np.unique(row_topk_cols.reshape(-1)).size / S.shape[0]),
        "col_topk_coverage": float(np.unique(col_topk_rows.reshape(-1)).size / S.shape[0]),
        "row_effective_support_median": float(np.median(row_effective)),
        "col_effective_support_median": float(np.median(col_effective)),
        "top50_non_diag_mass_mean": float(np.mean(top50_non_diag_mass)),
        "S_II_star_offdiag_mean": s_ii_offdiag,
        "S_TT_star_offdiag_mean": s_tt_offdiag,
        "S_II_star_effective_rank_approx": _participation_ratio_effective_rank(S_II),
        "S_TT_star_effective_rank_approx": _participation_ratio_effective_rank(S_TT),
        "S_II_star_diag_mean": s_ii_diag,
        "S_TT_star_diag_mean": s_tt_diag,
        "S_II_star_top50_non_diag_mean": s_ii_top50,
        "S_TT_star_top50_non_diag_mean": s_tt_top50,
        **label_metrics,
    }
    row["passes_unsupervised_criteria"] = _passes_selection_criteria(row)
    row["selection_score"] = _selection_score(row)
    return row


def _recommend_candidate(rows: list[dict[str, Any]], formal_lambda: float, formal_tau: float) -> dict[str, Any]:
    viable = [row for row in rows if row["passes_unsupervised_criteria"]]
    formal = _candidate_for(rows, formal_lambda, formal_tau)
    if not viable:
        formal_rank = _rank_of_formal(rows, formal_lambda, formal_tau)
        return {
            "best_candidate_by_unsupervised_criteria": None,
            "current_formal_candidate_rank": formal_rank,
            "recommended_lambda_ar_fusion": formal_lambda,
            "recommended_tau_confidence": formal_tau,
            "formal_revision_needed": False,
            "reason": "No candidate passed all unsupervised compatibility criteria; keep current formal pending manual review.",
        }
    ranked = sorted(viable, key=lambda row: row["selection_score"], reverse=True)
    best = ranked[0]
    formal_rank = _rank_of_formal(rows, formal_lambda, formal_tau)
    hard_gate = _hard_gate(best, formal)
    revision_needed = abs(best["lambda_ar_fusion"] - formal_lambda) > 1e-12 or abs(best["tau_confidence"] - formal_tau) > 1e-12
    return {
        "best_candidate_by_unsupervised_criteria": {
            "lambda_ar_fusion": best["lambda_ar_fusion"],
            "tau_confidence": best["tau_confidence"],
            "selection_score": best["selection_score"],
        },
        "current_formal_candidate_rank": formal_rank,
        "hard_gate_passed": hard_gate["passed"],
        "hard_gate_checks": hard_gate["checks"],
        "recommended_lambda_ar_fusion": best["lambda_ar_fusion"],
        "recommended_tau_confidence": best["tau_confidence"],
        "formal_revision_needed": bool(revision_needed and hard_gate["passed"]),
        "reason": "Recommendation balances S separation, support, non-diagonal neighborhood mass, and derived same-modal topology; label-aware metrics are not used for scoring.",
    }


def _selection_criteria() -> dict[str, float]:
    return {
        "paired_diag_quantile_in_row_median_min": 0.99,
        "paired_diag_quantile_in_col_median_min": 0.99,
        "diag_in_row_topk_rate_min": 0.80,
        "diag_in_col_topk_rate_min": 0.80,
        "row_topk_coverage_min": 0.99,
        "col_topk_coverage_min": 0.99,
        "row_effective_support_median_min": 10.0,
        "col_effective_support_median_min": 10.0,
        "top50_non_diag_mass_mean_min": 0.01,
        "S_II_star_offdiag_mean_min": 1e-5,
        "S_TT_star_offdiag_mean_min": 1e-5,
        "S_II_star_effective_rank_approx_min": 5.0,
        "S_TT_star_effective_rank_approx_min": 5.0,
    }


def _hard_gate(candidate: dict[str, Any], formal: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "paired_diag_quantile_in_row_median": candidate["paired_diag_quantile_in_row_median"] >= 0.99,
        "paired_diag_quantile_in_col_median": candidate["paired_diag_quantile_in_col_median"] >= 0.99,
        "diag_in_row_topk_rate": candidate["diag_in_row_topk_rate"] >= 0.80,
        "diag_in_col_topk_rate": candidate["diag_in_col_topk_rate"] >= 0.80,
        "row_topk_coverage": candidate["row_topk_coverage"] >= 0.99,
        "col_topk_coverage": candidate["col_topk_coverage"] >= 0.99,
        "row_effective_support_median": candidate["row_effective_support_median"] >= 500.0,
        "col_effective_support_median": candidate["col_effective_support_median"] >= 500.0,
        "top50_non_diag_mass_mean": candidate["top50_non_diag_mass_mean"] >= 0.05,
        "S_II_star_effective_rank_approx": candidate["S_II_star_effective_rank_approx"] >= 50.0,
        "S_TT_star_effective_rank_approx": candidate["S_TT_star_effective_rank_approx"] >= 50.0,
        "S_II_star_offdiag_mean": candidate["S_II_star_offdiag_mean"] > 0.0,
        "S_TT_star_offdiag_mean": candidate["S_TT_star_offdiag_mean"] > 0.0,
        "S_top50_label_precision_i2t_drop": candidate["S_top50_label_precision_i2t"] >= formal["S_top50_label_precision_i2t"] - 0.03,
        "S_top50_label_precision_t2i_drop": candidate["S_top50_label_precision_t2i"] >= formal["S_top50_label_precision_t2i"] - 0.03,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _current_vs_best(rows: list[dict[str, Any]], formal_lambda: float, formal_tau: float, recommendation: dict[str, Any]) -> list[dict[str, Any]]:
    formal = _candidate_for(rows, formal_lambda, formal_tau)
    best = _candidate_for(rows, float(recommendation["recommended_lambda_ar_fusion"]), float(recommendation["recommended_tau_confidence"]))
    return [
        _comparison_row("current_formal", formal),
        _comparison_row("selection_best", best),
    ]


def _top_candidates(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: row["selection_score"], reverse=True)
    return [_comparison_row(f"rank_{index}", row) for index, row in enumerate(ranked[:limit], start=1)]


def _comparison_row(name: str, row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "lambda_ar_fusion",
        "tau_confidence",
        "diag_mean_s",
        "offdiag_mean_s",
        "diag_minus_offdiag_s",
        "diag_over_offdiag_ratio",
        "paired_diag_quantile_in_row_median",
        "paired_diag_quantile_in_col_median",
        "diag_in_row_topk_rate",
        "diag_in_col_topk_rate",
        "row_topk_coverage",
        "col_topk_coverage",
        "row_effective_support_median",
        "col_effective_support_median",
        "top50_non_diag_mass_mean",
        "S_II_star_offdiag_mean",
        "S_TT_star_offdiag_mean",
        "S_II_star_top50_non_diag_mean",
        "S_TT_star_top50_non_diag_mean",
        "S_II_star_effective_rank_approx",
        "S_TT_star_effective_rank_approx",
        "S_top10_label_precision_i2t",
        "S_top50_label_precision_i2t",
        "S_top10_label_precision_t2i",
        "S_top50_label_precision_t2i",
    ]
    return {"candidate_name": name, **{key: row[key] for key in keys}}


def _candidate_for(rows: list[dict[str, Any]], lambda_ar_fusion: float, tau_confidence: float) -> dict[str, Any]:
    for row in rows:
        if abs(row["lambda_ar_fusion"] - lambda_ar_fusion) < 1e-12 and abs(row["tau_confidence"] - tau_confidence) < 1e-12:
            return row
    raise RuntimeError(f"candidate not found: lambda={lambda_ar_fusion}, tau={tau_confidence}")


def _passes_selection_criteria(row: dict[str, Any]) -> bool:
    criteria = _selection_criteria()
    return (
        row["diag_mean_s"] > row["offdiag_mean_s"]
        and row["paired_diag_quantile_in_row_median"] >= criteria["paired_diag_quantile_in_row_median_min"]
        and row["paired_diag_quantile_in_col_median"] >= criteria["paired_diag_quantile_in_col_median_min"]
        and row["diag_in_row_topk_rate"] >= criteria["diag_in_row_topk_rate_min"]
        and row["diag_in_col_topk_rate"] >= criteria["diag_in_col_topk_rate_min"]
        and row["row_topk_coverage"] >= criteria["row_topk_coverage_min"]
        and row["col_topk_coverage"] >= criteria["col_topk_coverage_min"]
        and row["row_effective_support_median"] >= criteria["row_effective_support_median_min"]
        and row["col_effective_support_median"] >= criteria["col_effective_support_median_min"]
        and row["top50_non_diag_mass_mean"] >= criteria["top50_non_diag_mass_mean_min"]
        and row["S_II_star_offdiag_mean"] >= criteria["S_II_star_offdiag_mean_min"]
        and row["S_TT_star_offdiag_mean"] >= criteria["S_TT_star_offdiag_mean_min"]
        and row["S_II_star_effective_rank_approx"] >= criteria["S_II_star_effective_rank_approx_min"]
        and row["S_TT_star_effective_rank_approx"] >= criteria["S_TT_star_effective_rank_approx_min"]
    )


def _selection_score(row: dict[str, Any]) -> float:
    if not row["passes_unsupervised_criteria"]:
        return -math.inf
    return float(
        math.log1p(row["diag_over_offdiag_ratio"])
        + 2.0 * row["top50_non_diag_mass_mean"]
        + 0.001 * row["row_effective_support_median"]
        + 0.001 * row["col_effective_support_median"]
        + 0.01 * row["S_II_star_effective_rank_approx"]
        + 0.01 * row["S_TT_star_effective_rank_approx"]
    )


def _rank_of_formal(rows: list[dict[str, Any]], formal_lambda: float, formal_tau: float) -> int | None:
    ranked = sorted(rows, key=lambda row: row["selection_score"], reverse=True)
    for index, row in enumerate(ranked, start=1):
        if abs(row["lambda_ar_fusion"] - formal_lambda) < 1e-12 and abs(row["tau_confidence"] - formal_tau) < 1e-12:
            return index
    return None


def _markdown(summary: dict[str, Any]) -> str:
    rec = summary["recommendation"]
    lines = [
        "# Stage 3A MIR Semantic Selection Audit",
        "",
        f"- Dataset: `{summary['dataset']}`",
        f"- Formal cache unchanged: `{summary['formal_cache_unchanged']}`",
        f"- Recommended lambda: `{rec['recommended_lambda_ar_fusion']}`",
        f"- Recommended tau: `{rec['recommended_tau_confidence']}`",
        f"- Formal revision needed: `{rec['formal_revision_needed']}`",
        "",
        "## Candidate Table",
        "",
        "| lambda | tau | pass | score | diag/offdiag | row q med | col q med | row support med | col support med | top50 non-diag mass | SII offdiag | STT offdiag | SII rank | STT rank |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['lambda_ar_fusion']} | {row['tau_confidence']} | {row['passes_unsupervised_criteria']} | "
            f"{row['selection_score']} | {row['diag_over_offdiag_ratio']} | "
            f"{row['paired_diag_quantile_in_row_median']} | {row['paired_diag_quantile_in_col_median']} | "
            f"{row['row_effective_support_median']} | {row['col_effective_support_median']} | "
            f"{row['top50_non_diag_mass_mean']} | {row['S_II_star_offdiag_mean']} | "
            f"{row['S_TT_star_offdiag_mean']} | {row['S_II_star_effective_rank_approx']} | "
            f"{row['S_TT_star_effective_rank_approx']} |"
        )
    lines.extend(["", "## Recommendation", "", rec["reason"], ""])
    lines.extend(
        [
            "## Current vs Best",
            "",
            "| candidate | lambda | tau | diag/offdiag | row support med | col support med | top50 non-diag mass | SII rank | STT rank | I2T top50 label | T2I top50 label |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["current_vs_best"]:
        lines.append(
            f"| {row['candidate_name']} | {row['lambda_ar_fusion']} | {row['tau_confidence']} | "
            f"{row['diag_over_offdiag_ratio']} | {row['row_effective_support_median']} | "
            f"{row['col_effective_support_median']} | {row['top50_non_diag_mass_mean']} | "
            f"{row['S_II_star_effective_rank_approx']} | {row['S_TT_star_effective_rank_approx']} | "
            f"{row['S_top50_label_precision_i2t']} | {row['S_top50_label_precision_t2i']} |"
        )
    lines.extend(
        [
            "",
            "## Top 5 Candidates",
            "",
            "| rank | lambda | tau | score | diag/offdiag | row support med | col support med | top50 non-diag mass | SII rank | STT rank |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["top5_candidates"]:
        rank = row["candidate_name"].replace("rank_", "")
        source = _candidate_for(summary["rows"], row["lambda_ar_fusion"], row["tau_confidence"])
        lines.append(
            f"| {rank} | {row['lambda_ar_fusion']} | {row['tau_confidence']} | "
            f"{source['selection_score']} | {row['diag_over_offdiag_ratio']} | "
            f"{row['row_effective_support_median']} | {row['col_effective_support_median']} | "
            f"{row['top50_non_diag_mass_mean']} | {row['S_II_star_effective_rank_approx']} | "
            f"{row['S_TT_star_effective_rank_approx']} |"
        )
    return "\n".join(lines)


def _stable_softmax(matrix: np.ndarray, tau: float, axis: int) -> np.ndarray:
    z = np.asarray(matrix, dtype=np.float64) / float(tau)
    z -= np.max(z, axis=axis, keepdims=True)
    exp_z = np.exp(z)
    denom = np.sum(exp_z, axis=axis, keepdims=True)
    if not np.isfinite(exp_z).all() or not np.isfinite(denom).all() or np.any(denom <= 0.0):
        raise RuntimeError("stable softmax denominator is non-positive or non-finite")
    result = exp_z / denom
    if not np.isfinite(result).all() or np.any(result <= 0.0) or np.any(result > 1.0):
        raise RuntimeError("stable softmax produced invalid probabilities")
    return result


def _row_normalize(matrix: np.ndarray, name: str) -> np.ndarray:
    sums = np.sum(matrix, axis=1, keepdims=True, dtype=np.float64)
    if not np.isfinite(sums).all() or np.any(sums <= 0.0):
        raise RuntimeError(f"{name} row sum is non-positive or non-finite")
    return np.asarray(matrix, dtype=np.float64) / sums


def _entropy(prob: np.ndarray, axis: int) -> np.ndarray:
    return -np.sum(prob * np.log(np.maximum(prob, np.finfo(np.float64).tiny)), axis=axis)


def _diag_quantiles_by_row(matrix: np.ndarray) -> np.ndarray:
    diag = np.diag(matrix)[:, None]
    return np.mean(matrix <= diag, axis=1, dtype=np.float64)


def _diag_quantiles_by_col(matrix: np.ndarray) -> np.ndarray:
    diag = np.diag(matrix)[None, :]
    return np.mean(matrix <= diag, axis=0, dtype=np.float64)


def _row_topk_indices(matrix: np.ndarray, topk: int) -> np.ndarray:
    indices = np.argpartition(matrix, -topk, axis=1)[:, -topk:]
    scores = np.take_along_axis(matrix, indices, axis=1)
    order = np.argsort(-scores, axis=1)
    return np.take_along_axis(indices, order, axis=1).astype(np.int64)


def _col_topk_indices(matrix: np.ndarray, topk: int) -> np.ndarray:
    indices = np.argpartition(matrix, -topk, axis=0)[-topk:, :]
    scores = np.take_along_axis(matrix, indices, axis=0)
    order = np.argsort(-scores, axis=0)
    return np.take_along_axis(indices, order, axis=0).astype(np.int64)


def _topk_non_diag_mass(prob: np.ndarray, topk: int) -> np.ndarray:
    masked = prob.copy()
    np.fill_diagonal(masked, -np.inf)
    values = np.partition(masked, -topk, axis=1)[:, -topk:]
    return np.sum(values, axis=1)


def _topk_non_diag_value_mean(matrix: np.ndarray, topk: int) -> float:
    masked = np.asarray(matrix, dtype=np.float64).copy()
    np.fill_diagonal(masked, -np.inf)
    values = np.partition(masked, -topk, axis=1)[:, -topk:]
    return float(np.mean(values))


def _label_precision_metrics(S: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "S_top10_label_precision_i2t": _label_precision_i2t(S, labels, 10),
        "S_top50_label_precision_i2t": _label_precision_i2t(S, labels, 50),
        "S_top10_label_precision_t2i": _label_precision_t2i(S, labels, 10),
        "S_top50_label_precision_t2i": _label_precision_t2i(S, labels, 50),
    }


def _label_precision_i2t(S: np.ndarray, labels: np.ndarray, topk: int) -> float:
    indices = _row_topk_indices(S, topk)
    relevant = np.any(labels[:, None, :] & labels[indices], axis=2)
    return float(np.mean(relevant))


def _label_precision_t2i(S: np.ndarray, labels: np.ndarray, topk: int) -> float:
    indices = _col_topk_indices(S, topk)
    relevant = np.any(labels[None, :, :] & labels[indices], axis=2)
    return float(np.mean(relevant))


def _load_train_labels(processed_root: Path) -> np.ndarray:
    rows = list(iter_jsonl(processed_root / "manifest" / "manifest_filtered.jsonl"))
    train_ids = _read_lines(processed_root / "splits" / "train_ids.txt")
    by_id = {str(row["sample_id"]): row for row in rows}
    labels = []
    for sample_id in train_ids:
        row = by_id.get(sample_id)
        if row is None:
            raise RuntimeError(f"train id not present in manifest_filtered: {sample_id}")
        labels.append([int(value) for value in row["label_vector"]])
    return np.asarray(labels, dtype=bool)


def _read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle if line.rstrip("\n")]


def _matmul_torch(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        left_t = torch.from_numpy(np.ascontiguousarray(left, dtype=np.float32)).to(device)
        right_t = torch.from_numpy(np.ascontiguousarray(right, dtype=np.float32)).to(device)
        result = (left_t @ right_t).detach().cpu().numpy().astype(np.float32, copy=False)
        del left_t, right_t
    return result


def _diag_offdiag(matrix: np.ndarray) -> tuple[float, float]:
    n = matrix.shape[0]
    diag_sum = float(np.trace(matrix, dtype=np.float64))
    total_sum = float(np.sum(matrix, dtype=np.float64))
    return diag_sum / n, (total_sum - diag_sum) / (n * (n - 1))


def _participation_ratio_effective_rank(matrix: np.ndarray) -> float:
    trace = float(np.trace(matrix, dtype=np.float64))
    fro_sq = float(np.sum(np.asarray(matrix, dtype=np.float64) ** 2))
    if fro_sq <= 0.0:
        return 0.0
    return (trace * trace) / fro_sq


def _check_matrix(matrix: np.ndarray, name: str, train_count: int) -> None:
    if matrix.shape != (train_count, train_count):
        raise RuntimeError(f"{name} shape mismatch: got {matrix.shape}")
    if matrix.dtype != np.float32:
        raise RuntimeError(f"{name} dtype mismatch: got {matrix.dtype}")
    if not np.isfinite(matrix).all():
        raise RuntimeError(f"{name} contains NaN or Inf")


def _require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing required files: {missing}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())

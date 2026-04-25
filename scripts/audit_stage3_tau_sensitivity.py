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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.jsonl import read_json, write_json


TAU_CANDIDATES = [0.07, 0.05, 0.03, 0.02, 0.015, 0.01]
FORMAL_TAU = 0.07
SUPPORTED_DATASETS = {"mirflickr25k"}
CONFIG_PATH = REPO_ROOT / "configs" / "stages" / "stage3_semantic.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Stage 3 tau sensitivity without overwriting formal semantic cache.")
    parser.add_argument("--dataset", required=True, choices=sorted(SUPPORTED_DATASETS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_json(CONFIG_PATH)
    dataset = args.dataset
    dataset_root = REPO_ROOT / config["inputs"]["processed_root"] / dataset
    semantic_dir = dataset_root / "semantic_cache" / config["outputs"]["semantic_cache_dirname"]
    output_dir = REPO_ROOT / "outputs" / "stage3_tau_sensitivity" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    formal_files = {
        "A": semantic_dir / "A.npy",
        "R": semantic_dir / "R.npy",
        "Se": semantic_dir / "Se.npy",
        "C": semantic_dir / "C.npy",
        "S": semantic_dir / "S.npy",
        "meta": semantic_dir / "meta.json",
        "diagnostics": semantic_dir / "semantic_diagnostics.json",
        "validator_summary": semantic_dir / "validator_summary.json",
        "omega": semantic_dir / "Omega_topk_diag.npz",
    }
    _require_files(formal_files.values())
    before_hashes = {name: _sha256_file(path) for name, path in formal_files.items()}

    meta = read_json(formal_files["meta"])
    diagnostics = read_json(formal_files["diagnostics"])
    validator = read_json(formal_files["validator_summary"])
    profile = config["profiles"][dataset]
    topk = int(profile["topk_for_diagnostics"])
    train_count = int(meta["train_count"])
    uniform = 1.0 / train_count

    Se = np.load(formal_files["Se"], mmap_mode="r")
    C_formal = np.load(formal_files["C"], mmap_mode="r")
    S_formal = np.load(formal_files["S"], mmap_mode="r")
    _check_matrix(Se, "Se", train_count)
    _check_matrix(C_formal, "C", train_count)
    _check_matrix(S_formal, "S", train_count)

    formal_diag_offdiag = {
        "se": _diag_offdiag_summary(Se),
        "c": _diag_offdiag_summary(C_formal),
        "s": _diag_offdiag_summary(S_formal),
    }

    rows = []
    for tau in TAU_CANDIDATES:
        rows.append(_audit_tau(Se, tau, topk, uniform))

    recommendation = _recommend_tau(rows, diagnostics)
    after_hashes = {name: _sha256_file(path) for name, path in formal_files.items()}
    formal_cache_unchanged = before_hashes == after_hashes

    summary = {
        "stage": "stage3_tau_sensitivity_audit",
        "dataset": dataset,
        "generated_at_utc": _utc_now(),
        "formal_semantic_cache_dir": _repo_relative(semantic_dir),
        "output_dir": _repo_relative(output_dir),
        "formal_cache_unchanged": formal_cache_unchanged,
        "formal_cache_hashes_before": before_hashes,
        "formal_cache_hashes_after": after_hashes,
        "tau_candidates": TAU_CANDIDATES,
        "formal_tau_confidence": FORMAL_TAU,
        "uniform_confidence": uniform,
        "train_count": train_count,
        "topk_for_diagnostics": topk,
        "formal_validator_passed": bool(validator.get("passed")),
        "formal_diagnostics_reference": {
            "diag_mean_s": diagnostics.get("diag_mean_s"),
            "offdiag_mean_s": diagnostics.get("offdiag_mean_s"),
            "diag_minus_offdiag_s": diagnostics.get("diag_minus_offdiag_s"),
            "paired_diag_quantile_in_row_median": diagnostics.get("paired_diag_quantile_in_row_median"),
            "paired_diag_quantile_in_col_median": diagnostics.get("paired_diag_quantile_in_col_median"),
            "diag_in_row_topk_rate": diagnostics.get("diag_in_row_topk_rate"),
            "diag_in_col_topk_rate": diagnostics.get("diag_in_col_topk_rate"),
            "row_topk_coverage": diagnostics.get("row_topk_coverage"),
            "col_topk_coverage": diagnostics.get("col_topk_coverage"),
        },
        "formal_tau_0_07_diag_offdiag": formal_diag_offdiag,
        "rows": rows,
        "recommendation": recommendation,
        "forbidden_transformations_used": {
            "normalize_s": False,
            "minmax_scale_s": False,
            "multiply_s_by_constant": False,
            "add_identity_to_s": False,
            "topk_mask_s": False,
        },
    }
    write_json(output_dir / "tau_sensitivity_summary.json", summary)
    (output_dir / "tau_sensitivity_summary.md").write_text(_markdown(summary), encoding="utf-8")

    print(f"dataset={dataset}")
    print(f"output_json={output_dir / 'tau_sensitivity_summary.json'}")
    print(f"output_md={output_dir / 'tau_sensitivity_summary.md'}")
    print(f"formal_cache_unchanged={str(formal_cache_unchanged).lower()}")
    print(f"recommended_tau={recommendation['recommended_tau']}")
    print(f"recommendation={recommendation['decision']}")
    return 0 if formal_cache_unchanged else 1


def _audit_tau(Se: np.ndarray, tau: float, topk: int, uniform: float) -> dict[str, Any]:
    if tau <= 0.0:
        raise RuntimeError("tau_confidence must be positive")
    P_I2T = _stable_softmax(np.asarray(Se), tau, axis=1)
    P_T2I = _stable_softmax(np.asarray(Se), tau, axis=0)
    C = np.sqrt(P_I2T * P_T2I)
    if not np.isfinite(C).all() or np.any(C <= 0.0):
        raise RuntimeError(f"C invalid for tau={tau}")
    S = C * np.asarray(Se, dtype=np.float64)
    if not np.isfinite(S).all():
        raise RuntimeError(f"S invalid for tau={tau}")

    c_diag_mean = _diag_mean(C)
    diag_mean_s, offdiag_mean_s = _diag_offdiag(S)
    row_quantiles = _diag_quantiles_by_row(S)
    col_quantiles = _diag_quantiles_by_col(S)
    row_topk_cols = _row_topk_indices(S, topk)
    col_topk_rows = _col_topk_indices(S, topk)
    diag = np.arange(S.shape[0])
    diag_in_row_topk = np.any(row_topk_cols == diag[:, None], axis=1)
    diag_in_col_topk = np.any(col_topk_rows == diag[None, :], axis=0)

    row_entropy = -np.sum(P_I2T * np.log(np.maximum(P_I2T, np.finfo(np.float64).tiny)), axis=1)
    col_entropy = -np.sum(P_T2I * np.log(np.maximum(P_T2I, np.finfo(np.float64).tiny)), axis=0)

    result = {
        "tau_confidence": float(tau),
        "uniform_confidence": float(uniform),
        "c_min": float(np.min(C)),
        "c_max": float(np.max(C)),
        "c_mean": float(np.mean(C)),
        "c_std": float(np.std(C)),
        "c_diag_mean": c_diag_mean,
        "c_max_over_uniform": float(np.max(C) / uniform),
        "c_mean_over_uniform": float(np.mean(C) / uniform),
        "c_diag_mean_over_uniform": float(c_diag_mean / uniform),
        "s_min": float(np.min(S)),
        "s_max": float(np.max(S)),
        "s_mean": float(np.mean(S)),
        "s_std": float(np.std(S)),
        "diag_mean_s": diag_mean_s,
        "offdiag_mean_s": offdiag_mean_s,
        "diag_minus_offdiag_s": float(diag_mean_s - offdiag_mean_s),
        "diag_over_offdiag_ratio": float(diag_mean_s / offdiag_mean_s) if offdiag_mean_s > 0.0 else math.inf,
        "paired_diag_quantile_in_row_mean": float(np.mean(row_quantiles)),
        "paired_diag_quantile_in_row_median": float(np.median(row_quantiles)),
        "paired_diag_quantile_in_col_mean": float(np.mean(col_quantiles)),
        "paired_diag_quantile_in_col_median": float(np.median(col_quantiles)),
        "row_topk_coverage": float(np.unique(row_topk_cols.reshape(-1)).size / S.shape[0]),
        "col_topk_coverage": float(np.unique(col_topk_rows.reshape(-1)).size / S.shape[0]),
        "diag_in_row_topk_rate": float(np.mean(diag_in_row_topk)),
        "diag_in_col_topk_rate": float(np.mean(diag_in_col_topk)),
        "row_entropy_mean": float(np.mean(row_entropy)),
        "col_entropy_mean": float(np.mean(col_entropy)),
        "row_max_c_mean": float(np.mean(np.max(C, axis=1))),
        "col_max_c_mean": float(np.mean(np.max(C, axis=0))),
    }
    return result


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


def _recommend_tau(rows: list[dict[str, Any]], formal: dict[str, Any]) -> dict[str, Any]:
    formal_row = next(row for row in rows if abs(float(row["tau_confidence"]) - FORMAL_TAU) < 1e-12)
    min_row_median = max(0.99, float(formal_row["paired_diag_quantile_in_row_median"]) - 0.005)
    min_col_median = max(0.99, float(formal_row["paired_diag_quantile_in_col_median"]) - 0.005)
    min_row_topk = max(0.0, float(formal_row["diag_in_row_topk_rate"]) - 0.03)
    min_col_topk = max(0.0, float(formal_row["diag_in_col_topk_rate"]) - 0.03)
    min_row_coverage = max(0.95, float(formal_row["row_topk_coverage"]) - 0.02)
    min_col_coverage = max(0.95, float(formal_row["col_topk_coverage"]) - 0.02)
    viable = []
    for row in rows:
        if row["diag_mean_s"] <= row["offdiag_mean_s"]:
            continue
        if row["paired_diag_quantile_in_row_median"] < min_row_median:
            continue
        if row["paired_diag_quantile_in_col_median"] < min_col_median:
            continue
        if row["diag_in_row_topk_rate"] < min_row_topk:
            continue
        if row["diag_in_col_topk_rate"] < min_col_topk:
            continue
        if row["row_topk_coverage"] < min_row_coverage:
            continue
        if row["col_topk_coverage"] < min_col_coverage:
            continue
        viable.append(row)
    if not viable:
        return {
            "decision": "keep_formal_tau_for_now",
            "recommended_tau": FORMAL_TAU,
            "reason": "No lower tau improves S scale without violating paired diagonal, top-k, or coverage guardrails.",
            "formal_config_change_required": False,
            "formal_rerun_required": False,
        }
    best = max(viable, key=lambda row: (row["diag_mean_s"], row["diag_minus_offdiag_s"]))
    change = abs(float(best["tau_confidence"]) - FORMAL_TAU) > 1e-12
    return {
        "decision": "adjust_tau_candidate_available" if change else "keep_formal_tau_for_now",
        "recommended_tau": float(best["tau_confidence"]),
        "reason": "Selected the viable tau with the largest diag_mean_s while preserving diagonal ranking and top-k coverage guardrails.",
        "formal_config_change_required": change,
        "formal_rerun_required": change,
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Stage 3A MIR Tau Sensitivity Audit",
        "",
        f"- Dataset: `{summary['dataset']}`",
        f"- Formal semantic cache unchanged: `{summary['formal_cache_unchanged']}`",
        f"- Uniform confidence: `{summary['uniform_confidence']}`",
        f"- Recommended tau: `{summary['recommendation']['recommended_tau']}`",
        f"- Decision: `{summary['recommendation']['decision']}`",
        "",
        "## Formal tau=0.07 diag/offdiag",
        "",
        "| Matrix | diag mean | offdiag mean | diag - offdiag | diag / offdiag |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, value in summary["formal_tau_0_07_diag_offdiag"].items():
        lines.append(
            f"| {key} | {value['diag_mean']} | {value['offdiag_mean']} | "
            f"{value['diag_minus_offdiag']} | {value['diag_over_offdiag_ratio']} |"
        )
    lines.extend(
        [
            "",
            "## Tau table",
            "",
            "| tau | c_diag/uniform | diag_mean_s | offdiag_mean_s | diag-offdiag | diag/offdiag | row q median | col q median | row diag top-k | col diag top-k | row coverage | col coverage |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["rows"]:
        lines.append(
            f"| {row['tau_confidence']} | {row['c_diag_mean_over_uniform']} | "
            f"{row['diag_mean_s']} | {row['offdiag_mean_s']} | {row['diag_minus_offdiag_s']} | "
            f"{row['diag_over_offdiag_ratio']} | {row['paired_diag_quantile_in_row_median']} | "
            f"{row['paired_diag_quantile_in_col_median']} | {row['diag_in_row_topk_rate']} | "
            f"{row['diag_in_col_topk_rate']} | {row['row_topk_coverage']} | {row['col_topk_coverage']} |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            summary["recommendation"]["reason"],
            "",
            "Forbidden S transformations used: none.",
            "",
        ]
    )
    return "\n".join(lines)


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


def _diag_mean(matrix: np.ndarray) -> float:
    return float(np.mean(np.diag(matrix), dtype=np.float64))


def _diag_offdiag(matrix: np.ndarray) -> tuple[float, float]:
    n = matrix.shape[0]
    diag_sum = float(np.trace(matrix, dtype=np.float64))
    total_sum = float(np.sum(matrix, dtype=np.float64))
    return diag_sum / n, (total_sum - diag_sum) / (n * (n - 1))


def _diag_offdiag_summary(matrix: np.ndarray) -> dict[str, float]:
    diag_mean, offdiag_mean = _diag_offdiag(matrix)
    return {
        "diag_mean": diag_mean,
        "offdiag_mean": offdiag_mean,
        "diag_minus_offdiag": diag_mean - offdiag_mean,
        "diag_over_offdiag_ratio": diag_mean / offdiag_mean if offdiag_mean > 0.0 else math.inf,
    }


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
        raise RuntimeError(f"Missing required formal files: {missing}")


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

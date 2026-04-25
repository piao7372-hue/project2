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


SUPPORTED_DATASETS = {"mirflickr25k"}
CONFIG_PATH = REPO_ROOT / "configs" / "stages" / "stage3_semantic.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Stage 3 formal S compatibility with Stage 5 derived supervision.")
    parser.add_argument("--dataset", required=True, choices=sorted(SUPPORTED_DATASETS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset
    config = read_json(CONFIG_PATH)
    processed_root = REPO_ROOT / config["inputs"]["processed_root"] / dataset
    semantic_dir = processed_root / "semantic_cache" / config["outputs"]["semantic_cache_dirname"]
    output_dir = REPO_ROOT / "outputs" / "stage3_semantic_compatibility" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    formal_paths = {
        "S": semantic_dir / "S.npy",
        "meta": semantic_dir / "meta.json",
        "diagnostics": semantic_dir / "semantic_diagnostics.json",
    }
    _require_files(formal_paths.values())
    s_hash_before = _sha256_file(formal_paths["S"])

    meta = read_json(formal_paths["meta"])
    diagnostics = read_json(formal_paths["diagnostics"])
    train_count = int(meta["train_count"])
    S = np.load(formal_paths["S"], mmap_mode="r")
    _check_matrix(S, "S", train_count)
    S_array = np.asarray(S, dtype=np.float32)

    sharpness = _sharpness_metrics(S_array)
    derived = _derived_supervision_metrics(S_array)
    label_sanity = _label_sanity_metrics(S_array, processed_root)
    conclusion = _compatibility_conclusion(sharpness, derived, diagnostics)

    s_hash_after = _sha256_file(formal_paths["S"])
    formal_s_unchanged = s_hash_before == s_hash_after

    summary = {
        "stage": "stage3_semantic_compatibility_audit",
        "dataset": dataset,
        "generated_at_utc": _utc_now(),
        "formal_semantic_cache_dir": _repo_relative(semantic_dir),
        "output_dir": _repo_relative(output_dir),
        "formal_inputs_read": {
            "S": _repo_relative(formal_paths["S"]),
            "meta": _repo_relative(formal_paths["meta"]),
            "semantic_diagnostics": _repo_relative(formal_paths["diagnostics"]),
        },
        "formal_outputs_written": [],
        "formal_s_hash_before": s_hash_before,
        "formal_s_hash_after": s_hash_after,
        "formal_s_unchanged": formal_s_unchanged,
        "formal_a_r_se_c_opened": False,
        "formal_cache_written": False,
        "meta": {
            "semantic_set_id": meta.get("semantic_set_id"),
            "feature_set_id": meta.get("feature_set_id"),
            "train_count": meta.get("train_count"),
            "lambda_ar_fusion": meta.get("lambda_ar_fusion"),
            "tau_confidence": meta.get("tau_confidence"),
            "omega_topk_diag_role": meta.get("omega_topk_diag_role"),
        },
        "formal_diagnostics_reference": {
            "diag_mean_s": diagnostics.get("diag_mean_s"),
            "offdiag_mean_s": diagnostics.get("offdiag_mean_s"),
            "diag_over_offdiag_ratio": diagnostics.get("diag_over_offdiag_ratio"),
            "paired_diag_quantile_in_row_median": diagnostics.get("paired_diag_quantile_in_row_median"),
            "paired_diag_quantile_in_col_median": diagnostics.get("paired_diag_quantile_in_col_median"),
            "row_topk_coverage": diagnostics.get("row_topk_coverage"),
            "col_topk_coverage": diagnostics.get("col_topk_coverage"),
        },
        "sharpness_support": sharpness,
        "derived_supervision": derived,
        "label_aware_sanity": label_sanity,
        "conclusion": conclusion,
        "diagnostic_only": {
            "label_aware_sanity_used_for_training_supervision": False,
            "derived_supervision_written_to_formal_cache": False,
        },
    }
    write_json(output_dir / "semantic_compatibility_summary.json", summary)
    (output_dir / "semantic_compatibility_summary.md").write_text(_markdown(summary), encoding="utf-8")

    print(f"dataset={dataset}")
    print(f"output_json={output_dir / 'semantic_compatibility_summary.json'}")
    print(f"output_md={output_dir / 'semantic_compatibility_summary.md'}")
    print(f"formal_s_unchanged={str(formal_s_unchanged).lower()}")
    print(f"current_S_good_for_stage4_and_stage5={summary['conclusion']['current_S_good_for_stage4_and_stage5']}")
    print(f"risk_level={summary['conclusion']['risk_level']}")
    return 0 if formal_s_unchanged else 1


def _sharpness_metrics(S: np.ndarray) -> dict[str, Any]:
    row_prob = _row_normalize(S, "S rows")
    col_prob = _row_normalize(S.T, "S columns")
    row_entropy = _entropy(row_prob, axis=1)
    col_entropy = _entropy(col_prob, axis=1)
    row_effective = np.exp(row_entropy)
    col_effective = np.exp(col_entropy)
    diag = np.diag(S).astype(np.float64)
    row_sum = np.sum(S, axis=1, dtype=np.float64)
    diag_mass = diag / row_sum
    top1_mass = _topk_mass(row_prob, 1)
    top5_mass = _topk_mass(row_prob, 5)
    top10_mass = _topk_mass(row_prob, 10)
    top50_mass = _topk_mass(row_prob, 50)
    top50_non_diag_mass = _topk_non_diag_mass(row_prob, 50)
    return {
        "row_entropy_mean": float(np.mean(row_entropy)),
        "row_entropy_median": float(np.median(row_entropy)),
        "col_entropy_mean": float(np.mean(col_entropy)),
        "col_entropy_median": float(np.median(col_entropy)),
        "row_effective_support_mean": float(np.mean(row_effective)),
        "row_effective_support_median": float(np.median(row_effective)),
        "col_effective_support_mean": float(np.mean(col_effective)),
        "col_effective_support_median": float(np.median(col_effective)),
        "diag_mass_fraction_mean": float(np.mean(diag_mass)),
        "diag_mass_fraction_median": float(np.median(diag_mass)),
        "top1_mass_mean": float(np.mean(top1_mass)),
        "top5_mass_mean": float(np.mean(top5_mass)),
        "top10_mass_mean": float(np.mean(top10_mass)),
        "top50_mass_mean": float(np.mean(top50_mass)),
        "top50_non_diag_mass_mean": float(np.mean(top50_non_diag_mass)),
    }


def _derived_supervision_metrics(S: np.ndarray) -> dict[str, Any]:
    Q_I = _row_normalize(S, "Q_I").astype(np.float32)
    Q_T = _row_normalize(S.T, "Q_T").astype(np.float32)
    S_II = _matmul_torch(Q_I, Q_I.T)
    S_TT = _matmul_torch(Q_T, Q_T.T)
    s_ii_diag, s_ii_offdiag = _diag_offdiag(S_II)
    s_tt_diag, s_tt_offdiag = _diag_offdiag(S_TT)
    s_ii_top10 = _topk_non_diag_value_mean(S_II, 10)
    s_ii_top50 = _topk_non_diag_value_mean(S_II, 50)
    s_tt_top10 = _topk_non_diag_value_mean(S_TT, 10)
    s_tt_top50 = _topk_non_diag_value_mean(S_TT, 50)
    return {
        "method": "Q_I=row_normalize(S), Q_T=row_normalize(S.T), S_II_star=Q_I@Q_I.T, S_TT_star=Q_T@Q_T.T",
        "effective_rank_approx_method": "participation_ratio_for_psd_matrix_trace_squared_over_frobenius_squared",
        "S_II_star_diag_mean": s_ii_diag,
        "S_II_star_offdiag_mean": s_ii_offdiag,
        "S_II_star_diag_over_offdiag": float(s_ii_diag / s_ii_offdiag) if s_ii_offdiag > 0.0 else math.inf,
        "S_TT_star_diag_mean": s_tt_diag,
        "S_TT_star_offdiag_mean": s_tt_offdiag,
        "S_TT_star_diag_over_offdiag": float(s_tt_diag / s_tt_offdiag) if s_tt_offdiag > 0.0 else math.inf,
        "S_II_star_top10_non_diag_mean": s_ii_top10,
        "S_II_star_top50_non_diag_mean": s_ii_top50,
        "S_TT_star_top10_non_diag_mean": s_tt_top10,
        "S_TT_star_top50_non_diag_mean": s_tt_top50,
        "S_II_star_effective_rank_approx": _participation_ratio_effective_rank(S_II),
        "S_TT_star_effective_rank_approx": _participation_ratio_effective_rank(S_TT),
    }


def _label_sanity_metrics(S: np.ndarray, processed_root: Path) -> dict[str, Any]:
    manifest_path = processed_root / "manifest" / "manifest_filtered.jsonl"
    train_ids_path = processed_root / "splits" / "train_ids.txt"
    _require_files([manifest_path, train_ids_path])
    rows = list(iter_jsonl(manifest_path))
    train_ids = _read_lines(train_ids_path)
    id_to_row = {str(row["sample_id"]): row for row in rows}
    labels = []
    for sample_id in train_ids:
        row = id_to_row.get(sample_id)
        if row is None:
            raise RuntimeError(f"train id not present in manifest_filtered: {sample_id}")
        labels.append([int(value) for value in row["label_vector"]])
    label_matrix = np.asarray(labels, dtype=bool)
    return {
        "computed": True,
        "diagnostic_only": True,
        "relevance_definition": "dot(label_vector_i, label_vector_j) > 0",
        "S_top10_label_precision_i2t": _label_precision_i2t(S, label_matrix, 10),
        "S_top50_label_precision_i2t": _label_precision_i2t(S, label_matrix, 50),
        "S_top10_label_precision_t2i": _label_precision_t2i(S, label_matrix, 10),
        "S_top50_label_precision_t2i": _label_precision_t2i(S, label_matrix, 50),
    }


def _compatibility_conclusion(sharpness: dict[str, Any], derived: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    row_support = float(sharpness["row_effective_support_median"])
    col_support = float(sharpness["col_effective_support_median"])
    top50_non_diag = float(sharpness["top50_non_diag_mass_mean"])
    s_ii_offdiag = float(derived["S_II_star_offdiag_mean"])
    s_tt_offdiag = float(derived["S_TT_star_offdiag_mean"])
    s_ii_rank = float(derived["S_II_star_effective_rank_approx"])
    s_tt_rank = float(derived["S_TT_star_effective_rank_approx"])
    diag_ratio = float(diagnostics.get("diag_over_offdiag_ratio", 0.0))

    if s_ii_offdiag < 1e-6 or s_tt_offdiag < 1e-6 or s_ii_rank < 5.0 or s_tt_rank < 5.0:
        return {
            "current_S_good_for_stage4_and_stage5": False,
            "risk_level": "high",
            "main_risk": "Derived S_II_star/S_TT_star is close to identity or has too little effective spectral support.",
            "recommendation": "Do not enter Stage 3B/4 until Stage 3A supervision compatibility is revised.",
        }
    if row_support < 10.0 or col_support < 10.0 or top50_non_diag < 0.01:
        return {
            "current_S_good_for_stage4_and_stage5": "uncertain",
            "risk_level": "medium",
            "main_risk": "S has strong diagonal structure but limited non-diagonal neighborhood mass.",
            "recommendation": "Keep current S as Stage 3A formal candidate, but review Stage 5 weighting and neighborhood behavior before Stage 4/5 training.",
        }
    if diag_ratio > 10.0 and s_ii_offdiag > 1e-5 and s_tt_offdiag > 1e-5:
        return {
            "current_S_good_for_stage4_and_stage5": True,
            "risk_level": "low",
            "main_risk": "No immediate compatibility blocker found; monitor Stage 5 loss scale because S remains sparse in mass despite nontrivial support.",
            "recommendation": "Stage 3A S is compatible with Stage 4/5 from this audit; keep diagnostic reports and do not modify formal S without a new review.",
        }
    return {
        "current_S_good_for_stage4_and_stage5": "uncertain",
        "risk_level": "medium",
        "main_risk": "Metrics are mixed and need manual review.",
        "recommendation": "Pause before Stage 3B/4 and review compatibility report.",
    }


def _markdown(summary: dict[str, Any]) -> str:
    sharp = summary["sharpness_support"]
    derived = summary["derived_supervision"]
    label = summary["label_aware_sanity"]
    conclusion = summary["conclusion"]
    lines = [
        "# Stage 3A MIR Semantic Compatibility Audit",
        "",
        f"- Dataset: `{summary['dataset']}`",
        f"- Formal S unchanged: `{summary['formal_s_unchanged']}`",
        f"- Risk level: `{conclusion['risk_level']}`",
        f"- Current S good for Stage 4/5: `{conclusion['current_S_good_for_stage4_and_stage5']}`",
        "",
        "## S Sharpness And Support",
        "",
        f"- row_effective_support_mean: `{sharp['row_effective_support_mean']}`",
        f"- row_effective_support_median: `{sharp['row_effective_support_median']}`",
        f"- col_effective_support_mean: `{sharp['col_effective_support_mean']}`",
        f"- col_effective_support_median: `{sharp['col_effective_support_median']}`",
        f"- diag_mass_fraction_mean: `{sharp['diag_mass_fraction_mean']}`",
        f"- top50_non_diag_mass_mean: `{sharp['top50_non_diag_mass_mean']}`",
        "",
        "## Derived Supervision",
        "",
        f"- S_II_star_diag_mean: `{derived['S_II_star_diag_mean']}`",
        f"- S_II_star_offdiag_mean: `{derived['S_II_star_offdiag_mean']}`",
        f"- S_II_star_effective_rank_approx: `{derived['S_II_star_effective_rank_approx']}`",
        f"- S_TT_star_diag_mean: `{derived['S_TT_star_diag_mean']}`",
        f"- S_TT_star_offdiag_mean: `{derived['S_TT_star_offdiag_mean']}`",
        f"- S_TT_star_effective_rank_approx: `{derived['S_TT_star_effective_rank_approx']}`",
        "",
        "## Label-Aware Sanity",
        "",
        f"- S_top10_label_precision_i2t: `{label['S_top10_label_precision_i2t']}`",
        f"- S_top50_label_precision_i2t: `{label['S_top50_label_precision_i2t']}`",
        f"- S_top10_label_precision_t2i: `{label['S_top10_label_precision_t2i']}`",
        f"- S_top50_label_precision_t2i: `{label['S_top50_label_precision_t2i']}`",
        "",
        "## Conclusion",
        "",
        f"- main_risk: {conclusion['main_risk']}",
        f"- recommendation: {conclusion['recommendation']}",
        "",
    ]
    return "\n".join(lines)


def _row_normalize(matrix: np.ndarray, name: str) -> np.ndarray:
    sums = np.sum(matrix, axis=1, keepdims=True, dtype=np.float64)
    if not np.isfinite(sums).all() or np.any(sums <= 0.0):
        raise RuntimeError(f"{name} row sum is non-positive or non-finite")
    return np.asarray(matrix, dtype=np.float64) / sums


def _entropy(prob: np.ndarray, axis: int) -> np.ndarray:
    return -np.sum(prob * np.log(np.maximum(prob, np.finfo(np.float64).tiny)), axis=axis)


def _topk_mass(prob: np.ndarray, topk: int) -> np.ndarray:
    values = np.partition(prob, -topk, axis=1)[:, -topk:]
    return np.sum(values, axis=1)


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


def _label_precision_i2t(S: np.ndarray, labels: np.ndarray, topk: int) -> float:
    indices = _topk_indices(S, topk, axis=1)
    relevant = _label_relevance(labels, indices, direction="i2t")
    return float(np.mean(relevant))


def _label_precision_t2i(S: np.ndarray, labels: np.ndarray, topk: int) -> float:
    indices = _topk_indices(S, topk, axis=0)
    relevant = _label_relevance(labels, indices, direction="t2i")
    return float(np.mean(relevant))


def _topk_indices(matrix: np.ndarray, topk: int, axis: int) -> np.ndarray:
    if axis == 1:
        indices = np.argpartition(matrix, -topk, axis=1)[:, -topk:]
        scores = np.take_along_axis(matrix, indices, axis=1)
        order = np.argsort(-scores, axis=1)
        return np.take_along_axis(indices, order, axis=1).astype(np.int64)
    indices = np.argpartition(matrix, -topk, axis=0)[-topk:, :]
    scores = np.take_along_axis(matrix, indices, axis=0)
    order = np.argsort(-scores, axis=0)
    return np.take_along_axis(indices, order, axis=0).astype(np.int64)


def _label_relevance(labels: np.ndarray, indices: np.ndarray, direction: str) -> np.ndarray:
    if direction == "i2t":
        query_labels = labels[:, None, :]
        retrieved_labels = labels[indices]
        return np.any(query_labels & retrieved_labels, axis=2)
    query_labels = labels[None, :, :]
    retrieved_labels = labels[indices]
    return np.any(query_labels & retrieved_labels, axis=2)


def _read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle if line.rstrip("\n")]


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

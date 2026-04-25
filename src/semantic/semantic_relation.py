from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from src.utils.jsonl import iter_jsonl, read_json, write_json


SUPPORTED_STAGE3_DATASETS = {"mirflickr25k", "nuswide", "mscoco"}
STAGE3_BUILDER_VERSION = "stage3_se_c_s_builder_v1"


def run_stage3_semantic(repo_root: Path, config_path: Path, dataset: str) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = read_json(_resolve_repo_path(repo_root, config_path))
    if dataset not in SUPPORTED_STAGE3_DATASETS or dataset not in config["datasets"]:
        raise ValueError("Stage 3 supports mirflickr25k, nuswide, and mscoco only")

    dataset_config = config["datasets"][dataset]
    profile = config["profiles"][dataset]
    lambda_ar_fusion = float(profile["lambda_ar_fusion"])
    tau_confidence = float(profile["tau_confidence"])
    topk = int(profile["topk_for_diagnostics"])
    if not (0.0 <= lambda_ar_fusion <= 1.0):
        raise RuntimeError("lambda_ar_fusion must be in [0, 1]")
    if tau_confidence <= 0.0:
        raise RuntimeError("tau_confidence must be > 0")
    if topk <= 0:
        raise RuntimeError("topk_for_diagnostics must be positive")

    processed_root = _resolve_repo_path(repo_root, Path(config["inputs"]["processed_root"])) / dataset
    paths = _stage3_paths(processed_root, config)
    output_dir = paths["semantic_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = list(iter_jsonl(paths["manifest_filtered"]))
    train_ids = _read_lines(paths["train_ids"])
    order_hashes = read_json(paths["order_hashes"])
    stage2_meta = read_json(paths["stage2_meta"])
    _check_stage1_stage2_inputs(rows, train_ids, order_hashes, stage2_meta, config, dataset_config, dataset)

    sample_ids = [str(row["sample_id"]) for row in rows]
    id_to_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    train_indices = np.asarray([id_to_index[sample_id] for sample_id in train_ids], dtype=np.int64)
    expected_train = int(dataset_config["expected_train_count"])
    feature_dim = int(dataset_config["feature_dim"])

    x_i_all = np.load(paths["x_i"], mmap_mode="r")
    x_t_all = np.load(paths["x_t"], mmap_mode="r")
    _check_feature_cache(x_i_all, "X_I", dataset_config)
    _check_feature_cache(x_t_all, "X_T", dataset_config)
    x_i_train = np.asarray(x_i_all[train_indices], dtype=np.float32)
    x_t_train = np.asarray(x_t_all[train_indices], dtype=np.float32)
    _check_train_features(x_i_train, "X_I_train", expected_train, feature_dim, config)
    _check_train_features(x_t_train, "X_T_train", expected_train, feature_dim, config)

    device = str(config["runtime"].get("device", "cuda:0"))
    A, R = _compute_a_r_with_torch(x_i_train, x_t_train, device)
    _check_unit_interval(A, "A", config)
    np.save(paths["a"], A)
    _check_unit_interval(R, "R", config)
    np.save(paths["r"], R)

    Se = np.asarray(lambda_ar_fusion * A + (1.0 - lambda_ar_fusion) * R, dtype=np.float32)
    _check_unit_interval(Se, "Se", config)
    np.save(paths["se"], Se)

    P_I2T = _stable_softmax(Se, tau_confidence, axis=1, name="P_I2T")
    P_T2I = _stable_softmax(Se, tau_confidence, axis=0, name="P_T2I")
    C = np.sqrt(P_I2T * P_T2I).astype(np.float32)
    del P_I2T, P_T2I
    _check_c_interval(C, config)
    np.save(paths["c"], C)

    S = np.asarray(C * Se, dtype=np.float32)
    _check_unit_interval(S, "S", config)
    np.save(paths["s"], S)

    diagnostics = _semantic_diagnostics(A, R, Se, C, S, topk, config)
    _write_omega(paths["omega"], S, topk)
    diagnostics["semantic_validator_passed"] = _core_stop_go_passed(diagnostics, config)
    diagnostics["failure_reason"] = None if diagnostics["semantic_validator_passed"] else _core_stop_go_failures(diagnostics, config)
    write_json(paths["diagnostics"], diagnostics)

    meta = {
        "builder_version": STAGE3_BUILDER_VERSION,
        "dataset": dataset,
        "semantic_set_id": config["semantic_set_id"],
        "feature_set_id": config["feature_set_id"],
        "train_count": expected_train,
        "matrix_shape": [expected_train, expected_train],
        "dtype": "float32",
        "lambda_ar_fusion": lambda_ar_fusion,
        "tau_confidence": tau_confidence,
        "topk_for_diagnostics": topk,
        "stage1_manifest_filtered_order_sha256": order_hashes["manifest_filtered_order_sha256"],
        "stage1_sample_id_order_sha256": order_hashes["sample_id_order_sha256"],
        "stage1_train_ids_sha256": order_hashes["train_ids_sha256"],
        "stage2_manifest_filtered_order_sha256": stage2_meta["manifest_filtered_order_sha256"],
        "stage2_feature_set_id": stage2_meta["feature_set_id"],
        "train_ids_sha256": hash_lines(train_ids),
        "train_sample_id_order_sha256": hash_lines(train_ids),
        "train_indices_sha256": hash_lines(str(int(index)) for index in train_indices),
        "x_i_train_shape": list(x_i_train.shape),
        "x_t_train_shape": list(x_t_train.shape),
        "generated_at_utc": _utc_now(),
        "formal_input_files": {
            "stage3_config": _repo_relative(repo_root, _resolve_repo_path(repo_root, config_path)),
            "stage2_x_i": _repo_relative(repo_root, paths["x_i"]),
            "stage2_x_t": _repo_relative(repo_root, paths["x_t"]),
            "stage2_meta": _repo_relative(repo_root, paths["stage2_meta"]),
            "stage1_manifest_filtered": _repo_relative(repo_root, paths["manifest_filtered"]),
            "stage1_train_ids": _repo_relative(repo_root, paths["train_ids"]),
            "stage1_order_hashes": _repo_relative(repo_root, paths["order_hashes"]),
        },
        "train_index_mapping_method": "train_ids sample_id -> manifest_filtered row index -> Stage 2 feature row",
        "s_formula": "S = C * Se",
        "s_post_normalization_used": False,
        "s_minmax_scale_used": False,
        "s_topk_mask_used": False,
        "s_identity_boost_used": False,
        "omega_topk_diag_role": "diagnostic_only_not_training_supervision",
        "softmax_numeric_stability": {
            "stable_softmax": True,
            "subtract_row_or_col_max": True,
            "internal_dtype": "float64",
            "artificial_confidence_scaling": False,
            "clamp_or_rescale_confidence": False,
            "epsilon_corrections": [],
        },
        "matrix_multiply_backend": {
            "backend": "torch",
            "device": device,
            "reason": "avoid Windows NumPy BLAS fatal exception while preserving exact Stage 3 formulas",
            "model_loaded": False,
            "training_used": False,
        },
    }
    write_json(paths["meta"], meta)

    summary = {
        "stage": "stage3",
        "substage": f"stage3a_{dataset}" if dataset == "mirflickr25k" else f"stage3_{dataset}",
        "dataset": dataset,
        "semantic_set_id": config["semantic_set_id"],
        "semantic_cache_dir": str(output_dir),
        "train_count": expected_train,
        "matrix_shape": [expected_train, expected_train],
        "lambda_ar_fusion": lambda_ar_fusion,
        "tau_confidence": tau_confidence,
        "topk_for_diagnostics": topk,
        "a_shape": list(A.shape),
        "r_shape": list(R.shape),
        "se_shape": list(Se.shape),
        "c_shape": list(C.shape),
        "s_shape": list(S.shape),
        "a_dtype": str(A.dtype),
        "r_dtype": str(R.dtype),
        "se_dtype": str(Se.dtype),
        "c_dtype": str(C.dtype),
        "s_dtype": str(S.dtype),
        "diagnostics": diagnostics,
    }
    return summary


def _stage3_paths(processed_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    feature_dir = processed_root / "feature_cache" / config["outputs"]["feature_cache_dirname"]
    semantic_dir = processed_root / "semantic_cache" / config["outputs"]["semantic_cache_dirname"]
    return {
        "manifest_filtered": processed_root / "manifest" / "manifest_filtered.jsonl",
        "train_ids": processed_root / "splits" / "train_ids.txt",
        "order_hashes": processed_root / "reports" / "order_hashes.json",
        "x_i": feature_dir / "X_I.npy",
        "x_t": feature_dir / "X_T.npy",
        "stage2_meta": feature_dir / "meta.json",
        "semantic_dir": semantic_dir,
        "a": semantic_dir / "A.npy",
        "r": semantic_dir / "R.npy",
        "se": semantic_dir / "Se.npy",
        "c": semantic_dir / "C.npy",
        "s": semantic_dir / "S.npy",
        "meta": semantic_dir / "meta.json",
        "diagnostics": semantic_dir / "semantic_diagnostics.json",
        "omega": semantic_dir / "Omega_topk_diag.npz",
        "validator_summary": semantic_dir / "validator_summary.json",
    }


def _check_stage1_stage2_inputs(
    rows: list[dict[str, Any]],
    train_ids: list[str],
    order_hashes: dict[str, Any],
    stage2_meta: dict[str, Any],
    config: dict[str, Any],
    dataset_config: dict[str, Any],
    dataset: str,
) -> None:
    expected_filtered = int(dataset_config["expected_filtered_count"])
    expected_train = int(dataset_config["expected_train_count"])
    if len(rows) != expected_filtered:
        raise RuntimeError(f"manifest_filtered count mismatch: expected {expected_filtered}, got {len(rows)}")
    if len(train_ids) != expected_train:
        raise RuntimeError(f"train_ids count mismatch: expected {expected_train}, got {len(train_ids)}")
    if len(set(train_ids)) != len(train_ids):
        raise RuntimeError("train_ids contains duplicates")
    sample_ids = [str(row.get("sample_id")) for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError("manifest_filtered sample_id is not unique")
    missing = sorted(set(train_ids) - set(sample_ids))
    if missing:
        raise RuntimeError(f"train_ids contains ids outside manifest_filtered: {missing[:5]}")
    expected_hashes = {
        "sample_id_order_sha256": hash_lines(sorted(sample_ids)),
        "manifest_filtered_order_sha256": hash_lines(sample_ids),
        "train_ids_sha256": hash_lines(train_ids),
    }
    for key, value in expected_hashes.items():
        if order_hashes.get(key) != value:
            raise RuntimeError(f"Stage 1 order_hashes mismatch for {key}")
    if stage2_meta.get("dataset") != dataset:
        raise RuntimeError("Stage 2 meta.dataset mismatch")
    if stage2_meta.get("feature_set_id") != config["feature_set_id"]:
        raise RuntimeError("Stage 2 meta.feature_set_id mismatch")
    if stage2_meta.get("manifest_filtered_order_sha256") != order_hashes.get("manifest_filtered_order_sha256"):
        raise RuntimeError("Stage 2 manifest hash does not match Stage 1 order_hashes")
    if stage2_meta.get("train_ids_sha256") != order_hashes.get("train_ids_sha256"):
        raise RuntimeError("Stage 2 train_ids hash does not match Stage 1 order_hashes")


def _check_feature_cache(array: np.ndarray, name: str, dataset_config: dict[str, Any]) -> None:
    expected_shape = (int(dataset_config["expected_filtered_count"]), int(dataset_config["feature_dim"]))
    if array.shape != expected_shape:
        raise RuntimeError(f"{name} shape mismatch: expected {expected_shape}, got {array.shape}")
    if array.dtype != np.float32:
        raise RuntimeError(f"{name} dtype mismatch: expected float32, got {array.dtype}")


def _check_train_features(array: np.ndarray, name: str, expected_train: int, feature_dim: int, config: dict[str, Any]) -> None:
    if array.shape != (expected_train, feature_dim):
        raise RuntimeError(f"{name} shape mismatch")
    if array.dtype != np.float32:
        raise RuntimeError(f"{name} dtype mismatch: expected float32, got {array.dtype}")
    if not np.isfinite(array).all():
        raise RuntimeError(f"{name} contains NaN or Inf")
    norms = np.linalg.norm(array, axis=1)
    tol = float(config["validation"]["norm_tolerance"])
    if not np.allclose(norms, 1.0, rtol=tol, atol=tol):
        raise RuntimeError(f"{name} rows are not L2 normalized")


def _compute_a_r_with_torch(x_i_train: np.ndarray, x_t_train: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    if device != "cuda:0" or not torch.cuda.is_available():
        raise RuntimeError("Stage 3 formal matrix construction requires cuda:0 for torch matmul")
    with torch.no_grad():
        x_i = torch.from_numpy(np.ascontiguousarray(x_i_train, dtype=np.float32)).to(device)
        x_t = torch.from_numpy(np.ascontiguousarray(x_t_train, dtype=np.float32)).to(device)

        A_t = (x_i @ x_t.T + 1.0) * 0.5
        A = A_t.detach().cpu().numpy().astype(np.float32, copy=False)
        del A_t

        M_I = x_i @ x_i.T
        M_T = x_t @ x_t.T
        M_I_norm = torch.linalg.norm(M_I, dim=1, keepdim=True)
        M_T_norm = torch.linalg.norm(M_T, dim=1, keepdim=True)
        if not torch.isfinite(M_I_norm).all() or not torch.isfinite(M_T_norm).all():
            raise RuntimeError("Stage 3 row-normalization norms contain NaN or Inf")
        if torch.any(M_I_norm <= 0.0) or torch.any(M_T_norm <= 0.0):
            raise RuntimeError("Stage 3 row-normalization norm is non-positive")
        M_I = M_I / M_I_norm
        M_T = M_T / M_T_norm
        R_t = (M_I @ M_T.T + 1.0) * 0.5
        R = R_t.detach().cpu().numpy().astype(np.float32, copy=False)
        del x_i, x_t, M_I, M_T, M_I_norm, M_T_norm, R_t
    return A, R


def _stable_softmax(matrix: np.ndarray, tau: float, axis: int, name: str) -> np.ndarray:
    z = np.asarray(matrix, dtype=np.float64) / float(tau)
    z = z - np.max(z, axis=axis, keepdims=True)
    exp_z = np.exp(z)
    denom = np.sum(exp_z, axis=axis, keepdims=True)
    if not np.isfinite(exp_z).all() or not np.isfinite(denom).all() or np.any(denom <= 0.0):
        raise RuntimeError(f"{name} stable softmax denominator is non-positive or non-finite")
    result = exp_z / denom
    if not np.isfinite(result).all() or np.any(result <= 0.0) or np.any(result > 1.0):
        raise RuntimeError(f"{name} stable softmax produced invalid probabilities")
    return result


def _check_unit_interval(array: np.ndarray, name: str, config: dict[str, Any]) -> None:
    tol = float(config["validation"]["range_tolerance"])
    if not np.isfinite(array).all():
        raise RuntimeError(f"{name} contains NaN or Inf")
    if float(array.min()) < -tol or float(array.max()) > 1.0 + tol:
        raise RuntimeError(f"{name} is outside [0, 1]")


def _check_c_interval(array: np.ndarray, config: dict[str, Any]) -> None:
    tol = float(config["validation"]["range_tolerance"])
    if not np.isfinite(array).all():
        raise RuntimeError("C contains NaN or Inf")
    if float(array.min()) <= 0.0 or float(array.max()) > 1.0 + tol:
        raise RuntimeError("C is outside (0, 1]")


def _semantic_diagnostics(
    A: np.ndarray,
    R: np.ndarray,
    Se: np.ndarray,
    C: np.ndarray,
    S: np.ndarray,
    topk: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    row_quantiles = _diag_quantiles_by_row(S)
    col_quantiles = _diag_quantiles_by_col(S)
    row_topk_cols = _row_topk_indices(S, topk)
    col_topk_rows = _col_topk_indices(S, topk)
    diag = np.arange(S.shape[0])
    diag_in_row_topk = np.any(row_topk_cols == diag[:, None], axis=1)
    diag_in_col_topk = np.any(col_topk_rows == diag[None, :], axis=0)
    row_topk_coverage = float(np.unique(row_topk_cols.reshape(-1)).size / S.shape[0])
    col_topk_coverage = float(np.unique(col_topk_rows.reshape(-1)).size / S.shape[0])

    diag_s, offdiag_s = _diag_offdiag_means(S)
    diagnostics: dict[str, Any] = {
        "train_count": int(S.shape[0]),
        "shape_ok": list(S.shape) == [5000, 5000],
        "range_a_ok": _range_ok(A, False, config),
        "range_r_ok": _range_ok(R, False, config),
        "range_se_ok": _range_ok(Se, False, config),
        "range_c_ok": _range_ok(C, True, config),
        "range_s_ok": _range_ok(S, False, config),
        "topk_for_diagnostics": int(topk),
        "row_topk_coverage": row_topk_coverage,
        "col_topk_coverage": col_topk_coverage,
        "diag_in_row_topk_rate": float(np.mean(diag_in_row_topk)),
        "diag_in_col_topk_rate": float(np.mean(diag_in_col_topk)),
        "paired_diag_quantile_in_row": float(np.mean(row_quantiles)),
        "paired_diag_quantile_in_col": float(np.mean(col_quantiles)),
        "paired_diag_quantile_in_row_mean": float(np.mean(row_quantiles)),
        "paired_diag_quantile_in_row_median": float(np.median(row_quantiles)),
        "paired_diag_quantile_in_col_mean": float(np.mean(col_quantiles)),
        "paired_diag_quantile_in_col_median": float(np.median(col_quantiles)),
        "diag_mean_s": diag_s,
        "offdiag_mean_s": offdiag_s,
        "diag_minus_offdiag_s": diag_s - offdiag_s,
        "diag_over_offdiag_ratio": diag_s / offdiag_s if offdiag_s > 0.0 else float("inf"),
    }
    for prefix, matrix in (("a", A), ("r", R), ("se", Se), ("c", C), ("s", S)):
        diagnostics.update(_matrix_stats(prefix, matrix))
    for prefix, matrix in (("a", A), ("r", R), ("se", Se), ("s", S)):
        diag_mean, offdiag_mean = _diag_offdiag_means(matrix)
        diagnostics[f"diag_mean_{prefix}"] = diag_mean
        diagnostics[f"offdiag_mean_{prefix}"] = offdiag_mean
    return diagnostics


def _matrix_stats(prefix: str, matrix: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_min": float(np.min(matrix)),
        f"{prefix}_max": float(np.max(matrix)),
        f"{prefix}_mean": float(np.mean(matrix, dtype=np.float64)),
        f"{prefix}_std": float(np.std(matrix, dtype=np.float64)),
    }


def _diag_offdiag_means(matrix: np.ndarray) -> tuple[float, float]:
    n = matrix.shape[0]
    diag_sum = float(np.trace(matrix, dtype=np.float64))
    total_sum = float(np.sum(matrix, dtype=np.float64))
    return diag_sum / n, (total_sum - diag_sum) / (n * (n - 1))


def _diag_quantiles_by_row(matrix: np.ndarray) -> np.ndarray:
    diag = np.diag(matrix)[:, None]
    return np.mean(matrix <= diag, axis=1, dtype=np.float64)


def _diag_quantiles_by_col(matrix: np.ndarray) -> np.ndarray:
    diag = np.diag(matrix)[None, :]
    return np.mean(matrix <= diag, axis=0, dtype=np.float64)


def _row_topk_indices(matrix: np.ndarray, topk: int) -> np.ndarray:
    if matrix.shape[1] < topk:
        raise RuntimeError("topk_for_diagnostics exceeds matrix width")
    indices = np.argpartition(matrix, -topk, axis=1)[:, -topk:]
    scores = np.take_along_axis(matrix, indices, axis=1)
    order = np.argsort(-scores, axis=1)
    return np.take_along_axis(indices, order, axis=1).astype(np.int64)


def _col_topk_indices(matrix: np.ndarray, topk: int) -> np.ndarray:
    if matrix.shape[0] < topk:
        raise RuntimeError("topk_for_diagnostics exceeds matrix height")
    indices = np.argpartition(matrix, -topk, axis=0)[-topk:, :]
    scores = np.take_along_axis(matrix, indices, axis=0)
    order = np.argsort(-scores, axis=0)
    return np.take_along_axis(indices, order, axis=0).astype(np.int64)


def _write_omega(path: Path, S: np.ndarray, topk: int) -> None:
    row_topk_cols = _row_topk_indices(S, topk)
    row_topk_rows = np.repeat(np.arange(S.shape[0], dtype=np.int64), topk)
    row_topk_cols_flat = row_topk_cols.reshape(-1).astype(np.int64)

    col_topk_rows = _col_topk_indices(S, topk)
    col_topk_rows_flat = col_topk_rows.reshape(-1).astype(np.int64)
    col_topk_cols = np.tile(np.arange(S.shape[1], dtype=np.int64), topk)

    diag = np.arange(S.shape[0], dtype=np.int64)
    pairs = np.concatenate(
        [
            np.stack([row_topk_rows, row_topk_cols_flat], axis=1),
            np.stack([col_topk_rows_flat, col_topk_cols], axis=1),
            np.stack([diag, diag], axis=1),
        ],
        axis=0,
    )
    omega = np.unique(pairs, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        omega_rows=omega[:, 0].astype(np.int64),
        omega_cols=omega[:, 1].astype(np.int64),
        row_topk_rows=row_topk_rows.astype(np.int64),
        row_topk_cols=row_topk_cols_flat.astype(np.int64),
        col_topk_rows=col_topk_rows_flat.astype(np.int64),
        col_topk_cols=col_topk_cols.astype(np.int64),
        diag_rows=diag,
        diag_cols=diag,
        topk_for_diagnostics=np.asarray([topk], dtype=np.int64),
        matrix_shape=np.asarray(S.shape, dtype=np.int64),
        diagnostic_only=np.asarray([1], dtype=np.int8),
    )


def _core_stop_go_passed(diagnostics: dict[str, Any], config: dict[str, Any]) -> bool:
    return not _core_stop_go_failures(diagnostics, config)


def _core_stop_go_failures(diagnostics: dict[str, Any], config: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    required_true = ["shape_ok", "range_a_ok", "range_r_ok", "range_se_ok", "range_c_ok", "range_s_ok"]
    for key in required_true:
        if diagnostics.get(key) is not True:
            failures.append(key)
    if diagnostics.get("diag_mean_s", 0.0) <= diagnostics.get("offdiag_mean_s", 0.0):
        failures.append("diag_mean_s must be greater than offdiag_mean_s")
    if diagnostics.get("diag_minus_offdiag_s", 0.0) <= float(config["validation"]["min_diag_minus_offdiag_s"]):
        failures.append("diag_minus_offdiag_s must be positive")
    if diagnostics.get("diag_over_offdiag_ratio", 0.0) < float(config["validation"]["min_diag_over_offdiag_ratio"]):
        failures.append("diag_over_offdiag_ratio below threshold")
    if diagnostics.get("row_topk_coverage", 0.0) < float(config["validation"]["min_row_topk_coverage"]):
        failures.append("row_topk_coverage below threshold")
    if diagnostics.get("col_topk_coverage", 0.0) < float(config["validation"]["min_col_topk_coverage"]):
        failures.append("col_topk_coverage below threshold")
    if diagnostics.get("diag_in_row_topk_rate", 0.0) < float(config["validation"]["min_diag_in_row_topk_rate"]):
        failures.append("diag_in_row_topk_rate below threshold")
    if diagnostics.get("diag_in_col_topk_rate", 0.0) < float(config["validation"]["min_diag_in_col_topk_rate"]):
        failures.append("diag_in_col_topk_rate below threshold")
    minimum_quantile = float(config["validation"]["min_paired_diag_quantile_median"])
    if diagnostics.get("paired_diag_quantile_in_row_median", 0.0) < minimum_quantile:
        failures.append("paired_diag_quantile_in_row_median below threshold")
    if diagnostics.get("paired_diag_quantile_in_col_median", 0.0) < minimum_quantile:
        failures.append("paired_diag_quantile_in_col_median below threshold")
    return failures


def _range_ok(array: np.ndarray, positive: bool, config: dict[str, Any]) -> bool:
    tol = float(config["validation"]["range_tolerance"])
    if not np.isfinite(array).all():
        return False
    lower_ok = float(array.min()) > 0.0 if positive else float(array.min()) >= -tol
    return lower_ok and float(array.max()) <= 1.0 + tol


def _read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle if line.rstrip("\n")]


def hash_lines(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(str(line).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _resolve_repo_path(repo_root: Path, path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
    resolved.relative_to(repo_root)
    return resolved


def _repo_relative(repo_root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

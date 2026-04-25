from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.semantic.semantic_relation import _semantic_diagnostics
from src.utils.jsonl import iter_jsonl, read_json, write_json


SUPPORTED_STAGE3_DATASETS = {"mirflickr25k", "nuswide", "mscoco"}
STAGE3_VALIDATOR_VERSION = "stage3_se_c_s_validator_v1"
REQUIRED_META_FIELDS = {
    "dataset",
    "semantic_set_id",
    "feature_set_id",
    "train_count",
    "matrix_shape",
    "dtype",
    "lambda_ar_fusion",
    "tau_confidence",
    "topk_for_diagnostics",
    "stage1_manifest_filtered_order_sha256",
    "stage1_sample_id_order_sha256",
    "stage1_train_ids_sha256",
    "stage2_manifest_filtered_order_sha256",
    "stage2_feature_set_id",
    "train_ids_sha256",
    "train_sample_id_order_sha256",
    "train_indices_sha256",
    "x_i_train_shape",
    "x_t_train_shape",
    "generated_at_utc",
    "formal_input_files",
}
REQUIRED_DIAGNOSTIC_FIELDS = {
    "train_count",
    "shape_ok",
    "range_a_ok",
    "range_r_ok",
    "range_se_ok",
    "range_c_ok",
    "range_s_ok",
    "a_min",
    "a_max",
    "a_mean",
    "a_std",
    "r_min",
    "r_max",
    "r_mean",
    "r_std",
    "se_min",
    "se_max",
    "se_mean",
    "se_std",
    "c_min",
    "c_max",
    "c_mean",
    "c_std",
    "s_min",
    "s_max",
    "s_mean",
    "s_std",
    "diag_mean_a",
    "offdiag_mean_a",
    "diag_mean_r",
    "offdiag_mean_r",
    "diag_mean_se",
    "offdiag_mean_se",
    "diag_mean_s",
    "offdiag_mean_s",
    "diag_minus_offdiag_s",
    "diag_over_offdiag_ratio",
    "paired_diag_quantile_in_row_mean",
    "paired_diag_quantile_in_row_median",
    "paired_diag_quantile_in_col_mean",
    "paired_diag_quantile_in_col_median",
    "row_topk_coverage",
    "col_topk_coverage",
    "diag_in_row_topk_rate",
    "diag_in_col_topk_rate",
    "topk_for_diagnostics",
    "semantic_validator_passed",
    "failure_reason",
}


def validate_stage3_semantic(repo_root: Path, config_path: Path, dataset: str) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = read_json(_resolve_repo_path(repo_root, config_path))
    if dataset not in SUPPORTED_STAGE3_DATASETS or dataset not in config["datasets"]:
        raise ValueError("Stage 3 validator supports mirflickr25k, nuswide, and mscoco only")

    dataset_config = config["datasets"][dataset]
    profile = config["profiles"][dataset]
    processed_root = _resolve_repo_path(repo_root, Path(config["inputs"]["processed_root"])) / dataset
    paths = _stage3_paths(processed_root, config)
    failures: list[str] = []
    input_presence = _check_presence(
        {
            "manifest_filtered": paths["manifest_filtered"],
            "train_ids": paths["train_ids"],
            "order_hashes": paths["order_hashes"],
            "stage2_x_i": paths["x_i"],
            "stage2_x_t": paths["x_t"],
            "stage2_meta": paths["stage2_meta"],
        },
        failures,
        "missing Stage 3 input",
    )
    output_presence = _check_presence(
        {
            "A": paths["a"],
            "R": paths["r"],
            "Se": paths["se"],
            "C": paths["c"],
            "S": paths["s"],
            "meta": paths["meta"],
            "semantic_diagnostics": paths["diagnostics"],
            "Omega_topk_diag": paths["omega"],
        },
        failures,
        "missing Stage 3 output",
    )
    output_presence["validator_summary"] = True

    rows: list[dict[str, Any]] = []
    train_ids: list[str] = []
    order_hashes: dict[str, Any] = {}
    stage2_meta: dict[str, Any] = {}
    meta: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    matrix_summary: dict[str, Any] = {}
    meta_hashes_match_stage1 = False
    meta_hashes_match_stage2 = False
    train_mapping_verified = False
    omega_diagnostic_only = False

    if not failures:
        rows = list(iter_jsonl(paths["manifest_filtered"]))
        train_ids = _read_lines(paths["train_ids"])
        order_hashes = read_json(paths["order_hashes"])
        stage2_meta = read_json(paths["stage2_meta"])
        meta = read_json(paths["meta"])
        diagnostics = read_json(paths["diagnostics"])

        _check_stage1_stage2_boundary(rows, train_ids, order_hashes, stage2_meta, config, dataset_config, dataset, failures)
        train_indices = _train_indices(rows, train_ids, failures)
        train_mapping_verified = train_indices is not None and meta.get("train_indices_sha256") == hash_lines(str(int(index)) for index in train_indices)
        if not train_mapping_verified:
            failures.append("train_indices_sha256 does not match sample_id -> manifest row index mapping")
        _check_meta(meta, order_hashes, stage2_meta, config, dataset_config, profile, dataset, train_ids, train_indices, failures)
        meta_hashes_match_stage1 = _meta_hashes_match_stage1(meta, order_hashes)
        meta_hashes_match_stage2 = _meta_hashes_match_stage2(meta, stage2_meta)

        matrices = {
            "A": np.load(paths["a"], mmap_mode="r"),
            "R": np.load(paths["r"], mmap_mode="r"),
            "Se": np.load(paths["se"], mmap_mode="r"),
            "C": np.load(paths["c"], mmap_mode="r"),
            "S": np.load(paths["s"], mmap_mode="r"),
        }
        for name, matrix in matrices.items():
            _check_matrix(name, matrix, bool(name == "C"), dataset_config, config, failures)
            matrix_summary[name] = {
                "shape": list(matrix.shape),
                "dtype": str(matrix.dtype),
                "min": float(np.min(matrix)),
                "max": float(np.max(matrix)),
                "mean": float(np.mean(matrix, dtype=np.float64)),
                "std": float(np.std(matrix, dtype=np.float64)),
            }

        recomputed = _semantic_diagnostics(
            np.asarray(matrices["A"]),
            np.asarray(matrices["R"]),
            np.asarray(matrices["Se"]),
            np.asarray(matrices["C"]),
            np.asarray(matrices["S"]),
            int(profile["topk_for_diagnostics"]),
            config,
        )
        _check_diagnostics(diagnostics, recomputed, failures)
        stop_go_thresholds = _stop_go_thresholds(config)
        stop_go_checks = _stop_go_checks(recomputed, stop_go_thresholds)
        _check_stop_go(stop_go_checks, failures)
        omega_diagnostic_only = _check_omega(paths["omega"], int(profile["topk_for_diagnostics"]), int(dataset_config["expected_train_count"]), failures)
    else:
        stop_go_thresholds = _stop_go_thresholds(config)
        stop_go_checks = {}

    passed = len(failures) == 0
    if diagnostics:
        diagnostics["semantic_validator_passed"] = passed
        diagnostics["failure_reason"] = None if passed else failures
        write_json(paths["diagnostics"], diagnostics)

    summary = {
        "stage": "stage3",
        "substage": f"stage3a_{dataset}" if dataset == "mirflickr25k" else f"stage3_{dataset}",
        "validator_version": STAGE3_VALIDATOR_VERSION,
        "generated_at_utc": _utc_now(),
        "dataset": dataset,
        "semantic_set_id": config["semantic_set_id"],
        "feature_set_id": config["feature_set_id"],
        "semantic_cache_dir": str(paths["semantic_dir"]),
        "passed": passed,
        "failure_count": len(failures),
        "failure_reason": failures,
        "stage3_input_file_presence": input_presence,
        "stage3_output_file_presence": output_presence,
        "train_count": len(train_ids),
        "matrix_shape": [int(dataset_config["expected_train_count"]), int(dataset_config["expected_train_count"])],
        "lambda_ar_fusion": float(profile["lambda_ar_fusion"]),
        "tau_confidence": float(profile["tau_confidence"]),
        "topk_for_diagnostics": int(profile["topk_for_diagnostics"]),
        "meta_hashes_match_stage1": meta_hashes_match_stage1,
        "meta_hashes_match_stage2": meta_hashes_match_stage2,
        "train_mapping_verified": train_mapping_verified,
        "omega_topk_diag_diagnostic_only": omega_diagnostic_only,
        "stop_go_thresholds": stop_go_thresholds,
        "stop_go_checks": stop_go_checks,
        "matrices": matrix_summary,
        "diagnostics": diagnostics,
    }
    write_json(paths["validator_summary"], summary)
    return summary


def _check_stage1_stage2_boundary(
    rows: list[dict[str, Any]],
    train_ids: list[str],
    order_hashes: dict[str, Any],
    stage2_meta: dict[str, Any],
    config: dict[str, Any],
    dataset_config: dict[str, Any],
    dataset: str,
    failures: list[str],
) -> None:
    expected_filtered = int(dataset_config["expected_filtered_count"])
    expected_train = int(dataset_config["expected_train_count"])
    if len(rows) != expected_filtered:
        failures.append(f"manifest_filtered count mismatch: expected {expected_filtered}, got {len(rows)}")
    if len(train_ids) != expected_train:
        failures.append(f"train_ids count mismatch: expected {expected_train}, got {len(train_ids)}")
    if len(set(train_ids)) != len(train_ids):
        failures.append("train_ids contains duplicates")
    sample_ids = [str(row.get("sample_id")) for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        failures.append("manifest_filtered sample_id is not unique")
    if set(train_ids) - set(sample_ids):
        failures.append("train_ids contains ids outside manifest_filtered")
    expected_hashes = {
        "sample_id_order_sha256": hash_lines(sorted(sample_ids)),
        "manifest_filtered_order_sha256": hash_lines(sample_ids),
        "train_ids_sha256": hash_lines(train_ids),
    }
    for key, value in expected_hashes.items():
        if order_hashes.get(key) != value:
            failures.append(f"Stage 1 order_hashes mismatch for {key}")
    if stage2_meta.get("dataset") != dataset:
        failures.append("Stage 2 meta.dataset mismatch")
    if stage2_meta.get("feature_set_id") != config["feature_set_id"]:
        failures.append("Stage 2 feature_set_id mismatch")
    if stage2_meta.get("manifest_filtered_order_sha256") != order_hashes.get("manifest_filtered_order_sha256"):
        failures.append("Stage 2 manifest hash does not match Stage 1")
    if stage2_meta.get("train_ids_sha256") != order_hashes.get("train_ids_sha256"):
        failures.append("Stage 2 train_ids hash does not match Stage 1")


def _train_indices(rows: list[dict[str, Any]], train_ids: list[str], failures: list[str]) -> np.ndarray | None:
    sample_ids = [str(row.get("sample_id")) for row in rows]
    id_to_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    indices: list[int] = []
    for sample_id in train_ids:
        index = id_to_index.get(sample_id)
        if index is None:
            failures.append(f"train id not present in manifest_filtered: {sample_id}")
            return None
        indices.append(index)
    return np.asarray(indices, dtype=np.int64)


def _check_meta(
    meta: dict[str, Any],
    order_hashes: dict[str, Any],
    stage2_meta: dict[str, Any],
    config: dict[str, Any],
    dataset_config: dict[str, Any],
    profile: dict[str, Any],
    dataset: str,
    train_ids: list[str],
    train_indices: np.ndarray | None,
    failures: list[str],
) -> None:
    missing = sorted(REQUIRED_META_FIELDS - set(meta))
    if missing:
        failures.append(f"meta.json missing fields: {missing}")
    expected_train = int(dataset_config["expected_train_count"])
    expected_shape = [expected_train, expected_train]
    if meta.get("dataset") != dataset:
        failures.append("meta.dataset mismatch")
    if meta.get("semantic_set_id") != config["semantic_set_id"]:
        failures.append("meta.semantic_set_id mismatch")
    if meta.get("feature_set_id") != config["feature_set_id"]:
        failures.append("meta.feature_set_id mismatch")
    if meta.get("train_count") != expected_train:
        failures.append("meta.train_count mismatch")
    if meta.get("matrix_shape") != expected_shape:
        failures.append("meta.matrix_shape mismatch")
    if meta.get("dtype") != "float32":
        failures.append("meta.dtype mismatch")
    if meta.get("lambda_ar_fusion") != float(profile["lambda_ar_fusion"]):
        failures.append("meta.lambda_ar_fusion mismatch")
    if meta.get("tau_confidence") != float(profile["tau_confidence"]):
        failures.append("meta.tau_confidence mismatch")
    if meta.get("topk_for_diagnostics") != int(profile["topk_for_diagnostics"]):
        failures.append("meta.topk_for_diagnostics mismatch")
    if meta.get("stage1_manifest_filtered_order_sha256") != order_hashes.get("manifest_filtered_order_sha256"):
        failures.append("meta Stage 1 manifest hash mismatch")
    if meta.get("stage1_sample_id_order_sha256") != order_hashes.get("sample_id_order_sha256"):
        failures.append("meta Stage 1 sample id hash mismatch")
    if meta.get("stage1_train_ids_sha256") != order_hashes.get("train_ids_sha256"):
        failures.append("meta Stage 1 train hash mismatch")
    if meta.get("stage2_manifest_filtered_order_sha256") != stage2_meta.get("manifest_filtered_order_sha256"):
        failures.append("meta Stage 2 manifest hash mismatch")
    if meta.get("stage2_feature_set_id") != stage2_meta.get("feature_set_id"):
        failures.append("meta Stage 2 feature_set_id mismatch")
    if meta.get("train_ids_sha256") != hash_lines(train_ids):
        failures.append("meta train_ids_sha256 mismatch")
    if meta.get("train_sample_id_order_sha256") != hash_lines(train_ids):
        failures.append("meta train_sample_id_order_sha256 mismatch")
    if train_indices is not None and meta.get("train_indices_sha256") != hash_lines(str(int(index)) for index in train_indices):
        failures.append("meta train_indices_sha256 mismatch")
    feature_shape = [expected_train, int(dataset_config["feature_dim"])]
    if meta.get("x_i_train_shape") != feature_shape:
        failures.append("meta x_i_train_shape mismatch")
    if meta.get("x_t_train_shape") != feature_shape:
        failures.append("meta x_t_train_shape mismatch")
    if meta.get("s_post_normalization_used") is not False:
        failures.append("meta indicates S post normalization was used")
    if meta.get("s_minmax_scale_used") is not False:
        failures.append("meta indicates S minmax scaling was used")
    if meta.get("s_topk_mask_used") is not False:
        failures.append("meta indicates S topk mask was used")
    if meta.get("s_identity_boost_used") is not False:
        failures.append("meta indicates S identity boost was used")
    if meta.get("omega_topk_diag_role") != "diagnostic_only_not_training_supervision":
        failures.append("meta omega_topk_diag_role mismatch")


def _check_matrix(
    name: str,
    matrix: np.ndarray,
    positive: bool,
    dataset_config: dict[str, Any],
    config: dict[str, Any],
    failures: list[str],
) -> None:
    expected_train = int(dataset_config["expected_train_count"])
    expected_shape = (expected_train, expected_train)
    if matrix.shape != expected_shape:
        failures.append(f"{name} shape mismatch: expected {expected_shape}, got {matrix.shape}")
    if matrix.dtype != np.float32:
        failures.append(f"{name} dtype mismatch: expected float32, got {matrix.dtype}")
    if not np.isfinite(matrix).all():
        failures.append(f"{name} contains NaN or Inf")
        return
    tol = float(config["validation"]["range_tolerance"])
    if positive:
        if float(np.min(matrix)) <= 0.0 or float(np.max(matrix)) > 1.0 + tol:
            failures.append(f"{name} range is not in (0, 1]")
    else:
        if float(np.min(matrix)) < -tol or float(np.max(matrix)) > 1.0 + tol:
            failures.append(f"{name} range is not in [0, 1]")


def _check_diagnostics(diagnostics: dict[str, Any], recomputed: dict[str, Any], failures: list[str]) -> None:
    missing = sorted(REQUIRED_DIAGNOSTIC_FIELDS - set(diagnostics))
    if missing:
        failures.append(f"semantic_diagnostics.json missing fields: {missing}")
    for key, value in recomputed.items():
        if key not in diagnostics:
            continue
        if isinstance(value, bool):
            if diagnostics.get(key) is not value:
                failures.append(f"diagnostic {key} mismatch")
        elif isinstance(value, (float, int)):
            actual = diagnostics.get(key)
            if not isinstance(actual, (float, int)) or abs(float(actual) - float(value)) > 1e-6:
                failures.append(f"diagnostic {key} mismatch")


def _stop_go_thresholds(config: dict[str, Any]) -> dict[str, float]:
    validation = config["validation"]
    return {
        "min_diag_minus_offdiag_s": float(validation["min_diag_minus_offdiag_s"]),
        "min_diag_over_offdiag_ratio": float(validation["min_diag_over_offdiag_ratio"]),
        "min_paired_diag_quantile_in_row_median": float(validation["min_paired_diag_quantile_median"]),
        "min_paired_diag_quantile_in_col_median": float(validation["min_paired_diag_quantile_median"]),
        "min_diag_in_row_topk_rate": float(validation["min_diag_in_row_topk_rate"]),
        "min_diag_in_col_topk_rate": float(validation["min_diag_in_col_topk_rate"]),
        "min_row_topk_coverage": float(validation["min_row_topk_coverage"]),
        "min_col_topk_coverage": float(validation["min_col_topk_coverage"]),
    }


def _stop_go_checks(diagnostics: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    return {
        "shape_ok": {"value": bool(diagnostics.get("shape_ok")), "passed": diagnostics.get("shape_ok") is True},
        "range_a_ok": {"value": bool(diagnostics.get("range_a_ok")), "passed": diagnostics.get("range_a_ok") is True},
        "range_r_ok": {"value": bool(diagnostics.get("range_r_ok")), "passed": diagnostics.get("range_r_ok") is True},
        "range_se_ok": {"value": bool(diagnostics.get("range_se_ok")), "passed": diagnostics.get("range_se_ok") is True},
        "range_c_ok": {"value": bool(diagnostics.get("range_c_ok")), "passed": diagnostics.get("range_c_ok") is True},
        "range_s_ok": {"value": bool(diagnostics.get("range_s_ok")), "passed": diagnostics.get("range_s_ok") is True},
        "diag_mean_s_gt_offdiag_mean_s": {
            "value": [diagnostics.get("diag_mean_s"), diagnostics.get("offdiag_mean_s")],
            "passed": diagnostics.get("diag_mean_s", 0.0) > diagnostics.get("offdiag_mean_s", 0.0),
        },
        "diag_minus_offdiag_s": {
            "value": diagnostics.get("diag_minus_offdiag_s"),
            "threshold": thresholds["min_diag_minus_offdiag_s"],
            "passed": diagnostics.get("diag_minus_offdiag_s", 0.0) > thresholds["min_diag_minus_offdiag_s"],
        },
        "diag_over_offdiag_ratio": {
            "value": diagnostics.get("diag_over_offdiag_ratio"),
            "threshold": thresholds["min_diag_over_offdiag_ratio"],
            "passed": diagnostics.get("diag_over_offdiag_ratio", 0.0) >= thresholds["min_diag_over_offdiag_ratio"],
        },
        "paired_diag_quantile_in_row_median": {
            "value": diagnostics.get("paired_diag_quantile_in_row_median"),
            "threshold": thresholds["min_paired_diag_quantile_in_row_median"],
            "passed": diagnostics.get("paired_diag_quantile_in_row_median", 0.0) >= thresholds["min_paired_diag_quantile_in_row_median"],
        },
        "paired_diag_quantile_in_col_median": {
            "value": diagnostics.get("paired_diag_quantile_in_col_median"),
            "threshold": thresholds["min_paired_diag_quantile_in_col_median"],
            "passed": diagnostics.get("paired_diag_quantile_in_col_median", 0.0) >= thresholds["min_paired_diag_quantile_in_col_median"],
        },
        "diag_in_row_topk_rate": {
            "value": diagnostics.get("diag_in_row_topk_rate"),
            "threshold": thresholds["min_diag_in_row_topk_rate"],
            "passed": diagnostics.get("diag_in_row_topk_rate", 0.0) >= thresholds["min_diag_in_row_topk_rate"],
        },
        "diag_in_col_topk_rate": {
            "value": diagnostics.get("diag_in_col_topk_rate"),
            "threshold": thresholds["min_diag_in_col_topk_rate"],
            "passed": diagnostics.get("diag_in_col_topk_rate", 0.0) >= thresholds["min_diag_in_col_topk_rate"],
        },
        "row_topk_coverage": {
            "value": diagnostics.get("row_topk_coverage"),
            "threshold": thresholds["min_row_topk_coverage"],
            "passed": diagnostics.get("row_topk_coverage", 0.0) >= thresholds["min_row_topk_coverage"],
        },
        "col_topk_coverage": {
            "value": diagnostics.get("col_topk_coverage"),
            "threshold": thresholds["min_col_topk_coverage"],
            "passed": diagnostics.get("col_topk_coverage", 0.0) >= thresholds["min_col_topk_coverage"],
        },
    }


def _check_stop_go(checks: dict[str, Any], failures: list[str]) -> None:
    for name, check in checks.items():
        if check.get("passed") is not True:
            failures.append(f"{name} failed stop/go check")


def _check_omega(path: Path, topk: int, train_count: int, failures: list[str]) -> bool:
    try:
        omega = np.load(path)
    except Exception as exc:
        failures.append(f"Omega_topk_diag.npz failed to load: {exc}")
        return False
    required = {"omega_rows", "omega_cols", "topk_for_diagnostics", "matrix_shape", "diagnostic_only"}
    missing = sorted(required - set(omega.files))
    if missing:
        failures.append(f"Omega_topk_diag.npz missing arrays: {missing}")
        return False
    diagnostic_only = bool(np.asarray(omega["diagnostic_only"]).reshape(-1)[0] == 1)
    if not diagnostic_only:
        failures.append("Omega_topk_diag.npz is not marked diagnostic_only")
    if int(np.asarray(omega["topk_for_diagnostics"]).reshape(-1)[0]) != topk:
        failures.append("Omega_topk_diag topk mismatch")
    if list(np.asarray(omega["matrix_shape"], dtype=np.int64)) != [train_count, train_count]:
        failures.append("Omega_topk_diag matrix_shape mismatch")
    return diagnostic_only


def _meta_hashes_match_stage1(meta: dict[str, Any], order_hashes: dict[str, Any]) -> bool:
    return (
        meta.get("stage1_manifest_filtered_order_sha256") == order_hashes.get("manifest_filtered_order_sha256")
        and meta.get("stage1_sample_id_order_sha256") == order_hashes.get("sample_id_order_sha256")
        and meta.get("stage1_train_ids_sha256") == order_hashes.get("train_ids_sha256")
    )


def _meta_hashes_match_stage2(meta: dict[str, Any], stage2_meta: dict[str, Any]) -> bool:
    return (
        meta.get("stage2_manifest_filtered_order_sha256") == stage2_meta.get("manifest_filtered_order_sha256")
        and meta.get("stage2_feature_set_id") == stage2_meta.get("feature_set_id")
    )


def _check_presence(paths: dict[str, Path], failures: list[str], prefix: str) -> dict[str, bool]:
    presence = {name: path.is_file() for name, path in paths.items()}
    for name, exists in presence.items():
        if not exists:
            failures.append(f"{prefix}: {name}={paths[name]}")
    return presence


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

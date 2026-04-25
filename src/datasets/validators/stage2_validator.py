from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.utils.jsonl import iter_jsonl, read_json, write_json


SUPPORTED_STAGE2_DATASETS = {"mirflickr25k", "nuswide", "mscoco"}
SUBSTAGE_BY_DATASET = {
    "mirflickr25k": "stage2a_mirflickr25k",
    "nuswide": "stage2b_nuswide",
    "mscoco": "stage2c_mscoco",
}
STAGE2_VALIDATOR_VERSION = "stage2_mir_nus_coco_clip_validator_v3"
REQUIRED_META_FIELDS = {
    "dataset",
    "feature_set_id",
    "backbone_id",
    "model_local_path",
    "local_files_only",
    "device",
    "dtype",
    "feature_dim",
    "filtered_count",
    "image_batch_size",
    "text_batch_size",
    "manifest_filtered_order_sha256",
    "sample_id_order_sha256",
    "query_ids_sha256",
    "retrieval_ids_sha256",
    "train_ids_sha256",
    "image_preprocess_protocol",
    "text_tokenizer_protocol",
    "model_eval",
    "torch_no_grad",
    "amp_enabled",
    "generated_at_utc",
}
REQUIRED_BASELINE_FIELDS = {
    "dataset",
    "feature_set_id",
    "filtered_count",
    "query_count",
    "retrieval_count",
    "paired_cosine_mean",
    "paired_cosine_median",
    "random_cosine_mean",
    "random_cosine_median",
    "cosine_gap_mean",
    "cosine_gap_median",
    "clip_i2t_map_at_50",
    "clip_t2i_map_at_50",
    "block_size_similarity",
    "baseline_completed",
    "failure_reason",
}


def validate_stage2_features(repo_root: Path, config_path: Path, dataset: str) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = read_json(_resolve_repo_path(repo_root, config_path))
    if dataset not in SUPPORTED_STAGE2_DATASETS or dataset not in config["datasets"]:
        raise ValueError("Stage 2 validator currently supports mirflickr25k, nuswide, and mscoco only")
    dataset_config = config["datasets"][dataset]
    processed_root = _resolve_repo_path(repo_root, Path(config["inputs"]["processed_root"])) / dataset
    input_paths = _stage2_input_paths(processed_root)
    output_dir = processed_root / "feature_cache" / config["outputs"]["feature_cache_dirname"]
    output_paths = _stage2_output_paths(output_dir)
    failures: list[str] = []
    stage1_presence = _check_presence(input_paths, failures, "missing Stage 1 input")
    stage2_presence = _check_presence(
        {name: path for name, path in output_paths.items() if name != "validator_summary"},
        failures,
        "missing Stage 2 output",
    )
    stage2_presence["validator_summary"] = True

    rows: list[dict[str, Any]] = []
    query_ids: list[str] = []
    retrieval_ids: list[str] = []
    train_ids: list[str] = []
    order_hashes: dict[str, Any] = {}
    meta: dict[str, Any] = {}
    baseline: dict[str, Any] = {}
    x_i_norm_range = [None, None]
    x_t_norm_range = [None, None]

    if not failures:
        rows = list(iter_jsonl(input_paths["manifest_filtered"]))
        query_ids = _read_lines(input_paths["query_ids"])
        retrieval_ids = _read_lines(input_paths["retrieval_ids"])
        train_ids = _read_lines(input_paths["train_ids"])
        order_hashes = read_json(input_paths["order_hashes"])
        meta = read_json(output_paths["meta"])
        baseline = read_json(output_paths["baseline_summary"])

        _check_stage1_boundary(rows, query_ids, retrieval_ids, train_ids, dataset_config, dataset, failures)
        _check_meta(meta, order_hashes, config, dataset_config, dataset, failures)
        _check_hashes(rows, query_ids, retrieval_ids, train_ids, order_hashes, meta, failures)
        x_i_norm_range = _check_feature_file(output_paths["x_i"], "X_I", int(dataset_config["expected_filtered_count"]), failures)
        x_t_norm_range = _check_feature_file(output_paths["x_t"], "X_T", int(dataset_config["expected_filtered_count"]), failures)
        _check_baseline(baseline, config, dataset_config, dataset, failures)

    summary = {
        "stage": "stage2",
        "substage": SUBSTAGE_BY_DATASET[dataset],
        "validator_version": STAGE2_VALIDATOR_VERSION,
        "generated_at_utc": _utc_now(),
        "dataset": dataset,
        "feature_set_id": config["feature_set_id"],
        "processed_root": str(processed_root),
        "feature_cache_dir": str(output_dir),
        "passed": len(failures) == 0,
        "failure_count": len(failures),
        "failure_reason": failures,
        "stage1_input_file_presence": stage1_presence,
        "stage2_output_file_presence": stage2_presence,
        "filtered_count": len(rows),
        "query_count": len(query_ids),
        "retrieval_count": len(retrieval_ids),
        "train_count": len(train_ids),
        "x_i_shape": list(np.load(output_paths["x_i"], mmap_mode="r").shape) if output_paths["x_i"].is_file() else None,
        "x_t_shape": list(np.load(output_paths["x_t"], mmap_mode="r").shape) if output_paths["x_t"].is_file() else None,
        "x_i_dtype": str(np.load(output_paths["x_i"], mmap_mode="r").dtype) if output_paths["x_i"].is_file() else None,
        "x_t_dtype": str(np.load(output_paths["x_t"], mmap_mode="r").dtype) if output_paths["x_t"].is_file() else None,
        "x_i_norm_range": x_i_norm_range,
        "x_t_norm_range": x_t_norm_range,
        "meta_hashes_match_stage1": _hashes_match_meta(order_hashes, meta) if meta and order_hashes else False,
        "baseline_completed": baseline.get("baseline_completed") if baseline else None,
        "paired_cosine_mean": baseline.get("paired_cosine_mean") if baseline else None,
        "random_cosine_mean": baseline.get("random_cosine_mean") if baseline else None,
        "cosine_gap_mean": baseline.get("cosine_gap_mean") if baseline else None,
        "paired_cosine_median": baseline.get("paired_cosine_median") if baseline else None,
        "random_cosine_median": baseline.get("random_cosine_median") if baseline else None,
        "cosine_gap_median": baseline.get("cosine_gap_median") if baseline else None,
        "clip_i2t_map_at_50": baseline.get("clip_i2t_map_at_50") if baseline else None,
        "clip_t2i_map_at_50": baseline.get("clip_t2i_map_at_50") if baseline else None,
        "silent_fallback_used": bool(meta.get("silent_fallback_used")) if meta else None,
    }
    write_json(output_paths["validator_summary"], summary)
    return summary


def _check_stage1_boundary(
    rows: list[dict[str, Any]],
    query_ids: list[str],
    retrieval_ids: list[str],
    train_ids: list[str],
    dataset_config: dict[str, Any],
    dataset: str,
    failures: list[str],
) -> None:
    expected_filtered = int(dataset_config["expected_filtered_count"])
    if len(rows) != expected_filtered:
        failures.append(f"manifest_filtered count mismatch: expected {expected_filtered}, got {len(rows)}")
    if len(query_ids) != int(dataset_config["expected_query_count"]):
        failures.append("query_ids count mismatch")
    if len(retrieval_ids) != int(dataset_config["expected_retrieval_count"]):
        failures.append("retrieval_ids count mismatch")
    if len(train_ids) != int(dataset_config["expected_train_count"]):
        failures.append("train_ids count mismatch")
    sample_ids = [str(row.get("sample_id")) for row in rows]
    sample_set = set(sample_ids)
    if len(sample_ids) != len(sample_set):
        failures.append("manifest_filtered sample_id is not unique")
    if set(query_ids) - sample_set:
        failures.append("query_ids contains ids outside manifest")
    if set(retrieval_ids) - sample_set:
        failures.append("retrieval_ids contains ids outside manifest")
    if not set(train_ids).issubset(set(retrieval_ids)):
        failures.append("train_ids is not a subset of retrieval_ids")
    if set(query_ids) & set(retrieval_ids):
        failures.append("query_ids and retrieval_ids overlap")
    label_dim = int(dataset_config["label_dimension"])
    sample_prefix = f"{dataset_config['sample_id_prefix']}_"
    for index, row in enumerate(rows, start=1):
        if row.get("dataset_name") != dataset:
            failures.append(f"manifest row {index} has wrong dataset_name")
        if not isinstance(row.get("sample_id"), str) or not row["sample_id"].startswith(sample_prefix):
            failures.append(f"manifest row {index} has invalid sample_id")
        if not isinstance(row.get("text_source"), str):
            failures.append(f"manifest row {index} has missing text_source")
        vector = row.get("label_vector")
        if not isinstance(vector, list) or len(vector) != label_dim or any(value not in (0, 1) for value in vector):
            failures.append(f"manifest row {index} has invalid label_vector")


def _check_meta(
    meta: dict[str, Any],
    order_hashes: dict[str, Any],
    config: dict[str, Any],
    dataset_config: dict[str, Any],
    dataset: str,
    failures: list[str],
) -> None:
    missing = sorted(REQUIRED_META_FIELDS - set(meta))
    if missing:
        failures.append(f"meta.json missing fields: {missing}")
    if meta.get("dataset") != dataset:
        failures.append("meta.dataset mismatch")
    if meta.get("feature_set_id") != config["feature_set_id"]:
        failures.append("meta.feature_set_id mismatch")
    if meta.get("backbone_id") != "openai/clip-vit-base-patch32":
        failures.append("meta.backbone_id mismatch")
    if meta.get("local_files_only") is not True:
        failures.append("meta.local_files_only is not true")
    if meta.get("device") != "cuda:0":
        failures.append("meta.device is not cuda:0")
    if meta.get("dtype") != "float32":
        failures.append("meta.dtype is not float32")
    if meta.get("feature_dim") != 512:
        failures.append("meta.feature_dim is not 512")
    if meta.get("filtered_count") != int(dataset_config["expected_filtered_count"]):
        failures.append("meta.filtered_count mismatch")
    if meta.get("model_eval") is not True:
        failures.append("meta.model_eval is not true")
    if meta.get("torch_no_grad") is not True:
        failures.append("meta.torch_no_grad is not true")
    if meta.get("amp_enabled") is not False:
        failures.append("meta.amp_enabled is not false")
    if meta.get("silent_fallback_used") not in (False, None):
        failures.append("meta.silent_fallback_used is not false")
    if meta.get("bad_sample_skip_used") not in (False, None):
        failures.append("meta.bad_sample_skip_used is not false")
    if meta.get("zero_vector_padding_used") not in (False, None):
        failures.append("meta.zero_vector_padding_used is not false")
    for key in ("manifest_filtered_order_sha256", "sample_id_order_sha256", "query_ids_sha256", "retrieval_ids_sha256", "train_ids_sha256"):
        if meta.get(key) != order_hashes.get(key):
            failures.append(f"meta.{key} does not match Stage 1 order_hashes.json")


def _check_hashes(
    rows: list[dict[str, Any]],
    query_ids: list[str],
    retrieval_ids: list[str],
    train_ids: list[str],
    order_hashes: dict[str, Any],
    meta: dict[str, Any],
    failures: list[str],
) -> None:
    expected = {
        "sample_id_order_sha256": hash_lines(sorted(str(row["sample_id"]) for row in rows)),
        "manifest_filtered_order_sha256": hash_lines(str(row["sample_id"]) for row in rows),
        "query_ids_sha256": hash_lines(query_ids),
        "retrieval_ids_sha256": hash_lines(retrieval_ids),
        "train_ids_sha256": hash_lines(train_ids),
    }
    for key, value in expected.items():
        if order_hashes.get(key) != value:
            failures.append(f"Stage 1 order_hashes mismatch for {key}")
        if meta.get(key) != value:
            failures.append(f"meta hash mismatch for {key}")


def _check_feature_file(path: Path, name: str, expected_rows: int, failures: list[str]) -> list[float | None]:
    array = np.load(path, mmap_mode="r")
    if array.shape != (expected_rows, 512):
        failures.append(f"{name} shape mismatch: expected {(expected_rows, 512)}, got {array.shape}")
    if array.dtype != np.float32:
        failures.append(f"{name} dtype mismatch: expected float32, got {array.dtype}")
    if not np.isfinite(array).all():
        failures.append(f"{name} contains NaN or Inf")
    norms = np.linalg.norm(array, axis=1)
    norm_min = float(norms.min())
    norm_max = float(norms.max())
    if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
        failures.append(f"{name} rows are not L2 normalized")
    return [norm_min, norm_max]


def _check_baseline(
    baseline: dict[str, Any],
    config: dict[str, Any],
    dataset_config: dict[str, Any],
    dataset: str,
    failures: list[str],
) -> None:
    missing = sorted(REQUIRED_BASELINE_FIELDS - set(baseline))
    if missing:
        failures.append(f"baseline_summary.json missing fields: {missing}")
    if baseline.get("dataset") != dataset:
        failures.append("baseline.dataset mismatch")
    if baseline.get("feature_set_id") != config["feature_set_id"]:
        failures.append("baseline.feature_set_id mismatch")
    if baseline.get("filtered_count") != int(dataset_config["expected_filtered_count"]):
        failures.append("baseline.filtered_count mismatch")
    if baseline.get("query_count") != int(dataset_config["expected_query_count"]):
        failures.append("baseline.query_count mismatch")
    if baseline.get("retrieval_count") != int(dataset_config["expected_retrieval_count"]):
        failures.append("baseline.retrieval_count mismatch")
    if baseline.get("baseline_completed") is not True:
        failures.append("baseline_completed is not true")
    if baseline.get("failure_reason") not in (None, "", []):
        failures.append("baseline failure_reason is not empty")
    paired_mean = baseline.get("paired_cosine_mean")
    random_mean = baseline.get("random_cosine_mean")
    paired_median = baseline.get("paired_cosine_median")
    random_median = baseline.get("random_cosine_median")
    if not isinstance(paired_mean, (float, int)) or not isinstance(random_mean, (float, int)) or paired_mean <= random_mean:
        failures.append("paired_cosine_mean is not greater than random_cosine_mean")
    if not isinstance(paired_median, (float, int)) or not isinstance(random_median, (float, int)) or paired_median <= random_median:
        failures.append("paired_cosine_median is not greater than random_cosine_median")


def _hashes_match_meta(order_hashes: dict[str, Any], meta: dict[str, Any]) -> bool:
    return all(
        meta.get(key) == order_hashes.get(key)
        for key in ("manifest_filtered_order_sha256", "sample_id_order_sha256", "query_ids_sha256", "retrieval_ids_sha256", "train_ids_sha256")
    )


def _check_presence(paths: dict[str, Path], failures: list[str], prefix: str) -> dict[str, bool]:
    presence = {name: path.is_file() for name, path in paths.items()}
    for name, exists in presence.items():
        if not exists:
            failures.append(f"{prefix}: {name}={paths[name]}")
    return presence


def _stage2_input_paths(processed_root: Path) -> dict[str, Path]:
    return {
        "manifest_filtered": processed_root / "manifest" / "manifest_filtered.jsonl",
        "manifest_meta": processed_root / "manifest" / "manifest_meta.json",
        "query_ids": processed_root / "splits" / "query_ids.txt",
        "retrieval_ids": processed_root / "splits" / "retrieval_ids.txt",
        "train_ids": processed_root / "splits" / "train_ids.txt",
        "order_hashes": processed_root / "reports" / "order_hashes.json",
    }


def _stage2_output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "x_i": output_dir / "X_I.npy",
        "x_t": output_dir / "X_T.npy",
        "meta": output_dir / "meta.json",
        "validator_summary": output_dir / "validator_summary.json",
        "baseline_summary": output_dir / "baseline_summary.json",
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

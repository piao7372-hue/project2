from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.utils.jsonl import iter_jsonl, read_json, write_json


SUPPORTED_STAGE4_FORWARD_DATASETS = {"mirflickr25k", "nuswide", "mscoco"}
STAGE4_VALIDATOR_VERSION = "stage4_forward_validator_v3"


def validate_stage4_forward(repo_root: Path, config_path: Path, dataset: str, all_bits: bool) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = read_json(_resolve_repo_path(repo_root, config_path))
    _check_stage4_forward_dataset(dataset, config)
    if not all_bits:
        raise RuntimeError("Stage 4 forward validator requires --all-bits")

    dataset_config = config["datasets"][dataset]
    processed_root = _resolve_repo_path(repo_root, Path(config["inputs"]["processed_root"])) / dataset
    paths = _stage4_paths(processed_root, config)
    context_failures: list[str] = []
    context = _load_and_check_context(paths, config, dataset_config, dataset, context_failures)

    bit_summaries: dict[str, Any] = {}
    for bit in [int(value) for value in config["hash_bits"]]:
        summary = _validate_bit(paths, config, dataset_config, context, dataset, bit, context_failures)
        bit_summaries[str(bit)] = summary
        write_json(paths["cache_root"] / str(bit) / "validator_summary.json", summary)

    passed = all(summary["passed"] for summary in bit_summaries.values())
    aggregate = {
        "stage": "stage4",
        "substage": _stage4_substage(dataset),
        "validator_version": STAGE4_VALIDATOR_VERSION,
        "generated_at_utc": _utc_now(),
        "dataset": dataset,
        "cache_id": config["cache_id"],
        "hash_bits": [int(value) for value in config["hash_bits"]],
        "passed": passed,
        "failure_count": sum(int(summary["failure_count"]) for summary in bit_summaries.values()),
        "bits": bit_summaries,
    }
    return aggregate


def _validate_bit(
    paths: dict[str, Path],
    config: dict[str, Any],
    dataset_config: dict[str, Any],
    context: dict[str, Any],
    dataset: str,
    bit: int,
    context_failures: list[str],
) -> dict[str, Any]:
    failures = list(context_failures)
    bit_dir = paths["cache_root"] / str(bit)
    output_paths = {
        "H_I": bit_dir / "H_I.npy",
        "H_T": bit_dir / "H_T.npy",
        "B_I": bit_dir / "B_I.npy",
        "B_T": bit_dir / "B_T.npy",
        "meta": bit_dir / "meta.json",
    }
    for name, path in output_paths.items():
        if not path.exists():
            failures.append(f"missing Stage 4 output {name}: {path}")

    H_I = H_T = B_I = B_T = None
    meta: dict[str, Any] = {}
    if not failures:
        H_I = np.load(output_paths["H_I"], mmap_mode="r")
        H_T = np.load(output_paths["H_T"], mmap_mode="r")
        B_I = np.load(output_paths["B_I"], mmap_mode="r")
        B_T = np.load(output_paths["B_T"], mmap_mode="r")
        meta = read_json(output_paths["meta"])

    expected_train = int(dataset_config["expected_train_count"])
    h_shape = [expected_train, bit]
    output_stats: dict[str, Any] = {}
    bit_health: dict[str, Any] = {}
    tree_diagnostics: dict[str, Any] = {}
    graph_diagnostics: dict[str, Any] = {}
    tree_risk: dict[str, Any] = {"risk_level": "none", "reasons": []}
    graph_summary: dict[str, Any] = {}

    if not failures:
        assert H_I is not None and H_T is not None and B_I is not None and B_T is not None
        _check_meta(meta, config, dataset_config, context, dataset, bit, failures)
        output_stats["image"] = _check_h_b_pair(H_I, B_I, "image", h_shape, failures)
        output_stats["text"] = _check_h_b_pair(H_T, B_T, "text", h_shape, failures)
        if np.array_equal(np.asarray(H_I), np.asarray(H_T)):
            failures.append("H_I and H_T are exactly identical")
        if np.array_equal(np.asarray(B_I), np.asarray(B_T)):
            failures.append("B_I and B_T are exactly identical")
        bit_health["image"] = _bit_health(np.asarray(H_I), np.asarray(B_I))
        bit_health["text"] = _bit_health(np.asarray(H_T), np.asarray(B_T))
        _check_bit_health(bit_health, config, failures)
        tree_diagnostics = dict(meta.get("tree_diagnostics", {}))
        graph_diagnostics = dict(meta.get("graph_diagnostics", {}))
        tree_risk = _check_tree_diagnostics(tree_diagnostics, dataset_config, config, bit_health, failures)
        graph_summary = _check_graph_diagnostics(graph_diagnostics, failures)

    passed = len(failures) == 0
    return {
        "stage": "stage4",
        "substage": _stage4_substage(dataset),
        "validator_version": STAGE4_VALIDATOR_VERSION,
        "generated_at_utc": _utc_now(),
        "dataset": dataset,
        "bit": bit,
        "passed": passed,
        "failure_count": len(failures),
        "failure_reason": failures,
        "input_boundary": context.get("input_boundary", {}),
        "hash_checks": context.get("hash_checks", {}),
        "output_stats": output_stats,
        "bit_health": bit_health,
        "tree_diagnostics": tree_diagnostics,
        "tree_risk": tree_risk,
        "graph_diagnostics": graph_diagnostics,
        "graph_summary": graph_summary,
        "untrained_sanity_flag": bool(meta.get("stage4_forward_is_untrained_sanity")) if meta else False,
        "not_final_retrieval_flag": bool(meta.get("not_final_retrieval_result")) if meta else False,
    }


def _load_and_check_context(
    paths: dict[str, Path],
    config: dict[str, Any],
    dataset_config: dict[str, Any],
    dataset: str,
    failures: list[str],
) -> dict[str, Any]:
    required = {
        "manifest_filtered": paths["manifest_filtered"],
        "train_ids": paths["train_ids"],
        "order_hashes": paths["order_hashes"],
        "stage2_x_i": paths["x_i"],
        "stage2_x_t": paths["x_t"],
        "stage2_meta": paths["stage2_meta"],
        "stage3_s": paths["s"],
        "stage3_meta": paths["stage3_meta"],
        "stage3_diagnostics": paths["stage3_diagnostics"],
    }
    for name, path in required.items():
        if not path.exists():
            failures.append(f"missing Stage 4 input {name}: {path}")

    context: dict[str, Any] = {"input_boundary": {}, "hash_checks": {}}
    if failures:
        return context

    rows = list(iter_jsonl(paths["manifest_filtered"]))
    train_ids = _read_lines(paths["train_ids"])
    order_hashes = read_json(paths["order_hashes"])
    stage2_meta = read_json(paths["stage2_meta"])
    stage3_meta = read_json(paths["stage3_meta"])
    stage3_diagnostics = read_json(paths["stage3_diagnostics"])
    expected_train = int(dataset_config["expected_train_count"])
    input_dim = int(dataset_config["input_feature_dim"])

    if len(train_ids) != expected_train:
        failures.append(f"train_count mismatch: expected {expected_train}, got {len(train_ids)}")
    sample_ids = [str(row["sample_id"]) for row in rows]
    expected_hashes = {
        "sample_id_order_sha256": hash_lines(sorted(sample_ids)),
        "manifest_filtered_order_sha256": hash_lines(sample_ids),
        "train_ids_sha256": hash_lines(train_ids),
    }
    for key, value in expected_hashes.items():
        if order_hashes.get(key) != value:
            failures.append(f"Stage 1 order_hashes mismatch for {key}")

    train_indices = _train_indices(rows, train_ids, failures)
    if train_indices is not None and stage3_meta.get("train_indices_sha256") != hash_lines(str(int(index)) for index in train_indices):
        failures.append("Stage 3 train_indices_sha256 mismatch")

    x_i = np.load(paths["x_i"], mmap_mode="r")
    x_t = np.load(paths["x_t"], mmap_mode="r")
    S = np.load(paths["s"], mmap_mode="r")
    _check_feature_matrix(x_i, "X_I", input_dim, failures)
    _check_feature_matrix(x_t, "X_T", input_dim, failures)
    if S.shape != (expected_train, expected_train):
        failures.append(f"S shape mismatch: expected {[expected_train, expected_train]}, got {list(S.shape)}")
    if S.dtype != np.float32:
        failures.append(f"S dtype mismatch: expected float32, got {S.dtype}")
    if not np.isfinite(S).all():
        failures.append("S contains NaN or Inf")

    stage1_hash_match = (
        stage2_meta.get("manifest_filtered_order_sha256") == order_hashes.get("manifest_filtered_order_sha256")
        and stage2_meta.get("train_ids_sha256") == order_hashes.get("train_ids_sha256")
        and stage3_meta.get("stage1_manifest_filtered_order_sha256") == order_hashes.get("manifest_filtered_order_sha256")
        and stage3_meta.get("stage1_train_ids_sha256") == order_hashes.get("train_ids_sha256")
    )
    stage2_hash_match = (
        stage3_meta.get("stage2_manifest_filtered_order_sha256") == stage2_meta.get("manifest_filtered_order_sha256")
        and stage3_meta.get("stage2_feature_set_id") == stage2_meta.get("feature_set_id")
        and stage2_meta.get("feature_set_id") == config["feature_set_id"]
    )
    stage3_hash_match = (
        stage3_meta.get("semantic_set_id") == config["semantic_set_id"]
        and stage3_meta.get("train_count") == expected_train
        and stage3_meta.get("matrix_shape") == [expected_train, expected_train]
        and bool(stage3_diagnostics.get("semantic_validator_passed")) is True
    )
    if not stage1_hash_match:
        failures.append("Stage 1 hash/meta consistency check failed")
    if not stage2_hash_match:
        failures.append("Stage 2 hash/meta consistency check failed")
    if not stage3_hash_match:
        failures.append("Stage 3 hash/meta consistency check failed")

    context["rows"] = rows
    context["train_ids"] = train_ids
    context["train_indices"] = train_indices
    context["order_hashes"] = order_hashes
    context["stage2_meta"] = stage2_meta
    context["stage3_meta"] = stage3_meta
    context["stage3_diagnostics"] = stage3_diagnostics
    context["input_boundary"] = {
        "stage2_feature_cache_exists": paths["x_i"].exists() and paths["x_t"].exists(),
        "stage3_s_exists": paths["s"].exists(),
        "train_count": len(train_ids),
        "s_shape": list(S.shape),
        "x_i_shape": list(x_i.shape),
        "x_t_shape": list(x_t.shape),
    }
    context["hash_checks"] = {
        "stage1_hash_match": stage1_hash_match,
        "stage2_hash_match": stage2_hash_match,
        "stage3_hash_match": stage3_hash_match,
        "train_mapping_verified": train_indices is not None,
    }
    return context


def _check_h_b_pair(H: np.ndarray, B: np.ndarray, name: str, expected_shape: list[int], failures: list[str]) -> dict[str, Any]:
    stats: dict[str, Any] = {"shape": list(H.shape), "b_shape": list(B.shape)}
    if list(H.shape) != expected_shape:
        failures.append(f"{name} H shape mismatch: expected {expected_shape}, got {list(H.shape)}")
    if list(B.shape) != expected_shape:
        failures.append(f"{name} B shape mismatch: expected {expected_shape}, got {list(B.shape)}")
    if H.dtype != np.float32:
        failures.append(f"{name} H dtype mismatch: expected float32, got {H.dtype}")
    if B.dtype != np.int8:
        failures.append(f"{name} B dtype mismatch: expected int8, got {B.dtype}")
    h_min = float(np.min(H))
    h_max = float(np.max(H))
    stats.update({"h_min": h_min, "h_max": h_max, "h_std": float(np.std(H, dtype=np.float64))})
    if not np.isfinite(H).all():
        failures.append(f"{name} H contains NaN or Inf")
    if h_min < -1.000001 or h_max > 1.000001:
        failures.append(f"{name} H range outside [-1, 1]: min={h_min}, max={h_max}")
    b_unique = sorted(int(value) for value in np.unique(B))
    stats["b_unique"] = b_unique
    if not set(b_unique).issubset({-1, 1}):
        failures.append(f"{name} B contains values outside {{-1,+1}}: {b_unique}")
    expected_B = np.where(H >= 0.0, 1, -1).astype(np.int8)
    sign_rule_ok = bool(np.array_equal(B, expected_B))
    stats["sign_rule_ok"] = sign_rule_ok
    if not sign_rule_ok:
        failures.append(f"{name} sign rule check failed")
    return stats


def _bit_health(H: np.ndarray, B: np.ndarray) -> dict[str, Any]:
    constant_columns = np.all(B == B[0:1, :], axis=0)
    unique_code_ratio = float(np.unique(B, axis=0).shape[0] / B.shape[0])
    bit_mean = np.mean(B.astype(np.float64), axis=0)
    return {
        "constant_bit_ratio": float(np.mean(constant_columns)),
        "unique_code_ratio": unique_code_ratio,
        "bit_mean_abs_max": float(np.max(np.abs(bit_mean))),
        "h_std": float(np.std(H, dtype=np.float64)),
    }


def _check_bit_health(bit_health: dict[str, Any], config: dict[str, Any], failures: list[str]) -> None:
    validation = config["validation"]
    for modality in ["image", "text"]:
        health = bit_health[modality]
        if health["constant_bit_ratio"] > float(validation["constant_bit_ratio_max"]):
            failures.append(f"{modality} constant_bit_ratio too high: {health['constant_bit_ratio']}")
        if health["unique_code_ratio"] < float(validation["unique_code_ratio_min"]):
            failures.append(f"{modality} unique_code_ratio too low: {health['unique_code_ratio']}")
        if health["bit_mean_abs_max"] >= float(validation["bit_mean_abs_max"]):
            failures.append(f"{modality} bit_mean_abs_max too high: {health['bit_mean_abs_max']}")
        if health["h_std"] <= float(validation["h_std_min"]):
            failures.append(f"{modality} H std too low: {health['h_std']}")


def _check_tree_diagnostics(
    diagnostics: dict[str, Any],
    dataset_config: dict[str, Any],
    config: dict[str, Any],
    bit_health: dict[str, Any],
    failures: list[str],
) -> dict[str, Any]:
    reasons: list[str] = []
    validation = config["validation"]
    prototypes = [int(value) for value in dataset_config["tree_prototypes"]]
    if diagnostics.get("tree_level_count") != int(dataset_config["tree_levels"]):
        failures.append("tree_level_count mismatch")
    if diagnostics.get("prototype_shapes") != [[count, int(dataset_config["d_z"])] for count in prototypes]:
        failures.append("prototype_shapes mismatch")
    row_error = float(diagnostics.get("assignment_row_sum_max_error", float("inf")))
    if row_error > float(validation["assignment_row_sum_error_max"]):
        failures.append(f"tree assignment row-sum error too high: {row_error}")

    entropy = [float(value) for value in diagnostics.get("assignment_entropy", [])]
    effective = [int(value) for value in diagnostics.get("effective_prototypes_used", [])]
    empty_counts = [int(value) for value in diagnostics.get("empty_prototype_count", [])]
    if len(entropy) != len(prototypes) or len(effective) != len(prototypes) or len(empty_counts) != len(prototypes):
        failures.append("tree per-level diagnostic length mismatch")
    else:
        for level, prototype_count in enumerate(prototypes):
            entropy_floor = float(validation["tree_entropy_near_zero_ratio"]) * float(np.log(prototype_count))
            if entropy[level] <= entropy_floor:
                reasons.append(f"level {level} assignment entropy near zero")
            min_effective = float(validation["tree_effective_prototype_ratio_min"]) * prototype_count
            if effective[level] < min_effective:
                reasons.append(f"level {level} effective prototypes too low")
            if empty_counts[level] > 0:
                reasons.append(f"level {level} has empty prototypes")

    yz_ratio = float(diagnostics.get("y_z_norm_ratio", float("inf")))
    if yz_ratio > float(validation["tree_y_z_norm_ratio_max"]):
        reasons.append("Y/Z norm ratio too high")
    for modality in ["image", "text"]:
        if bit_health.get(modality, {}).get("unique_code_ratio", 1.0) < float(validation["unique_code_ratio_min"]):
            reasons.append(f"{modality} B unique code ratio too low")

    return {
        "risk_level": "tree_risk" if reasons else "none",
        "reasons": reasons,
    }


def _check_graph_diagnostics(
    diagnostics: dict[str, Any],
    failures: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for modality in ["image", "text"]:
        graph = diagnostics.get(modality, {})
        required = [
            "degree_min",
            "degree_max",
            "degree_mean",
            "adjacency_finite",
            "normalized_graph_finite",
            "isolated_node_count",
            "self_loop_added",
            "no_isolated_train_node_after_self_loop",
        ]
        hubness_required = [
            "degree_p95",
            "degree_p99",
            "degree_max_over_mean",
            "degree_gini",
            "graph_hubness_risk",
        ]
        required.extend(hubness_required)
        for key in required:
            if key not in graph:
                failures.append(f"missing graph diagnostic {modality}.{key}")
        if graph:
            if float(graph.get("degree_min", 0.0)) <= 0.0:
                failures.append(f"{modality} graph degree_min must be positive")
            if int(graph.get("isolated_node_count", 1)) != 0:
                failures.append(f"{modality} graph has isolated nodes")
            if graph.get("adjacency_finite") is not True:
                failures.append(f"{modality} graph adjacency is not finite")
            if graph.get("normalized_graph_finite") is not True:
                failures.append(f"{modality} normalized graph is not finite")
            if graph.get("self_loop_added") is not True:
                failures.append(f"{modality} graph missing self-loop")
            if not np.isfinite(float(graph.get("degree_max_over_mean", float("nan")))):
                failures.append(f"{modality} graph degree_max_over_mean is not finite")
            if graph.get("graph_hubness_risk") not in {"low", "medium", "high"}:
                failures.append(f"{modality} graph_hubness_risk is invalid: {graph.get('graph_hubness_risk')}")
            summary[modality] = {
                "degree_min": float(graph.get("degree_min", 0.0)),
                "degree_max": float(graph.get("degree_max", 0.0)),
                "degree_mean": float(graph.get("degree_mean", 0.0)),
                "degree_p95": float(graph.get("degree_p95", float("nan"))),
                "degree_p99": float(graph.get("degree_p99", float("nan"))),
                "degree_max_over_mean": float(graph.get("degree_max_over_mean", float("nan"))),
                "degree_gini": float(graph.get("degree_gini", float("nan"))),
                "graph_hubness_risk": str(graph.get("graph_hubness_risk")),
                "isolated_node_count": int(graph.get("isolated_node_count", -1)),
                "normalized_graph_finite": bool(graph.get("normalized_graph_finite", False)),
            }
    if summary:
        risks = [
            str(value["graph_hubness_risk"])
            for value in summary.values()
            if isinstance(value, dict) and "graph_hubness_risk" in value
        ]
        if risks:
            summary["graph_hubness_risk"] = _max_hubness_risk(risks)
    return summary


def _check_meta(
    meta: dict[str, Any],
    config: dict[str, Any],
    dataset_config: dict[str, Any],
    context: dict[str, Any],
    dataset: str,
    bit: int,
    failures: list[str],
) -> None:
    required_fields = {
        "dataset",
        "bit",
        "stage4_model_id",
        "cache_id",
        "feature_set_id",
        "semantic_set_id",
        "train_count",
        "input_feature_dim",
        "d_z",
        "cheby_order",
        "tree_levels",
        "tree_prototypes",
        "graph_k_train",
        "beta_tree_injection",
        "hash_bits_all",
        "stage1_hashes",
        "stage2_hashes",
        "stage3_hashes",
        "random_seed",
        "device",
        "dtype",
        "model_eval_mode",
        "stage4_forward_is_untrained_sanity",
        "not_final_retrieval_result",
        "generated_at_utc",
        "tree_diagnostics",
        "graph_diagnostics",
        "hash_checks",
    }
    missing = sorted(required_fields.difference(meta))
    if missing:
        failures.append(f"meta missing required fields: {missing}")
    expected = {
        "dataset": dataset,
        "bit": bit,
        "stage4_model_id": config["stage4_model_id"],
        "cache_id": config["cache_id"],
        "feature_set_id": config["feature_set_id"],
        "semantic_set_id": config["semantic_set_id"],
        "train_count": int(dataset_config["expected_train_count"]),
        "input_feature_dim": int(dataset_config["input_feature_dim"]),
        "d_z": int(dataset_config["d_z"]),
        "cheby_order": int(dataset_config["cheby_order"]),
        "tree_levels": int(dataset_config["tree_levels"]),
        "tree_prototypes": [int(value) for value in dataset_config["tree_prototypes"]],
        "graph_k_train": int(dataset_config["graph_k_train"]),
        "beta_tree_injection": float(dataset_config["beta_tree_injection"]),
        "hash_bits_all": [int(value) for value in config["hash_bits"]],
        "device": config["runtime"]["device"],
        "dtype": config["runtime"]["dtype"],
        "model_eval_mode": True,
        "stage4_forward_is_untrained_sanity": True,
        "not_final_retrieval_result": True,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            failures.append(f"meta {key} mismatch: expected {value}, got {meta.get(key)}")
    if meta.get("hash_checks") != context.get("hash_checks"):
        failures.append("meta hash_checks mismatch")


def _check_stage4_forward_dataset(dataset: str, config: dict[str, Any]) -> None:
    allowed = set(config.get("execution_policy", {}).get("stage4_forward_allowed_datasets", []))
    if dataset not in SUPPORTED_STAGE4_FORWARD_DATASETS or dataset not in allowed:
        raise RuntimeError(f"Stage 4 forward is authorized for mirflickr25k/nuswide/mscoco only; got {dataset}")


def _stage4_substage(dataset: str) -> str:
    if dataset == "nuswide":
        return "stage4b_nuswide_untrained_forward"
    if dataset == "mirflickr25k":
        return "stage4b_mirflickr25k_regression"
    if dataset == "mscoco":
        return "stage4c_mscoco_untrained_forward"
    raise RuntimeError(f"unsupported Stage 4 substage dataset: {dataset}")


def _max_hubness_risk(risks: list[str]) -> str:
    rank = {"low": 0, "medium": 1, "high": 2}
    return max(risks, key=lambda value: rank.get(value, -1), default="low")


def _check_feature_matrix(matrix: np.ndarray, name: str, input_dim: int, failures: list[str]) -> None:
    if matrix.ndim != 2 or matrix.shape[1] != input_dim:
        failures.append(f"{name} feature shape invalid: {list(matrix.shape)}")
    if matrix.dtype != np.float32:
        failures.append(f"{name} dtype mismatch: expected float32, got {matrix.dtype}")


def _train_indices(rows: list[dict[str, Any]], train_ids: list[str], failures: list[str]) -> np.ndarray | None:
    sample_ids = [str(row["sample_id"]) for row in rows]
    id_to_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    indices: list[int] = []
    for sample_id in train_ids:
        index = id_to_index.get(sample_id)
        if index is None:
            failures.append(f"train_id not present in manifest_filtered: {sample_id}")
            return None
        indices.append(index)
    return np.asarray(indices, dtype=np.int64)


def _stage4_paths(processed_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    feature_dir = processed_root / "feature_cache" / str(config["feature_set_id"])
    semantic_dir = processed_root / "semantic_cache" / str(config["semantic_set_id"])
    return {
        "manifest_filtered": processed_root / "manifest" / "manifest_filtered.jsonl",
        "train_ids": processed_root / "splits" / "train_ids.txt",
        "order_hashes": processed_root / "reports" / "order_hashes.json",
        "x_i": feature_dir / "X_I.npy",
        "x_t": feature_dir / "X_T.npy",
        "stage2_meta": feature_dir / "meta.json",
        "s": semantic_dir / "S.npy",
        "stage3_meta": semantic_dir / "meta.json",
        "stage3_diagnostics": semantic_dir / "semantic_diagnostics.json",
        "cache_root": processed_root / "model_cache" / str(config["cache_id"]),
    }


def _read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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

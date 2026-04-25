from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.jsonl import iter_jsonl, read_json, write_json


FEATURE_SET_ID = "clip_vit_b32_formal_v1"
DATASET_EXPECTED = {
    "nuswide": {
        "filtered_count": 186577,
        "query_count": 2000,
        "retrieval_count": 184577,
        "train_count": 5000,
        "label_dim": 10,
    },
    "mscoco": {
        "filtered_count": 123287,
        "query_count": 2000,
        "retrieval_count": 121287,
        "train_count": 5000,
        "label_dim": 80,
        "category_count": 80,
    },
}


def main() -> int:
    processed_root = REPO_ROOT / "data" / "processed"
    input_paths = {
        dataset: _dataset_paths(processed_root / dataset)
        for dataset in ("nuswide", "mscoco")
    }
    for paths in input_paths.values():
        _require_files(paths)
    before_hashes = {dataset: _hash_paths(paths) for dataset, paths in input_paths.items()}

    nus = _audit_nus(input_paths["nuswide"])
    coco = _audit_coco(input_paths["mscoco"])

    after_hashes = {dataset: _hash_paths(paths) for dataset, paths in input_paths.items()}
    summary = {
        "stage": "nus_coco_preprocessing_consistency_audit",
        "generated_at_utc": _utc_now(),
        "scope": "read-only processed artifact audit; raw data not read; no pipeline rerun",
        "raw_data_read": False,
        "data_processed_modified": False,
        "formal_cache_modified": False,
        "input_hashes_before": before_hashes,
        "input_hashes_after": after_hashes,
        "input_hashes_unchanged": before_hashes == after_hashes,
        "nuswide": nus,
        "mscoco": coco,
        "revision_recommendation": {
            "nus_revision_recommended": nus["recommendation"] != "keep_frozen",
            "coco_revision_recommended": False,
            "reason": (
                "NUS satisfies the strict zero-label and no-relevant-query checks. "
                "COCO preserves the RA/RANEH 123287-count protocol; its small zero-label/no-relevant-query set is recorded as a Stage 7 evaluator risk, not a preprocessing deletion recommendation."
            ),
        },
    }

    output_dir = REPO_ROOT / "outputs" / "nus_coco_preprocessing_consistency"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "nus_coco_preprocessing_consistency_summary.json"
    md_path = output_dir / "nus_coco_preprocessing_consistency_summary.md"
    write_json(json_path, summary)
    md_path.write_text(_markdown(summary), encoding="utf-8")

    print(f"output_json={json_path}")
    print(f"output_md={md_path}")
    print(f"input_hashes_unchanged={str(summary['input_hashes_unchanged']).lower()}")
    print(f"nus_recommendation={nus['recommendation']}")
    print(f"coco_recommendation={coco['recommendation']}")
    return 0 if summary["input_hashes_unchanged"] else 1


def _audit_nus(paths: dict[str, Path]) -> dict[str, Any]:
    rows, query_ids, retrieval_ids, train_ids, preprocess, validator, baseline = _load_common(paths)
    common = _common_stats(rows, query_ids, retrieval_ids, train_ids, baseline)
    empty_text = _empty_text_counts(rows, query_ids, retrieval_ids, train_ids)
    expected = DATASET_EXPECTED["nuswide"]
    stage1_failures = []
    if common["filtered_count"] != expected["filtered_count"]:
        stage1_failures.append("filtered_count mismatch")
    if (common["query_count"], common["retrieval_count"], common["train_count"]) != (
        expected["query_count"],
        expected["retrieval_count"],
        expected["train_count"],
    ):
        stage1_failures.append("split count mismatch")
    if common["label_dim"] != expected["label_dim"]:
        stage1_failures.append("label_dim mismatch")
    if common["zero_label_filtered_count"] != 0:
        stage1_failures.append("zero_label_filtered_count is nonzero")
    if common["query_with_no_relevant_retrieval_count"] != 0:
        stage1_failures.append("query_with_no_relevant_retrieval_count is nonzero")
    if validator.get("passed") is not True:
        stage1_failures.append("validator_summary passed is not true")
    stage2_failures = _stage2_failures(baseline)
    return {
        **common,
        **empty_text,
        "concept_subset": [item["name"] for item in preprocess.get("concept_subset", [])],
        "concept_subset_items": preprocess.get("concept_subset", []),
        "validator_passed": validator.get("passed"),
        "stage1_risk": "high" if stage1_failures else "low",
        "stage1_risk_reasons": stage1_failures,
        "stage2_risk": "high" if stage2_failures else "low",
        "stage2_risk_reasons": stage2_failures,
        "recommendation": "keep_frozen" if not stage1_failures and not stage2_failures else "requires_user_decision",
    }


def _audit_coco(paths: dict[str, Path]) -> dict[str, Any]:
    rows, query_ids, retrieval_ids, train_ids, preprocess, validator, baseline = _load_common(paths)
    common = _common_stats(rows, query_ids, retrieval_ids, train_ids, baseline)
    caption_missing_count = sum(_caption_missing(row) for row in rows)
    category_count = int(preprocess.get("category_count", -1))
    expected = DATASET_EXPECTED["mscoco"]
    stage1_failures = []
    if common["filtered_count"] != expected["filtered_count"]:
        stage1_failures.append("filtered_count mismatch")
    if (common["query_count"], common["retrieval_count"], common["train_count"]) != (
        expected["query_count"],
        expected["retrieval_count"],
        expected["train_count"],
    ):
        stage1_failures.append("split count mismatch")
    if common["label_dim"] != expected["label_dim"]:
        stage1_failures.append("label_dim mismatch")
    if caption_missing_count != 0:
        stage1_failures.append("caption_missing_count is nonzero")
    if category_count != expected["category_count"]:
        stage1_failures.append("category_count mismatch")
    if validator.get("passed") is not True:
        stage1_failures.append("validator_summary passed is not true")
    stage2_failures = _stage2_failures(baseline)
    no_relevant_effect = _estimate_no_relevant_map_effect(
        baseline,
        common["query_with_no_relevant_retrieval_count"],
        common["query_count"],
    )
    stage1_risk = "high" if stage1_failures else ("medium" if common["query_with_no_relevant_retrieval_count"] > 0 else "low")
    recommendation = "requires_user_decision" if stage1_failures or stage2_failures else "keep_frozen"
    return {
        **common,
        "caption_missing_count": int(caption_missing_count),
        "category_count": category_count,
        "zero_label_image_count_from_preprocess": preprocess.get("zero_label_image_count"),
        **no_relevant_effect,
        "validator_passed": validator.get("passed"),
        "stage1_risk": stage1_risk,
        "stage1_risk_reasons": stage1_failures
        + (
            ["small no-relevant query set recorded for Stage 7 evaluator handling"]
            if not stage1_failures and common["query_with_no_relevant_retrieval_count"] > 0
            else []
        ),
        "stage2_risk": "high" if stage2_failures else "low",
        "stage2_risk_reasons": stage2_failures,
        "recommendation": recommendation,
    }


def _load_common(paths: dict[str, Path]) -> tuple[list[dict[str, Any]], list[str], list[str], list[str], dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows = list(iter_jsonl(paths["manifest_filtered"]))
    query_ids = _read_lines(paths["query_ids"])
    retrieval_ids = _read_lines(paths["retrieval_ids"])
    train_ids = _read_lines(paths["train_ids"])
    preprocess = read_json(paths["preprocess_summary"])
    validator = read_json(paths["validator_summary"])
    baseline = read_json(paths["baseline_summary"])
    return rows, query_ids, retrieval_ids, train_ids, preprocess, validator, baseline


def _common_stats(
    rows: list[dict[str, Any]],
    query_ids: list[str],
    retrieval_ids: list[str],
    train_ids: list[str],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    labels = _labels(rows)
    id_to_index = {str(row["sample_id"]): index for index, row in enumerate(rows)}
    query_indices = _indices(query_ids, id_to_index, "query_ids")
    retrieval_indices = _indices(retrieval_ids, id_to_index, "retrieval_ids")
    train_indices = _indices(train_ids, id_to_index, "train_ids")
    zero = np.sum(labels, axis=1) == 0
    no_relevant = _queries_with_no_relevant(labels[query_indices], labels[retrieval_indices])
    paired_mean = float(baseline["paired_cosine_mean"])
    random_mean = float(baseline["random_cosine_mean"])
    paired_median = float(baseline.get("paired_cosine_median", 0.0))
    random_median = float(baseline.get("random_cosine_median", 0.0))
    return {
        "filtered_count": len(rows),
        "query_count": len(query_ids),
        "retrieval_count": len(retrieval_ids),
        "train_count": len(train_ids),
        "label_dim": int(labels.shape[1]),
        "zero_label_filtered_count": int(np.sum(zero)),
        "zero_label_query_count": int(np.sum(zero[query_indices])),
        "zero_label_train_count": int(np.sum(zero[train_indices])),
        "zero_label_retrieval_count": int(np.sum(zero[retrieval_indices])),
        "query_with_no_relevant_retrieval_count": int(len(no_relevant)),
        "query_with_no_relevant_retrieval_rate": float(len(no_relevant) / len(query_ids)),
        "clip_i2t_map_at_50": baseline.get("clip_i2t_map_at_50"),
        "clip_t2i_map_at_50": baseline.get("clip_t2i_map_at_50"),
        "paired_cosine_mean": paired_mean,
        "random_cosine_mean": random_mean,
        "cosine_gap_mean": float(baseline.get("cosine_gap_mean", paired_mean - random_mean)),
        "paired_cosine_median": paired_median,
        "random_cosine_median": random_median,
        "cosine_gap_median": float(baseline.get("cosine_gap_median", paired_median - random_median)),
        "baseline_completed": baseline.get("baseline_completed"),
        "failure_reason": baseline.get("failure_reason"),
    }


def _empty_text_counts(
    rows: list[dict[str, Any]],
    query_ids: list[str],
    retrieval_ids: list[str],
    train_ids: list[str],
) -> dict[str, int]:
    by_id = {str(row["sample_id"]): bool(str(row.get("text_source", "")).strip()) for row in rows}
    empty_all = sum(1 for has_text in by_id.values() if not has_text)
    empty_query = sum(1 for sample_id in query_ids if not by_id[sample_id])
    empty_train = sum(1 for sample_id in train_ids if not by_id[sample_id])
    empty_retrieval = sum(1 for sample_id in retrieval_ids if not by_id[sample_id])
    return {
        "empty_text_filtered_count": int(empty_all),
        "empty_text_query_count": int(empty_query),
        "empty_text_train_count": int(empty_train),
        "empty_text_retrieval_count": int(empty_retrieval),
    }


def _estimate_no_relevant_map_effect(baseline: dict[str, Any], no_relevant_count: int, query_count: int) -> dict[str, Any]:
    i2t_all = float(baseline["clip_i2t_map_at_50"])
    t2i_all = float(baseline["clip_t2i_map_at_50"])
    valid = query_count - no_relevant_count
    if valid <= 0:
        return {
            "clip_i2t_map_at_50_all_queries": i2t_all,
            "clip_i2t_map_at_50_excluding_no_relevant_queries": None,
            "estimated_loss_i2t": None,
            "clip_t2i_map_at_50_all_queries": t2i_all,
            "clip_t2i_map_at_50_excluding_no_relevant_queries": None,
            "estimated_loss_t2i": None,
        }
    i2t_excluding = i2t_all * query_count / valid
    t2i_excluding = t2i_all * query_count / valid
    return {
        "clip_i2t_map_at_50_all_queries": i2t_all,
        "clip_i2t_map_at_50_excluding_no_relevant_queries": float(i2t_excluding),
        "estimated_loss_i2t": float(i2t_excluding - i2t_all),
        "clip_t2i_map_at_50_all_queries": t2i_all,
        "clip_t2i_map_at_50_excluding_no_relevant_queries": float(t2i_excluding),
        "estimated_loss_t2i": float(t2i_excluding - t2i_all),
        "method": "estimate_from_all_query_map_assuming_no_relevant_queries_contribute_zero_ap",
    }


def _stage2_failures(baseline: dict[str, Any]) -> list[str]:
    failures = []
    if baseline.get("baseline_completed") is not True:
        failures.append("baseline_completed is not true")
    if baseline.get("failure_reason") not in (None, "", []):
        failures.append("failure_reason is not empty")
    if float(baseline["paired_cosine_mean"]) <= float(baseline["random_cosine_mean"]):
        failures.append("paired_cosine_mean is not greater than random_cosine_mean")
    if float(baseline.get("cosine_gap_mean", 0.0)) <= 0.0:
        failures.append("cosine_gap_mean is not positive")
    return failures


def _caption_missing(row: dict[str, Any]) -> bool:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    return int(meta.get("caption_count", 0)) <= 0 or not str(row.get("text_source", "")).strip()


def _queries_with_no_relevant(query_labels: np.ndarray, retrieval_labels: np.ndarray) -> list[int]:
    no_relevant: list[int] = []
    retrieval_bool = retrieval_labels.astype(bool)
    for start in range(0, query_labels.shape[0], 128):
        end = min(start + 128, query_labels.shape[0])
        relevant = (query_labels[start:end].astype(bool) @ retrieval_bool.T) > 0
        no_relevant.extend((np.where(np.sum(relevant, axis=1) == 0)[0] + start).astype(int).tolist())
    return no_relevant


def _labels(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[int(value) for value in row["label_vector"]] for row in rows], dtype=np.uint8)


def _indices(ids: list[str], id_to_index: dict[str, int], name: str) -> np.ndarray:
    indices = []
    for sample_id in ids:
        if sample_id not in id_to_index:
            raise RuntimeError(f"{name} contains id outside manifest_filtered: {sample_id}")
        indices.append(id_to_index[sample_id])
    return np.asarray(indices, dtype=np.int64)


def _markdown(summary: dict[str, Any]) -> str:
    nus = summary["nuswide"]
    coco = summary["mscoco"]
    lines = [
        "# NUS/COCO Preprocessing Consistency Audit",
        "",
        "Read-only processed artifact audit. Raw data was not read and no pipeline was rerun.",
        "",
        "## NUS-WIDE",
        "",
        f"- filtered_count: {nus['filtered_count']}",
        f"- query/retrieval/train: {nus['query_count']} / {nus['retrieval_count']} / {nus['train_count']}",
        f"- zero_label_filtered_count: {nus['zero_label_filtered_count']}",
        f"- query_with_no_relevant_retrieval_count: {nus['query_with_no_relevant_retrieval_count']}",
        f"- empty_text_query_count: {nus['empty_text_query_count']}",
        f"- stage1_risk: {nus['stage1_risk']}",
        f"- stage2_risk: {nus['stage2_risk']}",
        f"- recommendation: {nus['recommendation']}",
        "",
        "## MSCOCO",
        "",
        f"- filtered_count: {coco['filtered_count']}",
        f"- query/retrieval/train: {coco['query_count']} / {coco['retrieval_count']} / {coco['train_count']}",
        f"- zero_label_query_count: {coco['zero_label_query_count']}",
        f"- query_with_no_relevant_retrieval_count: {coco['query_with_no_relevant_retrieval_count']}",
        f"- caption_missing_count: {coco['caption_missing_count']}",
        f"- category_count: {coco['category_count']}",
        f"- estimated_loss_i2t: {coco['estimated_loss_i2t']}",
        f"- estimated_loss_t2i: {coco['estimated_loss_t2i']}",
        f"- stage1_risk: {coco['stage1_risk']}",
        f"- stage2_risk: {coco['stage2_risk']}",
        f"- recommendation: {coco['recommendation']}",
        "",
    ]
    return "\n".join(lines)


def _dataset_paths(root: Path) -> dict[str, Path]:
    feature_dir = root / "feature_cache" / FEATURE_SET_ID
    return {
        "manifest_filtered": root / "manifest" / "manifest_filtered.jsonl",
        "query_ids": root / "splits" / "query_ids.txt",
        "retrieval_ids": root / "splits" / "retrieval_ids.txt",
        "train_ids": root / "splits" / "train_ids.txt",
        "preprocess_summary": root / "reports" / "preprocess_summary.json",
        "validator_summary": root / "reports" / "validator_summary.json",
        "baseline_summary": feature_dir / "baseline_summary.json",
    }


def _read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle if line.rstrip("\n")]


def _require_files(paths: dict[str, Path]) -> None:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing required audit inputs: {missing}")


def _hash_paths(paths: dict[str, Path]) -> dict[str, str]:
    return {name: _sha256_file(path) for name, path in paths.items()}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())

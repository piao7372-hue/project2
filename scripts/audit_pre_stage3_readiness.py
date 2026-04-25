from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.jsonl import iter_jsonl, read_json, write_json


DATASETS = ["mirflickr25k", "nuswide", "mscoco"]
FEATURE_SET_ID = "clip_vit_b32_formal_v1"
EXPECTED = {
    "mirflickr25k": {"filtered": 20015, "query": 2000, "retrieval": 18015, "train": 5000},
    "nuswide": {"filtered": 186577, "query": 2000, "retrieval": 184577, "train": 5000},
    "mscoco": {"filtered": 123287, "query": 2000, "retrieval": 121287, "train": 5000},
}
RA_128_TARGETS = {
    "mirflickr25k": {"i2t": 0.961, "t2i": 0.926},
    "nuswide": {"i2t": 0.876, "t2i": 0.837},
    "mscoco": {"i2t": 0.958, "t2i": 0.969},
}


def main() -> int:
    processed_root = REPO_ROOT / "data" / "processed"
    rows = [_audit_dataset(processed_root, dataset) for dataset in DATASETS]
    summary = {
        "stage": "pre_stage3_readiness_audit",
        "generated_at_utc": _utc_now(),
        "scope": "read-only Stage 1/2 frozen artifact audit",
        "raw_data_read": False,
        "stage1_or_stage2_outputs_modified": False,
        "datasets": {row["dataset"]: row for row in rows},
    }
    output_dir = REPO_ROOT / "outputs" / "pre_stage3_readiness"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "pre_stage3_readiness_summary.json", summary)
    (output_dir / "pre_stage3_readiness_summary.md").write_text(_markdown(summary), encoding="utf-8")

    print(f"output_json={output_dir / 'pre_stage3_readiness_summary.json'}")
    print(f"output_md={output_dir / 'pre_stage3_readiness_summary.md'}")
    for row in rows:
        print(
            f"{row['dataset']}: stage1_data_risk={row['stage1_data_risk']} "
            f"stage2_feature_risk={row['stage2_feature_risk']} action={row['recommended_action']}"
        )
    return 0


def _audit_dataset(processed_root: Path, dataset: str) -> dict[str, Any]:
    root = processed_root / dataset
    manifest_path = root / "manifest" / "manifest_filtered.jsonl"
    query_path = root / "splits" / "query_ids.txt"
    retrieval_path = root / "splits" / "retrieval_ids.txt"
    train_path = root / "splits" / "train_ids.txt"
    preprocess_summary_path = root / "reports" / "preprocess_summary.json"
    stage1_validator_path = root / "reports" / "validator_summary.json"
    stage2_meta_path = root / "feature_cache" / FEATURE_SET_ID / "meta.json"
    baseline_path = root / "feature_cache" / FEATURE_SET_ID / "baseline_summary.json"
    _require_files(
        [
            manifest_path,
            query_path,
            retrieval_path,
            train_path,
            preprocess_summary_path,
            stage1_validator_path,
            stage2_meta_path,
            baseline_path,
        ]
    )

    rows = list(iter_jsonl(manifest_path))
    query_ids = _read_lines(query_path)
    retrieval_ids = _read_lines(retrieval_path)
    train_ids = _read_lines(train_path)
    preprocess_summary = read_json(preprocess_summary_path)
    stage1_validator = read_json(stage1_validator_path)
    stage2_meta = read_json(stage2_meta_path)
    baseline = read_json(baseline_path)

    sample_ids = [str(row["sample_id"]) for row in rows]
    id_to_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    query_indices = _indices(query_ids, id_to_index, "query_ids")
    retrieval_indices = _indices(retrieval_ids, id_to_index, "retrieval_ids")
    train_indices = _indices(train_ids, id_to_index, "train_ids")
    labels = np.asarray([[int(value) for value in row["label_vector"]] for row in rows], dtype=np.uint8)
    label_positive = np.sum(labels, axis=1)
    zero_label = label_positive == 0
    no_relevant = _queries_with_no_relevant(labels[query_indices], labels[retrieval_indices])

    expected = EXPECTED[dataset]
    count_ok = (
        len(rows) == expected["filtered"]
        and len(query_ids) == expected["query"]
        and len(retrieval_ids) == expected["retrieval"]
        and len(train_ids) == expected["train"]
    )
    stage1_failures: list[str] = []
    if not count_ok:
        stage1_failures.append("Stage 1 count mismatch")
    if len(no_relevant) > 0:
        stage1_failures.append("queries with no relevant retrieval exist")
    if bool(stage1_validator.get("passed")) is False:
        stage1_failures.append("Stage 1 validator did not pass")

    stage2_failures: list[str] = []
    if baseline.get("baseline_completed") is not True:
        stage2_failures.append("baseline_completed is false")
    if baseline.get("failure_reason") not in (None, "", []):
        stage2_failures.append("baseline failure_reason is not empty")
    if float(baseline.get("cosine_gap_mean", 0.0)) <= 0.0:
        stage2_failures.append("cosine_gap_mean is not positive")
    if float(baseline.get("cosine_gap_median", 0.0)) <= 0.0:
        stage2_failures.append("cosine_gap_median is not positive")
    if stage2_meta.get("feature_set_id") != FEATURE_SET_ID:
        stage2_failures.append("Stage 2 feature_set_id mismatch")

    stage1_risk = _stage1_risk(stage1_failures, len(no_relevant), len(query_ids), dataset, rows, query_indices, train_indices, retrieval_indices)
    stage2_risk = "high" if stage2_failures else "low"
    recommended_action = _recommended_action(stage1_risk, stage2_risk)

    result = {
        "dataset": dataset,
        "filtered_count": len(rows),
        "query_count": len(query_ids),
        "retrieval_count": len(retrieval_ids),
        "train_count": len(train_ids),
        "label_dim": int(labels.shape[1]),
        "label_positive_count_mean": float(np.mean(label_positive)),
        "label_positive_count_median": float(np.median(label_positive)),
        "zero_label_filtered_count": int(np.sum(zero_label)),
        "zero_label_query_count": int(np.sum(zero_label[query_indices])),
        "zero_label_train_count": int(np.sum(zero_label[train_indices])),
        "zero_label_retrieval_count": int(np.sum(zero_label[retrieval_indices])),
        "query_with_no_relevant_retrieval_count": int(len(no_relevant)),
        "query_with_no_relevant_retrieval_rate": float(len(no_relevant) / len(query_ids)),
        "stage1_data_risk": stage1_risk,
        "stage1_risk_reasons": stage1_failures,
        "stage2_feature_risk": stage2_risk,
        "stage2_risk_reasons": stage2_failures,
        "recommended_action": recommended_action,
        "paired_cosine_mean": baseline.get("paired_cosine_mean"),
        "random_cosine_mean": baseline.get("random_cosine_mean"),
        "cosine_gap_mean": baseline.get("cosine_gap_mean"),
        "paired_cosine_median": baseline.get("paired_cosine_median"),
        "random_cosine_median": baseline.get("random_cosine_median"),
        "cosine_gap_median": baseline.get("cosine_gap_median"),
        "clip_i2t_map_at_50": baseline.get("clip_i2t_map_at_50"),
        "clip_t2i_map_at_50": baseline.get("clip_t2i_map_at_50"),
        "baseline_completed": baseline.get("baseline_completed"),
        "failure_reason": baseline.get("failure_reason"),
        "ra_128_i2t_target": RA_128_TARGETS[dataset]["i2t"],
        "ra_128_t2i_target": RA_128_TARGETS[dataset]["t2i"],
        "clip_to_ra_gap_i2t": float(RA_128_TARGETS[dataset]["i2t"] - float(baseline.get("clip_i2t_map_at_50"))),
        "clip_to_ra_gap_t2i": float(RA_128_TARGETS[dataset]["t2i"] - float(baseline.get("clip_t2i_map_at_50"))),
        "downstream_challenge_note": "CLIP-to-RA gap is diagnostic only and is not a Stage 1/2 failure condition.",
    }
    if dataset == "nuswide":
        empty_text = np.asarray([not str(row.get("text_source", "")).strip() for row in rows], dtype=bool)
        result.update(
            {
                "empty_text_filtered_count": int(np.sum(empty_text)),
                "empty_text_query_count": int(np.sum(empty_text[query_indices])),
                "empty_text_train_count": int(np.sum(empty_text[train_indices])),
                "empty_text_retrieval_count": int(np.sum(empty_text[retrieval_indices])),
                "empty_text_query_rate": float(np.mean(empty_text[query_indices])),
                "empty_text_train_rate": float(np.mean(empty_text[train_indices])),
            }
        )
        if result["empty_text_query_rate"] > 0.01 or result["empty_text_train_rate"] > 0.01:
            result["stage1_data_risk"] = "medium" if result["stage1_data_risk"] == "low" else result["stage1_data_risk"]
            result["stage1_risk_reasons"].append("NUS empty_text split rate above 1%")
            result["recommended_action"] = _recommended_action(result["stage1_data_risk"], result["stage2_feature_risk"])
    if dataset == "mscoco":
        result["zero_label_image_count"] = int(preprocess_summary.get("zero_label_image_count", result["zero_label_filtered_count"]))
    return result


def _recommended_action(stage1_risk: str, stage2_risk: str) -> str:
    if stage1_risk == "high" or stage2_risk == "high":
        return "requires_user_decision"
    if stage1_risk == "medium" or stage2_risk == "medium":
        return "audit_later"
    return "keep_frozen"


def _queries_with_no_relevant(query_labels: np.ndarray, retrieval_labels: np.ndarray) -> list[int]:
    no_relevant = []
    retrieval_bool = retrieval_labels.astype(bool)
    for start in range(0, query_labels.shape[0], 128):
        end = min(start + 128, query_labels.shape[0])
        relevant = (query_labels[start:end].astype(bool) @ retrieval_bool.T) > 0
        no_relevant.extend((np.where(np.sum(relevant, axis=1) == 0)[0] + start).astype(int).tolist())
    return no_relevant


def _stage1_risk(
    failures: list[str],
    no_relevant_count: int,
    query_count: int,
    dataset: str,
    rows: list[dict[str, Any]],
    query_indices: np.ndarray,
    train_indices: np.ndarray,
    retrieval_indices: np.ndarray,
) -> str:
    if failures and any("count mismatch" in failure or "validator" in failure for failure in failures):
        return "high"
    if no_relevant_count / query_count > 0.01:
        return "high"
    if no_relevant_count > 0:
        return "medium"
    return "low"


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Pre-Stage3 Readiness Audit",
        "",
        "Read-only audit of frozen Stage 1/2 artifacts. Raw data was not read.",
        "",
        "| Dataset | Stage1 risk | Stage2 risk | no-relevant query count | CLIP I->T mAP@50 | CLIP T->I mAP@50 | Action |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in summary["datasets"].values():
        lines.append(
            f"| {row['dataset']} | {row['stage1_data_risk']} | {row['stage2_feature_risk']} | "
            f"{row['query_with_no_relevant_retrieval_count']} | {row['clip_i2t_map_at_50']} | "
            f"{row['clip_t2i_map_at_50']} | {row['recommended_action']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _indices(ids: list[str], id_to_index: dict[str, int], name: str) -> np.ndarray:
    indices = []
    for sample_id in ids:
        if sample_id not in id_to_index:
            raise RuntimeError(f"{name} contains id outside manifest: {sample_id}")
        indices.append(id_to_index[sample_id])
    return np.asarray(indices, dtype=np.int64)


def _read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle if line.rstrip("\n")]


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing required files: {missing}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())

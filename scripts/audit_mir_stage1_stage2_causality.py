from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.jsonl import iter_jsonl, read_json, write_json


DATASET = "mirflickr25k"
EXPECTED_FILTERED_COUNT = 20015
QUERY_COUNT = 2000
TRAIN_COUNT = 5000
SPLIT_SEED = 0


def main() -> int:
    root = REPO_ROOT / "data" / "processed" / DATASET
    raw_root = REPO_ROOT / "data" / "raw" / DATASET
    paths = {
        "raw_root": raw_root,
        "manifest_raw": root / "manifest" / "manifest_raw.jsonl",
        "manifest_filtered": root / "manifest" / "manifest_filtered.jsonl",
        "query_ids": root / "splits" / "query_ids.txt",
        "retrieval_ids": root / "splits" / "retrieval_ids.txt",
        "train_ids": root / "splits" / "train_ids.txt",
        "preprocess_summary": root / "reports" / "preprocess_summary.json",
        "validator_summary": root / "reports" / "validator_summary.json",
        "baseline_summary": root / "feature_cache" / "clip_vit_b32_formal_v1" / "baseline_summary.json",
        "semantic_diagnostics": root / "semantic_cache" / "se_c_s_formal_v1" / "semantic_diagnostics.json",
    }
    _require_inputs(paths)
    before_hashes = _hash_readonly_inputs(paths)

    raw_rows = list(iter_jsonl(paths["manifest_raw"]))
    filtered_rows = list(iter_jsonl(paths["manifest_filtered"]))
    query_ids = _read_lines(paths["query_ids"])
    retrieval_ids = _read_lines(paths["retrieval_ids"])
    train_ids = _read_lines(paths["train_ids"])
    preprocess_summary = read_json(paths["preprocess_summary"])
    validator_summary = read_json(paths["validator_summary"])
    baseline_summary = read_json(paths["baseline_summary"])
    semantic_diagnostics = read_json(paths["semantic_diagnostics"])

    current = _current_risk(len(raw_rows), filtered_rows, query_ids, retrieval_ids, train_ids)
    candidates = _candidate_audit(raw_rows)
    no_relevant_map = _estimate_no_relevant_map_effect(
        baseline_summary=baseline_summary,
        no_relevant_count=current["query_with_no_relevant_retrieval_count"],
        query_count=current["query_count"],
    )
    stage2 = _stage2_feature_quality(baseline_summary)
    recommendation = _recommend(candidates, stage2)
    after_hashes = _hash_readonly_inputs(paths)

    summary = {
        "stage": "mir_stage1_stage2_causality_audit",
        "dataset": DATASET,
        "generated_at_utc": _utc_now(),
        "scope": "audit_only_no_stage1_stage2_stage3_formal_mutation",
        "raw_root_read": _repo_relative(raw_root),
        "formal_inputs": {name: _repo_relative(path) for name, path in paths.items() if name != "raw_root"},
        "readonly_input_hashes_before": before_hashes,
        "readonly_input_hashes_after": after_hashes,
        "readonly_inputs_unchanged": before_hashes == after_hashes,
        "current_mir_risk": current,
        "candidate_filtering_audit": candidates,
        "no_relevant_query_map_effect": no_relevant_map,
        "stage2_feature_quality": stage2,
        "recommendation": recommendation,
        "stage1_context": {
            "preprocess_filter_policy": preprocess_summary.get("filter_policy"),
            "stage1_validator_passed": validator_summary.get("passed"),
            "stage3_semantic_diag_mean_s": semantic_diagnostics.get("diag_mean_s"),
            "stage3_semantic_offdiag_mean_s": semantic_diagnostics.get("offdiag_mean_s"),
            "stage3_semantic_diag_over_offdiag_ratio": semantic_diagnostics.get("diag_over_offdiag_ratio"),
        },
        "forbidden_stage2_direct_optimizations": {
            "change_clip_backbone": False,
            "change_tokenizer": False,
            "change_image_preprocessing": False,
            "add_prompt_templates": False,
            "rewrite_text_source": False,
            "skip_samples": False,
            "zero_vector_fallback": False,
            "cpu_fallback": False,
        },
        "formal_artifacts_modified": {
            "data_processed": False,
            "data_raw": False,
            "models": False,
            "stage1": False,
            "stage2": False,
            "stage3": False,
        },
    }

    output_dir = REPO_ROOT / "outputs" / "mir_stage1_stage2_causality"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "mir_stage1_stage2_causality_summary.json"
    md_path = output_dir / "mir_stage1_stage2_causality_summary.md"
    write_json(json_path, summary)
    md_path.write_text(_markdown(summary), encoding="utf-8")

    print(f"output_json={json_path}")
    print(f"output_md={md_path}")
    print(f"readonly_inputs_unchanged={str(summary['readonly_inputs_unchanged']).lower()}")
    print(f"stage1_revision_recommended={str(recommendation['stage1_revision_recommended']).lower()}")
    print(f"stage2_direct_revision_recommended={str(recommendation['stage2_direct_revision_recommended']).lower()}")
    print(f"recommended_action={recommendation['recommended_action']}")
    return 0 if summary["readonly_inputs_unchanged"] else 1


def _current_risk(
    raw_count: int,
    filtered_rows: list[dict[str, Any]],
    query_ids: list[str],
    retrieval_ids: list[str],
    train_ids: list[str],
) -> dict[str, Any]:
    by_id = {str(row["sample_id"]): row for row in filtered_rows}
    labels = _labels(filtered_rows)
    id_to_index = {str(row["sample_id"]): index for index, row in enumerate(filtered_rows)}
    query_indices = _indices(query_ids, id_to_index, "query_ids")
    retrieval_indices = _indices(retrieval_ids, id_to_index, "retrieval_ids")
    train_indices = _indices(train_ids, id_to_index, "train_ids")
    zero = np.sum(labels, axis=1) == 0
    no_relevant = _queries_with_no_relevant(labels[query_indices], labels[retrieval_indices])
    return {
        "manifest_raw_count": raw_count,
        "manifest_filtered_count": len(filtered_rows),
        "query_count": len(query_ids),
        "retrieval_count": len(retrieval_ids),
        "train_count": len(train_ids),
        "zero_label_filtered_count": int(np.sum(zero)),
        "zero_label_query_count": int(np.sum(zero[query_indices])),
        "zero_label_train_count": int(np.sum(zero[train_indices])),
        "zero_label_retrieval_count": int(np.sum(zero[retrieval_indices])),
        "train_zero_label_count": int(np.sum(zero[train_indices])),
        "retrieval_zero_label_count": int(np.sum(zero[retrieval_indices])),
        "query_with_no_relevant_retrieval_count": int(len(no_relevant)),
        "query_with_no_relevant_retrieval_rate": float(len(no_relevant) / len(query_ids)),
        "zero_label_query_ids": [str(query_ids[index]) for index in no_relevant if np.sum(np.asarray(by_id[str(query_ids[index])]["label_vector"], dtype=np.int64)) == 0],
    }


def _candidate_audit(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    current_policy_rows = _current_policy_rows(raw_rows)
    nonempty_label_positive = [row for row in raw_rows if _has_text(row) and _label_sum(row) > 0]
    candidates = [
        ("candidate_1_current_pragmatic_high_signal_v1", current_policy_rows, len(current_policy_rows), False),
        (
            "candidate_2_tag20_only",
            [row for row in raw_rows if int(row.get("meta", {}).get("raw_tag_token_count", 0)) >= 20],
            None,
            False,
        ),
        (
            "candidate_3_tag20_and_label_positive",
            [row for row in raw_rows if int(row.get("meta", {}).get("raw_tag_token_count", 0)) >= 20 and _label_sum(row) > 0],
            None,
            True,
        ),
        ("candidate_4_nonempty_text_and_label_positive", nonempty_label_positive, None, True),
        (
            "candidate_5_nonempty_text_label_positive_then_current_sort_truncate",
            _truncate_current_rank(nonempty_label_positive),
            len(nonempty_label_positive),
            True,
        ),
    ]
    rows = [_candidate_stats(name, rows, source_count, ra_like_positive_label) for name, rows, source_count, ra_like_positive_label in candidates]
    selected = _select_revision_candidate(rows)
    for row in rows:
        row["selected_revision_candidate"] = row["candidate_name"] == selected
    return rows


def _candidate_stats(name: str, rows: list[dict[str, Any]], source_count: int | None, ra_like_positive_label: bool) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: str(row["sample_id"]))
    labels = _labels(rows)
    zero = np.sum(labels, axis=1) == 0 if labels.size else np.asarray([], dtype=bool)
    can_reach = len(rows) >= EXPECTED_FILTERED_COUNT
    stats = {
        "candidate_name": name,
        "candidate_count": len(rows),
        "source_candidate_count_before_truncate": source_count if source_count is not None else len(rows),
        "can_reach_20015": bool(can_reach),
        "zero_label_count": int(np.sum(zero)),
        "mean_raw_tag_token_count": _mean_meta(rows, "raw_tag_token_count"),
        "median_raw_tag_token_count": _median_meta(rows, "raw_tag_token_count"),
        "min_raw_tag_token_count": _min_meta(rows, "raw_tag_token_count"),
        "mean_annotation_positive_count": _mean_meta(rows, "annotation_positive_count"),
        "median_annotation_positive_count": _median_meta(rows, "annotation_positive_count"),
        "query_count_under_seed0": None,
        "retrieval_count_under_seed0": None,
        "train_count_under_seed0": None,
        "zero_label_query_count_under_seed0_split": None,
        "zero_label_train_count_under_seed0_split": None,
        "zero_label_retrieval_count_under_seed0_split": None,
        "query_with_no_relevant_retrieval_count_under_seed0_split": None,
        "query_with_no_relevant_retrieval_rate_under_seed0_split": None,
        "strong_revision_candidate": False,
        "ra_like_positive_label_filter": ra_like_positive_label,
    }
    if not can_reach:
        return stats
    split = _make_split([str(row["sample_id"]) for row in rows], SPLIT_SEED, QUERY_COUNT, TRAIN_COUNT)
    id_to_index = {str(row["sample_id"]): index for index, row in enumerate(rows)}
    query_indices = _indices(split["query_ids"], id_to_index, "candidate_query_ids")
    retrieval_indices = _indices(split["retrieval_ids"], id_to_index, "candidate_retrieval_ids")
    train_indices = _indices(split["train_ids"], id_to_index, "candidate_train_ids")
    no_relevant = _queries_with_no_relevant(labels[query_indices], labels[retrieval_indices])
    zero_query_count = int(np.sum(zero[query_indices]))
    stats.update(
        {
            "query_count_under_seed0": len(split["query_ids"]),
            "retrieval_count_under_seed0": len(split["retrieval_ids"]),
            "train_count_under_seed0": len(split["train_ids"]),
            "zero_label_query_count_under_seed0_split": zero_query_count,
            "zero_label_train_count_under_seed0_split": int(np.sum(zero[train_indices])),
            "zero_label_retrieval_count_under_seed0_split": int(np.sum(zero[retrieval_indices])),
            "query_with_no_relevant_retrieval_count_under_seed0_split": int(len(no_relevant)),
            "query_with_no_relevant_retrieval_rate_under_seed0_split": float(len(no_relevant) / len(split["query_ids"])),
        }
    )
    stats["strong_revision_candidate"] = bool(
        ra_like_positive_label
        and can_reach
        and zero_query_count == 0
        and len(no_relevant) == 0
        and int(np.sum(zero)) == 0
        and int(np.sum(zero[train_indices])) == 0
        and int(np.sum(zero[retrieval_indices])) == 0
        and len(rows) >= EXPECTED_FILTERED_COUNT
    )
    return stats


def _estimate_no_relevant_map_effect(baseline_summary: dict[str, Any], no_relevant_count: int, query_count: int) -> dict[str, Any]:
    valid_query_count = query_count - no_relevant_count
    if valid_query_count <= 0:
        raise RuntimeError("No valid queries remain after excluding no-relevant queries")
    i2t_all = float(baseline_summary["clip_i2t_map_at_50"])
    t2i_all = float(baseline_summary["clip_t2i_map_at_50"])
    scale = query_count / valid_query_count
    i2t_excluding = i2t_all * scale
    t2i_excluding = t2i_all * scale
    return {
        "method": "estimate_from_all_query_map_assuming_no_relevant_queries_contribute_zero_ap",
        "query_count": query_count,
        "excluded_no_relevant_query_count": no_relevant_count,
        "remaining_query_count": valid_query_count,
        "clip_i2t_map_at_50_all_queries": i2t_all,
        "clip_t2i_map_at_50_all_queries": t2i_all,
        "clip_i2t_map_at_50_excluding_no_relevant_queries": float(i2t_excluding),
        "clip_t2i_map_at_50_excluding_no_relevant_queries": float(t2i_excluding),
        "estimated_map_loss_from_no_relevant_queries_i2t": float(i2t_excluding - i2t_all),
        "estimated_map_loss_from_no_relevant_queries_t2i": float(t2i_excluding - t2i_all),
    }


def _stage2_feature_quality(baseline_summary: dict[str, Any]) -> dict[str, Any]:
    completed = baseline_summary.get("baseline_completed") is True
    failure_reason = baseline_summary.get("failure_reason")
    gap_mean = float(baseline_summary["cosine_gap_mean"])
    gap_median = float(baseline_summary["cosine_gap_median"])
    risk = "low" if completed and not failure_reason and gap_mean > 0.0 and gap_median > 0.0 else "high"
    return {
        "paired_cosine_mean": baseline_summary.get("paired_cosine_mean"),
        "random_cosine_mean": baseline_summary.get("random_cosine_mean"),
        "cosine_gap_mean": baseline_summary.get("cosine_gap_mean"),
        "paired_cosine_median": baseline_summary.get("paired_cosine_median"),
        "random_cosine_median": baseline_summary.get("random_cosine_median"),
        "cosine_gap_median": baseline_summary.get("cosine_gap_median"),
        "clip_i2t_map_at_50": baseline_summary.get("clip_i2t_map_at_50"),
        "clip_t2i_map_at_50": baseline_summary.get("clip_t2i_map_at_50"),
        "baseline_completed": completed,
        "failure_reason": failure_reason,
        "stage2_feature_risk": risk,
        "judgment": "Stage 2 feature risk remains low because paired/random cosine gaps are positive; RA final is trained binary Hamming retrieval, not a CLIP cosine baseline.",
    }


def _recommend(candidates: list[dict[str, Any]], stage2: dict[str, Any]) -> dict[str, Any]:
    selected = _select_revision_candidate(candidates)
    strong = [candidate for candidate in candidates if candidate["strong_revision_candidate"]]
    stage1_revision = selected is not None
    if stage1_revision:
        action = "Revise MIR Stage 1 filtering, then rerun MIR Stage 1/2/3 only. Do not modify NUS/COCO."
        reason = (
            "At least one positive-label MIR filtering candidate can preserve the required sample count "
            "and removes zero-label/no-relevant query cases under the frozen seed0 split. "
            "This targets evaluator relevance validity, not CLIP feature tuning."
        )
    else:
        action = "Keep Stage 1/2 frozen. Continue Stage 3B/C. Record MIR no-relevant query handling requirement for Stage 7 evaluator."
        reason = "No candidate met all strong revision conditions."
    return {
        "stage1_revision_recommended": stage1_revision,
        "stage2_direct_revision_recommended": False,
        "recommended_action": action,
        "reason": reason,
        "selected_revision_candidate": selected,
        "strong_revision_candidates": [candidate["candidate_name"] for candidate in strong],
        "stage2_direct_revision_reason": (
            "Direct Stage 2 optimization is not recommended; if Stage 1 changes, rerun the same formal CLIP protocol only. "
            f"Current Stage 2 feature risk: {stage2['stage2_feature_risk']}."
        ),
    }


def _select_revision_candidate(candidates: list[dict[str, Any]]) -> str | None:
    by_name = {candidate["candidate_name"]: candidate for candidate in candidates}
    for name in (
        "candidate_3_tag20_and_label_positive",
        "candidate_5_nonempty_text_label_positive_then_current_sort_truncate",
    ):
        candidate = by_name.get(name)
        if candidate and candidate["strong_revision_candidate"]:
            return name
    return None


def _mean_meta(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return float(np.mean([float(row.get("meta", {}).get(key, 0)) for row in rows]))


def _median_meta(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return float(np.median([float(row.get("meta", {}).get(key, 0)) for row in rows]))


def _min_meta(rows: list[dict[str, Any]], key: str) -> int | None:
    if not rows:
        return None
    return int(min(int(row.get("meta", {}).get(key, 0)) for row in rows))


def _current_policy_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    non_empty = [row for row in raw_rows if _has_text(row)]
    ranked = sorted(
        non_empty,
        key=lambda row: (
            -int(row.get("meta", {}).get("raw_tag_token_count", 0)),
            -_label_sum(row),
            str(row["sample_id"]),
        ),
    )
    return sorted(ranked[:EXPECTED_FILTERED_COUNT], key=lambda row: str(row["sample_id"]))


def _truncate_current_rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: (
            -int(row.get("meta", {}).get("raw_tag_token_count", 0)),
            -_label_sum(row),
            str(row["sample_id"]),
        ),
    )
    return sorted(ranked[:EXPECTED_FILTERED_COUNT], key=lambda row: str(row["sample_id"]))


def _make_split(sample_ids: list[str], seed: int, query_count: int, train_count: int) -> dict[str, list[str]]:
    sorted_ids = sorted(sample_ids)
    indices = np.random.RandomState(seed).permutation(len(sorted_ids))
    permuted = [sorted_ids[int(index)] for index in indices]
    query_ids = permuted[:query_count]
    retrieval_ids = permuted[query_count:]
    train_ids = retrieval_ids[:train_count]
    return {"query_ids": query_ids, "retrieval_ids": retrieval_ids, "train_ids": train_ids}


def _queries_with_no_relevant(query_labels: np.ndarray, retrieval_labels: np.ndarray) -> list[int]:
    no_relevant: list[int] = []
    retrieval_bool = retrieval_labels.astype(bool)
    for start in range(0, query_labels.shape[0], 128):
        end = min(start + 128, query_labels.shape[0])
        relevant = (query_labels[start:end].astype(bool) @ retrieval_bool.T) > 0
        no_relevant.extend((np.where(np.sum(relevant, axis=1) == 0)[0] + start).astype(int).tolist())
    return no_relevant


def _labels(rows: list[dict[str, Any]]) -> np.ndarray:
    if not rows:
        return np.zeros((0, 0), dtype=np.uint8)
    return np.asarray([[int(value) for value in row["label_vector"]] for row in rows], dtype=np.uint8)


def _indices(ids: list[str], id_to_index: dict[str, int], name: str) -> np.ndarray:
    indices = []
    for sample_id in ids:
        if sample_id not in id_to_index:
            raise RuntimeError(f"{name} contains id outside candidate set: {sample_id}")
        indices.append(id_to_index[sample_id])
    return np.asarray(indices, dtype=np.int64)


def _label_sum(row: dict[str, Any]) -> int:
    return int(sum(int(value) for value in row["label_vector"]))


def _has_text(row: dict[str, Any]) -> bool:
    return bool(str(row.get("text_source", "")).strip())


def _markdown(summary: dict[str, Any]) -> str:
    current = summary["current_mir_risk"]
    map_effect = summary["no_relevant_query_map_effect"]
    stage2 = summary["stage2_feature_quality"]
    rec = summary["recommendation"]
    lines = [
        "# MIR Stage 1/2 Causality Audit",
        "",
        "Audit-only report. Stage 1/2/3 formal artifacts were not modified.",
        "",
        "## Current MIR Risk",
        "",
        f"- manifest_filtered_count: {current['manifest_filtered_count']}",
        f"- zero_label_query_count: {current['zero_label_query_count']}",
        f"- query_with_no_relevant_retrieval_count: {current['query_with_no_relevant_retrieval_count']}",
        f"- query_with_no_relevant_retrieval_rate: {current['query_with_no_relevant_retrieval_rate']}",
        "",
        "## Candidate Filtering Audit",
        "",
        "| candidate | count | can reach 20015 | zero labels | seed0 zero-label query | seed0 no-relevant query | strong revision |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in summary["candidate_filtering_audit"]:
        lines.append(
        f"| {candidate['candidate_name']} | {candidate['candidate_count']} | {candidate['can_reach_20015']} | "
            f"{candidate['zero_label_count']} | {candidate['zero_label_query_count_under_seed0_split']} | "
            f"{candidate['query_with_no_relevant_retrieval_count_under_seed0_split']} | "
            f"{candidate['strong_revision_candidate']} |"
        )
    lines.extend(
        [
            "",
            "## No-Relevant Query mAP Effect",
            "",
            f"- I2T all queries: {map_effect['clip_i2t_map_at_50_all_queries']}",
            f"- I2T excluding no-relevant: {map_effect['clip_i2t_map_at_50_excluding_no_relevant_queries']}",
            f"- T2I all queries: {map_effect['clip_t2i_map_at_50_all_queries']}",
            f"- T2I excluding no-relevant: {map_effect['clip_t2i_map_at_50_excluding_no_relevant_queries']}",
            "",
            "## Stage 2 Feature Quality",
            "",
            f"- cosine_gap_mean: {stage2['cosine_gap_mean']}",
            f"- cosine_gap_median: {stage2['cosine_gap_median']}",
            f"- stage2_feature_risk: {stage2['stage2_feature_risk']}",
            "",
            "## Recommendation",
            "",
            f"- stage1_revision_recommended: {rec['stage1_revision_recommended']}",
            f"- stage2_direct_revision_recommended: {rec['stage2_direct_revision_recommended']}",
            f"- recommended_action: {rec['recommended_action']}",
            f"- reason: {rec['reason']}",
            "",
        ]
    )
    return "\n".join(lines)


def _read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle if line.rstrip("\n")]


def _require_inputs(paths: dict[str, Path]) -> None:
    for name, path in paths.items():
        if name == "raw_root":
            if not path.is_dir():
                raise RuntimeError(f"Missing required raw root: {path}")
        elif not path.is_file():
            raise RuntimeError(f"Missing required input {name}: {path}")


def _hash_readonly_inputs(paths: dict[str, Path]) -> dict[str, str]:
    return {name: _sha256_file(path) for name, path in paths.items() if path.is_file()}


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

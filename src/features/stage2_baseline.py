from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.utils.jsonl import write_json


def compute_stage2_baseline(
    dataset: str,
    feature_set_id: str,
    output_dir: Path,
    rows: list[dict[str, Any]],
    query_ids: list[str],
    retrieval_ids: list[str],
    config: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
    baseline_config = config["baseline"]
    top_k = int(baseline_config["top_k"])
    block_size = int(baseline_config["block_size_similarity"])
    query_block_size = int(baseline_config["query_block_size"])
    random_seed = int(baseline_config["random_seed"])
    random_pair_sample_size = int(baseline_config["random_pair_sample_size"])

    x_i = np.load(output_dir / "X_I.npy")
    x_t = np.load(output_dir / "X_T.npy")
    if x_i.dtype != np.float32 or x_t.dtype != np.float32:
        raise RuntimeError("Stage 2 baseline requires float32 features")

    sample_ids = [str(row["sample_id"]) for row in rows]
    id_to_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    query_indices = _indices_for_ids(query_ids, id_to_index, "query_ids")
    retrieval_indices = _indices_for_ids(retrieval_ids, id_to_index, "retrieval_ids")
    labels = _label_matrix(rows)

    paired_cosine = np.sum(x_i * x_t, axis=1, dtype=np.float32)
    random_indices = _random_unpaired_indices(len(rows), random_seed, min(random_pair_sample_size, len(rows)))
    random_cosine = np.sum(x_i[random_indices[:, 0]] * x_t[random_indices[:, 1]], axis=1, dtype=np.float32)

    clip_i2t = _blockwise_map_at_k(
        query_features=x_i[query_indices],
        retrieval_features=x_t[retrieval_indices],
        query_labels=labels[query_indices],
        retrieval_labels=labels[retrieval_indices],
        top_k=top_k,
        query_block_size=query_block_size,
        retrieval_block_size=block_size,
        device=str(config["runtime"]["device"]),
    )
    clip_t2i = _blockwise_map_at_k(
        query_features=x_t[query_indices],
        retrieval_features=x_i[retrieval_indices],
        query_labels=labels[query_indices],
        retrieval_labels=labels[retrieval_indices],
        top_k=top_k,
        query_block_size=query_block_size,
        retrieval_block_size=block_size,
        device=str(config["runtime"]["device"]),
    )

    paired_mean = float(np.mean(paired_cosine))
    paired_median = float(np.median(paired_cosine))
    random_mean = float(np.mean(random_cosine))
    random_median = float(np.median(random_cosine))
    summary = {
        "dataset": dataset,
        "feature_set_id": feature_set_id,
        "filtered_count": len(rows),
        "query_count": len(query_ids),
        "retrieval_count": len(retrieval_ids),
        "paired_cosine_mean": paired_mean,
        "paired_cosine_median": paired_median,
        "random_cosine_mean": random_mean,
        "random_cosine_median": random_median,
        "cosine_gap_mean": paired_mean - random_mean,
        "cosine_gap_median": paired_median - random_median,
        "clip_i2t_map_at_50": float(clip_i2t),
        "clip_t2i_map_at_50": float(clip_t2i),
        "block_size_similarity": block_size,
        "baseline_completed": True,
        "failure_reason": None,
        "backbone_id": meta["backbone_id"],
        "model_local_path": meta["model_local_path"],
        "local_files_only": meta["local_files_only"],
        "device": meta["device"],
        "feature_dim": meta["feature_dim"],
        "feature_dtype": meta["dtype"],
        "manifest_filtered_order_sha256": meta["manifest_filtered_order_sha256"],
        "query_ids_sha256": meta["query_ids_sha256"],
        "retrieval_ids_sha256": meta["retrieval_ids_sha256"],
        "random_seed": random_seed,
        "random_pair_sample_size": int(len(random_indices)),
        "generated_at_utc": _utc_now(),
    }
    write_json(output_dir / "baseline_summary.json", summary)
    return summary


def _indices_for_ids(ids: list[str], id_to_index: dict[str, int], name: str) -> np.ndarray:
    indices: list[int] = []
    for sample_id in ids:
        index = id_to_index.get(sample_id)
        if index is None:
            raise RuntimeError(f"{name} contains id not present in manifest_filtered: {sample_id}")
        indices.append(index)
    return np.asarray(indices, dtype=np.int64)


def _label_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    labels = []
    for index, row in enumerate(rows, start=1):
        vector = row.get("label_vector")
        if not isinstance(vector, list) or not vector:
            raise RuntimeError(f"manifest row {index} has missing label_vector")
        labels.append([int(value) for value in vector])
    return np.asarray(labels, dtype=np.uint8)


def _random_unpaired_indices(count: int, seed: int, sample_size: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    left = rng.choice(count, size=sample_size, replace=sample_size > count)
    right = rng.choice(count, size=sample_size, replace=sample_size > count)
    same = left == right
    while np.any(same):
        right[same] = rng.choice(count, size=int(np.sum(same)), replace=sample_size > count)
        same = left == right
    return np.stack([left, right], axis=1).astype(np.int64)


def _blockwise_map_at_k(
    query_features: np.ndarray,
    retrieval_features: np.ndarray,
    query_labels: np.ndarray,
    retrieval_labels: np.ndarray,
    top_k: int,
    query_block_size: int,
    retrieval_block_size: int,
    device: str,
) -> float:
    ap_values: list[float] = []
    retrieval_count = retrieval_features.shape[0]
    if retrieval_count < top_k:
        raise RuntimeError(f"retrieval_count {retrieval_count} is smaller than top_k {top_k}")
    if device != "cuda:0" or not torch.cuda.is_available():
        raise RuntimeError("Stage 2 baseline requires cuda:0; CPU fallback is forbidden")

    retrieval_tensor = torch.from_numpy(np.ascontiguousarray(retrieval_features, dtype=np.float32)).to(device)

    with torch.no_grad():
        for q_start in range(0, query_features.shape[0], query_block_size):
            q_end = min(q_start + query_block_size, query_features.shape[0])
            q_feat = torch.from_numpy(np.ascontiguousarray(query_features[q_start:q_end], dtype=np.float32)).to(device)
            q_labels = query_labels[q_start:q_end].astype(bool)
            q_size = q_end - q_start
            top_scores = np.full((q_size, top_k), -np.inf, dtype=np.float32)
            top_indices = np.full((q_size, top_k), -1, dtype=np.int64)
            total_relevant = np.zeros(q_size, dtype=np.int64)

            for r_start in range(0, retrieval_count, retrieval_block_size):
                r_end = min(r_start + retrieval_block_size, retrieval_count)
                scores = (q_feat @ retrieval_tensor[r_start:r_end].T).detach().cpu().numpy()
                local_count = min(top_k, scores.shape[1])
                if scores.shape[1] == local_count:
                    local_indices = np.tile(np.arange(scores.shape[1]), (q_size, 1))
                    local_scores = scores
                else:
                    local_indices = np.argpartition(scores, -local_count, axis=1)[:, -local_count:]
                    local_scores = np.take_along_axis(scores, local_indices, axis=1)
                local_indices = local_indices.astype(np.int64) + r_start

                merged_scores = np.concatenate([top_scores, local_scores], axis=1)
                merged_indices = np.concatenate([top_indices, local_indices], axis=1)
                keep = np.argpartition(merged_scores, -top_k, axis=1)[:, -top_k:]
                top_scores = np.take_along_axis(merged_scores, keep, axis=1)
                top_indices = np.take_along_axis(merged_indices, keep, axis=1)

                relevance_block = np.any(q_labels[:, None, :] & retrieval_labels[None, r_start:r_end, :].astype(bool), axis=2)
                total_relevant += relevance_block.sum(axis=1)

            order = np.argsort(-top_scores, axis=1)
            top_indices = np.take_along_axis(top_indices, order, axis=1)
            relevance_top = np.any(
                retrieval_labels[top_indices].astype(bool) & q_labels[:, None, :],
                axis=2,
            )
            ap_values.extend(_average_precision_at_k(relevance_top, total_relevant, top_k))

    return float(np.mean(np.asarray(ap_values, dtype=np.float64)))


def _average_precision_at_k(relevance_top: np.ndarray, total_relevant: np.ndarray, top_k: int) -> list[float]:
    ranks = np.arange(1, top_k + 1, dtype=np.float64)
    result = []
    for row, relevant_count in zip(relevance_top, total_relevant):
        if relevant_count <= 0:
            result.append(0.0)
            continue
        hits = row.astype(np.float64)
        precision = np.cumsum(hits) / ranks
        denominator = min(int(relevant_count), top_k)
        result.append(float(np.sum(precision * hits) / denominator))
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

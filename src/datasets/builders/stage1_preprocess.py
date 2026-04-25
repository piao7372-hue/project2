from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.jsonl import jsonl_dumps, read_json, write_json, write_jsonl

STAGE1A_BUILDER_VERSION = "stage1a_mir_builder_v2"
STAGE1B_BUILDER_VERSION = "stage1b_nus_builder_v1"
STAGE1C_BUILDER_VERSION = "stage1c_coco_builder_v1"
MIR_DATASET = "mirflickr25k"
NUS_DATASET = "nuswide"
COCO_DATASET = "mscoco"
MIR_SAMPLE_RE = re.compile(r"^mir_\d{5}$")
NUS_SAMPLE_RE = re.compile(r"^nus_\d{6}$")
COCO_SAMPLE_RE = re.compile(r"^coco_\d{12}$")
MIR_FIELD_NAMES = ["sample_id", "dataset_name", "image_path", "text_source", "label_vector", "raw_index", "meta"]


def run_stage1_preprocess(repo_root: Path, config_path: Path, raw_roots_path: Path, dataset: str) -> dict[str, Any]:
    if dataset not in (MIR_DATASET, NUS_DATASET, COCO_DATASET):
        raise ValueError(f"Stage 1 supports {MIR_DATASET}, {NUS_DATASET}, and {COCO_DATASET}; got {dataset}")
    repo_root = repo_root.resolve()
    config = read_json(_resolve_repo_path(repo_root, config_path))
    raw_roots = read_json(_resolve_repo_path(repo_root, raw_roots_path))
    dataset_config = config["datasets"][dataset]
    raw_root = _resolve_repo_path(repo_root, Path(raw_roots[dataset]["raw_root"]))
    processed_root = _resolve_repo_path(repo_root, Path(config["outputs"]["processed_root"])) / dataset
    _ensure_within(repo_root, raw_root)
    _ensure_within(repo_root, processed_root)
    if dataset == NUS_DATASET:
        return _run_nuswide_preprocess(repo_root, config, raw_roots, raw_root, processed_root)
    if dataset == COCO_DATASET:
        return _run_coco_preprocess(repo_root, config, raw_roots, raw_root, processed_root)

    generated_at = _utc_now()
    raw_samples = _build_mir_raw_samples(repo_root, raw_root, dataset_config)
    filtered_samples, filter_stats = _filter_mir_samples(raw_samples, dataset_config)
    split = make_split(
        [sample["sample_id"] for sample in filtered_samples],
        seed=int(config["split"]["seed"]),
        query_count=int(config["split"]["query_count"]),
        train_count=int(config["split"]["train_count"]),
    )
    _validate_counts(raw_samples, filtered_samples, split, dataset_config)
    paths = _stage1_paths(processed_root)
    order_hashes = {
        "sample_id_order_sha256": hash_lines(sorted(sample["sample_id"] for sample in filtered_samples)),
        "manifest_filtered_order_sha256": hash_lines(sample["sample_id"] for sample in filtered_samples),
        "query_ids_sha256": hash_lines(split["query_ids"]),
        "retrieval_ids_sha256": hash_lines(split["retrieval_ids"]),
        "train_ids_sha256": hash_lines(split["train_ids"]),
    }
    write_jsonl(paths["manifest_raw"], raw_samples)
    write_jsonl(paths["manifest_filtered"], filtered_samples)
    write_json(paths["manifest_meta"], _manifest_meta(dataset_config, raw_root, raw_samples, filtered_samples, filter_stats, generated_at))
    _write_lines(paths["query_ids"], split["query_ids"])
    _write_lines(paths["retrieval_ids"], split["retrieval_ids"])
    _write_lines(paths["train_ids"], split["train_ids"])
    write_json(paths["split_summary"], _split_summary(split, config["split"], dataset_config, generated_at))
    write_json(paths["preprocess_summary"], _preprocess_summary(dataset_config, raw_root, processed_root, filter_stats, split, order_hashes, generated_at))
    write_json(paths["config_snapshot"], {"stage1_config": config, "raw_roots_config": {MIR_DATASET: raw_roots[MIR_DATASET]}})
    write_json(paths["order_hashes"], order_hashes)
    return {
        "dataset": MIR_DATASET,
        "generated_at_utc": generated_at,
        "raw_count": len(raw_samples),
        "filtered_count": len(filtered_samples),
        "empty_text_removed": filter_stats["empty_text_removed"],
        "zero_label_filtered_count": filter_stats["zero_label_filtered_count"],
        "query_count": len(split["query_ids"]),
        "retrieval_count": len(split["retrieval_ids"]),
        "train_count": len(split["train_ids"]),
        "processed_root": str(processed_root),
        "order_hashes": order_hashes,
    }


def _run_coco_preprocess(repo_root: Path, config: dict[str, Any], raw_roots: dict[str, Any], raw_root: Path, processed_root: Path) -> dict[str, Any]:
    dataset_config = config["datasets"][COCO_DATASET]
    generated_at = _utc_now()
    paths = _stage1_paths(processed_root)
    samples, stats, category_order = build_coco_samples(repo_root, raw_root, dataset_config)
    split = make_split(
        [sample["sample_id"] for sample in samples],
        seed=int(config["split"]["seed"]),
        query_count=int(config["split"]["query_count"]),
        train_count=int(config["split"]["train_count"]),
    )
    _validate_coco_counts(samples, split, dataset_config)
    order_hashes = {
        "sample_id_order_sha256": hash_lines(sorted(sample["sample_id"] for sample in samples)),
        "manifest_filtered_order_sha256": hash_lines(sample["sample_id"] for sample in samples),
        "query_ids_sha256": hash_lines(split["query_ids"]),
        "retrieval_ids_sha256": hash_lines(split["retrieval_ids"]),
        "train_ids_sha256": hash_lines(split["train_ids"]),
    }
    write_jsonl(paths["manifest_raw"], samples)
    write_jsonl(paths["manifest_filtered"], samples)
    write_json(paths["manifest_meta"], _coco_manifest_meta(dataset_config, raw_root, stats, category_order, generated_at))
    _write_lines(paths["query_ids"], split["query_ids"])
    _write_lines(paths["retrieval_ids"], split["retrieval_ids"])
    _write_lines(paths["train_ids"], split["train_ids"])
    write_json(paths["split_summary"], _split_summary(split, config["split"], dataset_config, generated_at, COCO_DATASET))
    write_json(paths["preprocess_summary"], _coco_preprocess_summary(dataset_config, raw_root, processed_root, stats, category_order, split, order_hashes, generated_at))
    write_json(paths["config_snapshot"], {"stage1_config": config, "raw_roots_config": {COCO_DATASET: raw_roots[COCO_DATASET]}})
    write_json(paths["order_hashes"], order_hashes)
    return {
        "dataset": COCO_DATASET,
        "generated_at_utc": generated_at,
        "raw_count": len(samples),
        "filtered_count": len(samples),
        "caption_image_count": stats["caption_image_count"],
        "instance_image_count": stats["instance_image_count"],
        "category_count": stats["category_count"],
        "zero_label_image_count": stats["zero_label_image_count"],
        "query_count": len(split["query_ids"]),
        "retrieval_count": len(split["retrieval_ids"]),
        "train_count": len(split["train_ids"]),
        "processed_root": str(processed_root),
        "order_hashes": order_hashes,
    }

def _run_nuswide_preprocess(repo_root: Path, config: dict[str, Any], raw_roots: dict[str, Any], raw_root: Path, processed_root: Path) -> dict[str, Any]:
    dataset_config = config["datasets"][NUS_DATASET]
    expected_raw = int(dataset_config["expected_raw_count"])
    generated_at = _utc_now()
    paths = _stage1_paths(processed_root)
    tag_vocab = _read_nus_final_tag_list(raw_root, int(dataset_config["expected_final_tag_count"]))
    image_index = _read_nus_image_index(repo_root, raw_root, expected_raw)
    concept_stats = _scan_nus_concepts(raw_root, expected_raw, int(dataset_config["expected_concept_count"]))
    concept_subset = sorted(concept_stats, key=lambda item: (-item["positive_count"], item["name"]))[: int(dataset_config["label_dimension"])]
    top_label_columns = _read_nus_label_columns(raw_root, concept_subset, expected_raw)
    filtered_samples, tag_stats = _write_nus_manifests(repo_root, raw_root, paths, image_index, tag_vocab, top_label_columns, concept_subset, expected_raw)
    if len(filtered_samples) != int(dataset_config["expected_filtered_count"]):
        raise RuntimeError(f"NUS filtered count mismatch: expected {dataset_config['expected_filtered_count']}, got {len(filtered_samples)}")
    split = make_split(
        [sample["sample_id"] for sample in filtered_samples],
        seed=int(config["split"]["seed"]),
        query_count=int(config["split"]["query_count"]),
        train_count=int(config["split"]["train_count"]),
    )
    _validate_nus_counts(expected_raw, filtered_samples, split, dataset_config)
    order_hashes = {
        "sample_id_order_sha256": hash_lines(sorted(sample["sample_id"] for sample in filtered_samples)),
        "manifest_filtered_order_sha256": hash_lines(sample["sample_id"] for sample in filtered_samples),
        "query_ids_sha256": hash_lines(split["query_ids"]),
        "retrieval_ids_sha256": hash_lines(split["retrieval_ids"]),
        "train_ids_sha256": hash_lines(split["train_ids"]),
    }
    concept_summary = _concept_summary(concept_subset)
    write_json(paths["manifest_meta"], _nus_manifest_meta(dataset_config, raw_root, tag_stats, concept_summary, generated_at))
    _write_lines(paths["query_ids"], split["query_ids"])
    _write_lines(paths["retrieval_ids"], split["retrieval_ids"])
    _write_lines(paths["train_ids"], split["train_ids"])
    write_json(paths["split_summary"], _split_summary(split, config["split"], dataset_config, generated_at, NUS_DATASET))
    write_json(paths["preprocess_summary"], _nus_preprocess_summary(dataset_config, raw_root, processed_root, tag_stats, concept_summary, split, order_hashes, generated_at))
    write_json(paths["config_snapshot"], {"stage1_config": config, "raw_roots_config": {NUS_DATASET: raw_roots[NUS_DATASET]}})
    write_json(paths["order_hashes"], order_hashes)
    return {
        "dataset": NUS_DATASET,
        "generated_at_utc": generated_at,
        "raw_count": expected_raw,
        "filtered_count": len(filtered_samples),
        "empty_tag_row_count": tag_stats["empty_tag_row_count"],
        "query_count": len(split["query_ids"]),
        "retrieval_count": len(split["retrieval_ids"]),
        "train_count": len(split["train_ids"]),
        "processed_root": str(processed_root),
        "concept_subset": concept_summary["names"],
        "concept_positive_counts": concept_summary["positive_counts"],
        "final_tag_list_tag_count": tag_stats["final_tag_list_tag_count"],
        "all_tags_row_count": tag_stats["all_tags_row_count"],
        "order_hashes": order_hashes,
    }


def make_split(sample_ids: list[str], seed: int, query_count: int, train_count: int) -> dict[str, list[str]]:
    sorted_ids = sorted(sample_ids)
    indices = np.random.RandomState(seed).permutation(len(sorted_ids))
    permuted = [sorted_ids[int(index)] for index in indices]
    query_ids = permuted[:query_count]
    retrieval_ids = permuted[query_count:]
    train_ids = retrieval_ids[:train_count]
    return {"query_ids": query_ids, "retrieval_ids": retrieval_ids, "train_ids": train_ids}


def hash_lines(lines: Any) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(str(line).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def decode_nus_tag_line(line: str, vocab: list[str], vocab_index: dict[str, int]) -> tuple[list[str], dict[str, int]]:
    parts = line.split()
    raw_tokens = parts[1:] if parts else []
    active_indices = set()
    known_token_count = 0
    for token in raw_tokens:
        index = vocab_index.get(token)
        if index is not None:
            active_indices.add(index)
            known_token_count += 1
    tags = [vocab[index] for index in sorted(active_indices)]
    return tags, {
        "raw_all_tags_token_count": len(raw_tokens),
        "raw_known_vocab_token_count": known_token_count,
        "raw_out_of_vocab_token_count": len(raw_tokens) - known_token_count,
        "raw_duplicate_vocab_token_count": known_token_count - len(active_indices),
    }


def _build_mir_raw_samples(repo_root: Path, raw_root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    images_dir = raw_root / "extracted" / "images"
    tags_dir = raw_root / "extracted" / "meta" / "tags"
    annotations_dir = raw_root / "extracted" / "annotations"
    _require_dir(images_dir)
    _require_dir(tags_dir)
    _require_dir(annotations_dir)
    label_names = list(config["label_names"])
    label_sets = {name: _read_positive_indices(annotations_dir / f"{name}.txt", int(config["expected_raw_count"])) for name in label_names}
    samples = []
    for raw_index in range(1, int(config["expected_raw_count"]) + 1):
        image_path = images_dir / f"im{raw_index}.jpg"
        tags_path = tags_dir / f"tags{raw_index}.txt"
        _require_file(image_path)
        _require_file(tags_path)
        tags = _read_tags(tags_path)
        label_vector = [1 if raw_index in label_sets[name] else 0 for name in label_names]
        sample_id = f"mir_{raw_index:05d}"
        samples.append(
            {
                "sample_id": sample_id,
                "dataset_name": MIR_DATASET,
                "image_path": _repo_relative(repo_root, image_path),
                "text_source": " ".join(tags),
                "label_vector": label_vector,
                "raw_index": raw_index,
                "meta": {
                    "annotation_positive_count": int(sum(label_vector)),
                    "raw_tag_token_count": len(tags),
                },
            }
        )
    return samples


def _filter_mir_samples(samples: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_filtered_count = int(config["expected_filtered_count"])
    filter_policy = str(config["filter_policy"])
    filter_candidate = str(config.get("filter_candidate", "candidate_1_current_pragmatic_high_signal_v1"))
    non_empty = [sample for sample in samples if sample["text_source"]]
    label_positive = [sample for sample in samples if int(sample["meta"]["annotation_positive_count"]) > 0]
    non_empty_label_positive = [sample for sample in non_empty if int(sample["meta"]["annotation_positive_count"]) > 0]
    if filter_policy == "pragmatic_high_signal_v1":
        candidate_rows = non_empty
    elif filter_policy == "mir_pragmatic_high_signal_label_positive_v2":
        if filter_candidate == "candidate_3_tag20_and_label_positive":
            candidate_rows = [sample for sample in samples if int(sample["meta"]["raw_tag_token_count"]) >= 20 and int(sample["meta"]["annotation_positive_count"]) > 0]
        elif filter_candidate == "candidate_5_nonempty_text_label_positive_then_current_sort_truncate":
            candidate_rows = non_empty_label_positive
        else:
            raise RuntimeError(f"unsupported MIR filter_candidate for {filter_policy}: {filter_candidate}")
    else:
        raise RuntimeError(f"unsupported MIR filter_policy: {filter_policy}")
    ranked = _rank_mir_samples(candidate_rows)
    if len(ranked) < expected_filtered_count:
        raise RuntimeError(f"MIR filtered candidates below expected filtered count: {len(ranked)} < {expected_filtered_count}")
    selected = ranked[:expected_filtered_count]
    zero_label_filtered_count = sum(int(sample["meta"]["annotation_positive_count"]) == 0 for sample in selected)
    if filter_policy == "mir_pragmatic_high_signal_label_positive_v2" and zero_label_filtered_count:
        raise RuntimeError(f"MIR v2 filter selected zero-label samples: {zero_label_filtered_count}")
    return sorted(selected, key=lambda sample: sample["sample_id"]), {
        "filter_policy": filter_policy,
        "filter_candidate": filter_candidate,
        "raw_count": len(samples),
        "non_empty_text_count": len(non_empty),
        "label_positive_count": len(label_positive),
        "non_empty_text_label_positive_count": len(non_empty_label_positive),
        "empty_text_removed": len(samples) - len(non_empty),
        "zero_label_filtered_count": zero_label_filtered_count,
        "filtered_count": expected_filtered_count,
        "selection_order_sample_id_sha256": hash_lines(sample["sample_id"] for sample in selected),
    }


def _rank_mir_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        samples,
        key=lambda sample: (
            -int(sample["meta"]["raw_tag_token_count"]),
            -int(sample["meta"]["annotation_positive_count"]),
            sample["sample_id"],
        ),
    )


def _read_positive_indices(path: Path, expected_raw_count: int) -> set[int]:
    _require_file(path)
    values: set[int] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw_index = int(stripped)
            except ValueError as exc:
                raise RuntimeError(f"non-integer annotation value in {path} line {line_number}: {stripped}") from exc
            if raw_index < 1 or raw_index > expected_raw_count:
                raise RuntimeError(f"annotation value out of MIR range in {path} line {line_number}: {raw_index}")
            if raw_index in values:
                raise RuntimeError(f"duplicate annotation value in {path} line {line_number}: {raw_index}")
            values.add(raw_index)
    return values


def _read_tags(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [token for token in (line.strip() for line in handle) if token]


def _validate_counts(raw_samples: list[dict[str, Any]], filtered_samples: list[dict[str, Any]], split: dict[str, list[str]], config: dict[str, Any]) -> None:
    expected_raw = int(config["expected_raw_count"])
    expected_filtered = int(config["expected_filtered_count"])
    expected_retrieval = int(config["expected_retrieval_count"])
    if len(raw_samples) != expected_raw:
        raise RuntimeError(f"manifest_raw count mismatch: expected {expected_raw}, got {len(raw_samples)}")
    if len(filtered_samples) != expected_filtered:
        raise RuntimeError(f"manifest_filtered count mismatch: expected {expected_filtered}, got {len(filtered_samples)}")
    if len(split["retrieval_ids"]) != expected_retrieval:
        raise RuntimeError(f"retrieval count mismatch: expected {expected_retrieval}, got {len(split['retrieval_ids'])}")
    if not set(split["train_ids"]).issubset(set(split["retrieval_ids"])):
        raise RuntimeError("train split is not a subset of retrieval split")


def _validate_nus_counts(expected_raw: int, filtered_samples: list[dict[str, Any]], split: dict[str, list[str]], config: dict[str, Any]) -> None:
    expected_filtered = int(config["expected_filtered_count"])
    expected_retrieval = int(config["expected_retrieval_count"])
    if expected_raw != int(config["expected_raw_count"]):
        raise RuntimeError(f"NUS raw count mismatch: expected {config['expected_raw_count']}, got {expected_raw}")
    if len(filtered_samples) != expected_filtered:
        raise RuntimeError(f"NUS manifest_filtered count mismatch: expected {expected_filtered}, got {len(filtered_samples)}")
    if len(split["retrieval_ids"]) != expected_retrieval:
        raise RuntimeError(f"NUS retrieval count mismatch: expected {expected_retrieval}, got {len(split['retrieval_ids'])}")
    if not set(split["train_ids"]).issubset(set(split["retrieval_ids"])):
        raise RuntimeError("NUS train split is not a subset of retrieval split")


def _validate_coco_counts(samples: list[dict[str, Any]], split: dict[str, list[str]], config: dict[str, Any]) -> None:
    expected_raw = int(config["expected_raw_count"])
    expected_filtered = int(config["expected_filtered_count"])
    expected_retrieval = int(config["expected_retrieval_count"])
    if len(samples) != expected_raw:
        raise RuntimeError(f"COCO manifest_raw count mismatch: expected {expected_raw}, got {len(samples)}")
    if len(samples) != expected_filtered:
        raise RuntimeError(f"COCO manifest_filtered count mismatch: expected {expected_filtered}, got {len(samples)}")
    if len(split["retrieval_ids"]) != expected_retrieval:
        raise RuntimeError(f"COCO retrieval count mismatch: expected {expected_retrieval}, got {len(split['retrieval_ids'])}")
    if not set(split["train_ids"]).issubset(set(split["retrieval_ids"])):
        raise RuntimeError("COCO train split is not a subset of retrieval split")


def _manifest_meta(config: dict[str, Any], raw_root: Path, raw_samples: list[dict[str, Any]], filtered_samples: list[dict[str, Any]], stats: dict[str, Any], generated_at: str) -> dict[str, Any]:
    return {
        "stage": "stage1",
        "substage": "stage1a_mirflickr25k",
        "builder_version": STAGE1A_BUILDER_VERSION,
        "generated_at_utc": generated_at,
        "dataset_name": MIR_DATASET,
        "raw_root": str(raw_root),
        "sample_fields": MIR_FIELD_NAMES,
        "label_names": config["label_names"],
        "label_dimension": config["label_dimension"],
        "raw_count": len(raw_samples),
        "filtered_count": len(filtered_samples),
        "filter_stats": stats,
        "manifest_filtered_order": "sample_id_ascending",
        "no_silent_fallback": True,
    }


def _split_summary(split: dict[str, list[str]], split_config: dict[str, Any], dataset_config: dict[str, Any], generated_at: str, dataset_name: str = MIR_DATASET) -> dict[str, Any]:
    substage_by_dataset = {
        MIR_DATASET: "stage1a_mirflickr25k",
        NUS_DATASET: "stage1b_nuswide",
        COCO_DATASET: "stage1c_mscoco",
    }
    return {
        "stage": "stage1",
        "substage": substage_by_dataset[dataset_name],
        "generated_at_utc": generated_at,
        "seed": split_config["seed"],
        "rng_protocol": split_config["rng_protocol"],
        "source_order": "sample_id_ascending",
        "query_count": len(split["query_ids"]),
        "retrieval_count": len(split["retrieval_ids"]),
        "train_count": len(split["train_ids"]),
        "expected_query_count": split_config["query_count"],
        "expected_retrieval_count": dataset_config["expected_retrieval_count"],
        "expected_train_count": split_config["train_count"],
        "train_is_retrieval_subset": set(split["train_ids"]).issubset(set(split["retrieval_ids"])),
        "query_retrieval_disjoint": set(split["query_ids"]).isdisjoint(set(split["retrieval_ids"])),
    }


def build_coco_samples(repo_root: Path, raw_root: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    annotations_dir = raw_root / "extracted" / "annotations"
    _require_dir(annotations_dir)
    split_names = ("train2014", "val2014")
    expected_by_split = {
        "train2014": int(config["expected_train_image_count"]),
        "val2014": int(config["expected_val_image_count"]),
    }
    category_order: list[dict[str, Any]] | None = None
    samples: list[dict[str, Any]] = []
    total_caption_images = 0
    total_instance_images = 0
    zero_label_count = 0
    for split_name in split_names:
        captions = _load_json(annotations_dir / f"captions_{split_name}.json")
        instances = _load_json(annotations_dir / f"instances_{split_name}.json")
        split_category_order = _coco_category_order(instances, int(config["expected_category_count"]))
        if category_order is None:
            category_order = split_category_order
        elif category_order != split_category_order:
            raise RuntimeError("COCO category order differs between train and val instances JSON")
        category_position = {item["id"]: index for index, item in enumerate(category_order)}
        caption_images = _coco_image_map(captions, split_name)
        instance_images = _coco_image_map(instances, split_name)
        if len(caption_images) != expected_by_split[split_name]:
            raise RuntimeError(f"COCO {split_name} caption image count mismatch: expected {expected_by_split[split_name]}, got {len(caption_images)}")
        if len(instance_images) != expected_by_split[split_name]:
            raise RuntimeError(f"COCO {split_name} instance image count mismatch: expected {expected_by_split[split_name]}, got {len(instance_images)}")
        if set(caption_images) != set(instance_images):
            raise RuntimeError(f"COCO {split_name} captions/instances image_id sets differ")
        captions_by_image = _coco_captions_by_image(captions)
        instance_by_image = _coco_instances_by_image(instances, category_position)
        total_caption_images += len(caption_images)
        total_instance_images += len(instance_images)
        image_dir = raw_root / "extracted" / split_name
        _require_dir(image_dir)
        for image_id in sorted(caption_images):
            image_info = caption_images[image_id]
            file_name = image_info["file_name"]
            image_path = image_dir / file_name
            _require_file(image_path)
            captions_for_image = captions_by_image.get(image_id, [])
            if not captions_for_image:
                raise RuntimeError(f"COCO image_id {image_id} has no captions")
            caption_ids = [item["id"] for item in captions_for_image]
            text_source = ". ".join(item["caption"] for item in captions_for_image)
            instance_info = instance_by_image.get(image_id, {"annotation_count": 0, "category_positions": set()})
            label_vector = [0] * len(category_order)
            for position in instance_info["category_positions"]:
                label_vector[position] = 1
            category_positive_count = int(sum(label_vector))
            zero_label_count += int(category_positive_count == 0)
            samples.append(
                {
                    "sample_id": f"coco_{image_id:012d}",
                    "dataset_name": COCO_DATASET,
                    "image_path": _repo_relative(repo_root, image_path),
                    "text_source": text_source,
                    "label_vector": label_vector,
                    "raw_index": int(image_id),
                    "meta": {
                        "caption_count": len(captions_for_image),
                        "caption_annotation_ids": caption_ids,
                        "coco_split": split_name,
                        "instance_annotation_count": int(instance_info["annotation_count"]),
                        "category_positive_count": category_positive_count,
                    },
                }
            )
    if category_order is None:
        raise RuntimeError("COCO category order was not loaded")
    samples = sorted(samples, key=lambda sample: sample["sample_id"])
    sample_ids = [sample["sample_id"] for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("COCO duplicate sample_id detected")
    stats = {
        "raw_count": len(samples),
        "filtered_count": len(samples),
        "caption_image_count": total_caption_images,
        "instance_image_count": total_instance_images,
        "category_count": len(category_order),
        "zero_label_image_count": zero_label_count,
    }
    return samples, stats, category_order


def _load_json(path: Path) -> dict[str, Any]:
    _require_file(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _coco_category_order(instances: dict[str, Any], expected_count: int) -> list[dict[str, Any]]:
    categories = sorted(instances.get("categories", []), key=lambda item: int(item["id"]))
    if len(categories) != expected_count:
        raise RuntimeError(f"COCO category count mismatch: expected {expected_count}, got {len(categories)}")
    ids = [int(item["id"]) for item in categories]
    if len(ids) != len(set(ids)):
        raise RuntimeError("COCO duplicate category id detected")
    return [{"id": int(item["id"]), "name": item["name"], "supercategory": item["supercategory"]} for item in categories]


def _coco_image_map(payload: dict[str, Any], split_name: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for image in payload.get("images", []):
        image_id = int(image["id"])
        if image_id in result:
            raise RuntimeError(f"COCO duplicate image_id in {split_name}: {image_id}")
        result[image_id] = image
    return result


def _coco_captions_by_image(captions: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {}
    for annotation in captions.get("annotations", []):
        image_id = int(annotation["image_id"])
        caption = str(annotation["caption"]).strip()
        if not caption:
            raise RuntimeError(f"COCO empty caption annotation id={annotation['id']}")
        result.setdefault(image_id, []).append({"id": int(annotation["id"]), "caption": caption})
    for annotations in result.values():
        annotations.sort(key=lambda item: item["id"])
    return result


def _coco_instances_by_image(instances: dict[str, Any], category_position: dict[int, int]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for annotation in instances.get("annotations", []):
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        if category_id not in category_position:
            raise RuntimeError(f"COCO annotation uses unknown category_id={category_id}")
        info = result.setdefault(image_id, {"annotation_count": 0, "category_positions": set()})
        info["annotation_count"] += 1
        info["category_positions"].add(category_position[category_id])
    return result


def _coco_manifest_meta(config: dict[str, Any], raw_root: Path, stats: dict[str, Any], category_order: list[dict[str, Any]], generated_at: str) -> dict[str, Any]:
    return {
        "stage": "stage1",
        "substage": "stage1c_mscoco",
        "builder_version": STAGE1C_BUILDER_VERSION,
        "generated_at_utc": generated_at,
        "dataset_name": COCO_DATASET,
        "raw_root": str(raw_root),
        "sample_fields": MIR_FIELD_NAMES,
        "label_dimension": config["label_dimension"],
        "raw_count": stats["raw_count"],
        "filtered_count": stats["filtered_count"],
        "caption_image_count": stats["caption_image_count"],
        "instance_image_count": stats["instance_image_count"],
        "category_count": stats["category_count"],
        "category_order_protocol": config["category_order_protocol"],
        "category_order": category_order,
        "zero_label_image_count": stats["zero_label_image_count"],
        "no_silent_fallback": True,
    }


def _coco_preprocess_summary(config: dict[str, Any], raw_root: Path, processed_root: Path, stats: dict[str, Any], category_order: list[dict[str, Any]], split: dict[str, list[str]], hashes: dict[str, str], generated_at: str) -> dict[str, Any]:
    return {
        "stage": "stage1",
        "substage": "stage1c_mscoco",
        "builder_version": STAGE1C_BUILDER_VERSION,
        "generated_at_utc": generated_at,
        "dataset_name": COCO_DATASET,
        "raw_root": str(raw_root),
        "processed_root": str(processed_root),
        "filter_policy": config["filter_policy"],
        "manifest_raw_count": stats["raw_count"],
        "manifest_filtered_count": stats["filtered_count"],
        "caption_image_count": stats["caption_image_count"],
        "instance_image_count": stats["instance_image_count"],
        "category_count": stats["category_count"],
        "category_order_protocol": config["category_order_protocol"],
        "category_order": category_order,
        "zero_label_image_count": stats["zero_label_image_count"],
        "query_count": len(split["query_ids"]),
        "retrieval_count": len(split["retrieval_ids"]),
        "train_count": len(split["train_ids"]),
        "order_hashes": hashes,
        "silent_fallback_used": False,
    }


def _read_nus_final_tag_list(raw_root: Path, expected_count: int) -> list[str]:
    path = raw_root / "extracted" / "tags" / "Final_Tag_List.txt"
    _require_file(path)
    tags = [line.strip() for line in path.open("r", encoding="utf-8-sig") if line.strip()]
    if len(tags) != expected_count:
        raise RuntimeError(f"NUS Final_Tag_List count mismatch: expected {expected_count}, got {len(tags)}")
    if len(tags) != len(set(tags)):
        raise RuntimeError("NUS Final_Tag_List contains duplicate tags")
    return tags


def _read_nus_image_index(repo_root: Path, raw_root: Path, expected_count: int) -> list[dict[str, Any]]:
    path = raw_root / "image_index.tsv"
    _require_file(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                raise RuntimeError(f"NUS image_index.tsv line {line_number} must have raw_index<TAB>image_relative_path")
            try:
                raw_index = int(parts[0])
            except ValueError as exc:
                raise RuntimeError(f"NUS image_index.tsv line {line_number} has non-integer raw_index: {parts[0]}") from exc
            expected_raw_index = line_number - 1
            if raw_index != expected_raw_index:
                raise RuntimeError(f"NUS raw_index order mismatch at line {line_number}: expected {expected_raw_index}, got {raw_index}")
            relative_path = parts[1].strip().replace("\\", "/")
            if not relative_path:
                raise RuntimeError(f"NUS image_index.tsv line {line_number} has empty image path")
            image_path = raw_root / "images" / relative_path
            _require_file(image_path)
            records.append(
                {
                    "raw_index": raw_index,
                    "image_relative_path": relative_path,
                    "repo_image_path": _repo_relative(repo_root, image_path),
                }
            )
    if len(records) != expected_count:
        raise RuntimeError(f"NUS image_index.tsv row count mismatch: expected {expected_count}, got {len(records)}")
    return records


def _scan_nus_concepts(raw_root: Path, expected_count: int, expected_concepts: int) -> list[dict[str, Any]]:
    labels_dir = raw_root / "extracted" / "Groundtruth" / "AllLabels"
    _require_dir(labels_dir)
    files = sorted(labels_dir.glob("Labels_*.txt"))
    if len(files) != expected_concepts:
        raise RuntimeError(f"NUS concept label file count mismatch: expected {expected_concepts}, got {len(files)}")
    stats = []
    for path in files:
        concept = path.stem.removeprefix("Labels_")
        positive_count = 0
        line_count = 0
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_count, line in enumerate(handle, start=1):
                value = line.strip()
                if value not in ("0", "1"):
                    raise RuntimeError(f"NUS label file {path} line {line_count} is not binary: {value!r}")
                positive_count += value == "1"
        if line_count != expected_count:
            raise RuntimeError(f"NUS label file {path} row count mismatch: expected {expected_count}, got {line_count}")
        stats.append({"name": concept, "positive_count": int(positive_count), "path": str(path)})
    return stats


def _read_nus_label_columns(raw_root: Path, concept_subset: list[dict[str, Any]], expected_count: int) -> dict[str, bytearray]:
    labels_dir = raw_root / "extracted" / "Groundtruth" / "AllLabels"
    columns: dict[str, bytearray] = {}
    for item in concept_subset:
        name = item["name"]
        path = labels_dir / f"Labels_{name}.txt"
        values = bytearray()
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = line.strip()
                if value not in ("0", "1"):
                    raise RuntimeError(f"NUS label file {path} line {line_number} is not binary: {value!r}")
                values.append(1 if value == "1" else 0)
        if len(values) != expected_count:
            raise RuntimeError(f"NUS label file {path} row count mismatch: expected {expected_count}, got {len(values)}")
        columns[name] = values
    return columns


def _write_nus_manifests(
    repo_root: Path,
    raw_root: Path,
    paths: dict[str, Path],
    image_index: list[dict[str, Any]],
    tag_vocab: list[str],
    top_label_columns: dict[str, bytearray],
    concept_subset: list[dict[str, Any]],
    expected_raw: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_tags_path = raw_root / "extracted" / "tags" / "All_Tags.txt"
    _require_file(all_tags_path)
    paths["manifest_raw"].parent.mkdir(parents=True, exist_ok=True)
    paths["manifest_filtered"].parent.mkdir(parents=True, exist_ok=True)
    vocab_index = {tag: index for index, tag in enumerate(tag_vocab)}
    concept_names = [item["name"] for item in concept_subset]
    filtered_samples: list[dict[str, Any]] = []
    empty_tag_rows = 0
    raw_known_tokens = 0
    raw_out_of_vocab_tokens = 0
    all_tags_rows = 0
    with all_tags_path.open("r", encoding="utf-8-sig") as all_tags_handle, paths["manifest_raw"].open("w", encoding="utf-8", newline="\n") as raw_handle:
        for row_number, (line, image_record) in enumerate(zip(all_tags_handle, image_index), start=1):
            raw_index = int(image_record["raw_index"])
            if raw_index != row_number - 1:
                raise RuntimeError(f"NUS raw_index and All_Tags row mismatch at row {row_number}")
            tags, tag_counts = decode_nus_tag_line(line, tag_vocab, vocab_index)
            label_vector = [int(top_label_columns[name][row_number - 1]) for name in concept_names]
            empty_tag_row = len(tags) == 0
            empty_tag_rows += int(empty_tag_row)
            raw_known_tokens += tag_counts["raw_known_vocab_token_count"]
            raw_out_of_vocab_tokens += tag_counts["raw_out_of_vocab_token_count"]
            all_tags_rows += 1
            sample = {
                "sample_id": f"nus_{raw_index:06d}",
                "dataset_name": NUS_DATASET,
                "image_path": image_record["repo_image_path"],
                "text_source": " ".join(tags),
                "label_vector": label_vector,
                "raw_index": raw_index,
                "meta": {
                    "annotation_positive_count": int(sum(label_vector)),
                    "raw_empty_tag_row": empty_tag_row,
                    "raw_tag_token_count": len(tags),
                    "raw_all_tags_token_count": tag_counts["raw_all_tags_token_count"],
                    "raw_out_of_vocab_token_count": tag_counts["raw_out_of_vocab_token_count"],
                    "text_source_protocol": "final_tag_list_projected_binary_decode_v1",
                },
            }
            raw_handle.write(jsonl_dumps(sample))
            raw_handle.write("\n")
            if sum(label_vector) > 0:
                filtered_samples.append(sample)
        remaining = next(all_tags_handle, None)
        if remaining is not None:
            raise RuntimeError("NUS All_Tags.txt has more rows than image_index.tsv")
    if all_tags_rows != expected_raw:
        raise RuntimeError(f"NUS All_Tags.txt row count mismatch: expected {expected_raw}, got {all_tags_rows}")
    filtered_samples = sorted(filtered_samples, key=lambda sample: sample["sample_id"])
    write_jsonl(paths["manifest_filtered"], filtered_samples)
    return filtered_samples, {
        "raw_count": expected_raw,
        "filtered_count": len(filtered_samples),
        "all_tags_row_count": all_tags_rows,
        "final_tag_list_tag_count": len(tag_vocab),
        "empty_tag_row_count": empty_tag_rows,
        "raw_known_vocab_token_count": raw_known_tokens,
        "raw_out_of_vocab_token_count": raw_out_of_vocab_tokens,
        "text_source_protocol": "final_tag_list_projected_binary_decode_v1",
    }


def _concept_summary(concept_subset: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "names": [item["name"] for item in concept_subset],
        "positive_counts": {item["name"]: int(item["positive_count"]) for item in concept_subset},
        "items": [{"name": item["name"], "positive_count": int(item["positive_count"])} for item in concept_subset],
    }


def _nus_manifest_meta(config: dict[str, Any], raw_root: Path, tag_stats: dict[str, Any], concept_summary: dict[str, Any], generated_at: str) -> dict[str, Any]:
    return {
        "stage": "stage1",
        "substage": "stage1b_nuswide",
        "builder_version": STAGE1B_BUILDER_VERSION,
        "generated_at_utc": generated_at,
        "dataset_name": NUS_DATASET,
        "raw_root": str(raw_root),
        "sample_fields": MIR_FIELD_NAMES,
        "label_dimension": config["label_dimension"],
        "raw_count": tag_stats["raw_count"],
        "filtered_count": tag_stats["filtered_count"],
        "concept_subset": concept_summary["items"],
        "concept_positive_counts": concept_summary["positive_counts"],
        "final_tag_list_tag_count": tag_stats["final_tag_list_tag_count"],
        "all_tags_row_count": tag_stats["all_tags_row_count"],
        "empty_tag_row_count": tag_stats["empty_tag_row_count"],
        "text_source_protocol": config["text_source_protocol"],
        "deprecated_kaggle_top10_used": False,
        "no_silent_fallback": True,
    }


def _nus_preprocess_summary(config: dict[str, Any], raw_root: Path, processed_root: Path, tag_stats: dict[str, Any], concept_summary: dict[str, Any], split: dict[str, list[str]], hashes: dict[str, str], generated_at: str) -> dict[str, Any]:
    return {
        "stage": "stage1",
        "substage": "stage1b_nuswide",
        "builder_version": STAGE1B_BUILDER_VERSION,
        "generated_at_utc": generated_at,
        "dataset_name": NUS_DATASET,
        "raw_root": str(raw_root),
        "processed_root": str(processed_root),
        "filter_policy": config["filter_policy"],
        "manifest_raw_count": tag_stats["raw_count"],
        "manifest_filtered_count": tag_stats["filtered_count"],
        "concept_subset": concept_summary["items"],
        "concept_positive_counts": concept_summary["positive_counts"],
        "final_tag_list_tag_count": tag_stats["final_tag_list_tag_count"],
        "all_tags_row_count": tag_stats["all_tags_row_count"],
        "empty_tag_row_count": tag_stats["empty_tag_row_count"],
        "raw_out_of_vocab_token_count": tag_stats["raw_out_of_vocab_token_count"],
        "text_source_protocol": config["text_source_protocol"],
        "query_count": len(split["query_ids"]),
        "retrieval_count": len(split["retrieval_ids"]),
        "train_count": len(split["train_ids"]),
        "order_hashes": hashes,
        "deprecated_kaggle_top10_used": False,
        "silent_fallback_used": False,
    }


def _preprocess_summary(config: dict[str, Any], raw_root: Path, processed_root: Path, stats: dict[str, Any], split: dict[str, list[str]], hashes: dict[str, str], generated_at: str) -> dict[str, Any]:
    return {
        "stage": "stage1",
        "substage": "stage1a_mirflickr25k",
        "builder_version": STAGE1A_BUILDER_VERSION,
        "generated_at_utc": generated_at,
        "dataset_name": MIR_DATASET,
        "raw_root": str(raw_root),
        "processed_root": str(processed_root),
        "filter_policy": config["filter_policy"],
        "filter_candidate": stats.get("filter_candidate"),
        "manifest_raw_count": stats["raw_count"],
        "manifest_filtered_count": stats["filtered_count"],
        "empty_text_removed": stats["empty_text_removed"],
        "non_empty_text_count": stats["non_empty_text_count"],
        "label_positive_count": stats["label_positive_count"],
        "non_empty_text_label_positive_count": stats["non_empty_text_label_positive_count"],
        "zero_label_filtered_count": stats["zero_label_filtered_count"],
        "query_count": len(split["query_ids"]),
        "retrieval_count": len(split["retrieval_ids"]),
        "train_count": len(split["train_ids"]),
        "order_hashes": hashes,
        "silent_fallback_used": False,
    }


def _stage1_paths(processed_root: Path) -> dict[str, Path]:
    return {
        "manifest_raw": processed_root / "manifest" / "manifest_raw.jsonl",
        "manifest_filtered": processed_root / "manifest" / "manifest_filtered.jsonl",
        "manifest_meta": processed_root / "manifest" / "manifest_meta.json",
        "query_ids": processed_root / "splits" / "query_ids.txt",
        "retrieval_ids": processed_root / "splits" / "retrieval_ids.txt",
        "train_ids": processed_root / "splits" / "train_ids.txt",
        "split_summary": processed_root / "splits" / "split_summary.json",
        "preprocess_summary": processed_root / "reports" / "preprocess_summary.json",
        "validator_summary": processed_root / "reports" / "validator_summary.json",
        "config_snapshot": processed_root / "reports" / "config_snapshot.json",
        "order_hashes": processed_root / "reports" / "order_hashes.json",
    }


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def _resolve_repo_path(repo_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _ensure_within(repo_root: Path, path: Path) -> None:
    path.resolve().relative_to(repo_root)


def _require_dir(path: Path) -> None:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"required directory missing: {path}")


def _require_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"required file missing: {path}")


def _repo_relative(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root).as_posix()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

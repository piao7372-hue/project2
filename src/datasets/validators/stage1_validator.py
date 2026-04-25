from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

from src.datasets.builders.stage1_preprocess import (
    _read_nus_final_tag_list,
    _read_nus_image_index,
    _read_nus_label_columns,
    _scan_nus_concepts,
    build_coco_samples,
    decode_nus_tag_line,
    hash_lines,
    make_split,
)
from src.utils.jsonl import iter_jsonl, read_json, write_json

STAGE1_VALIDATOR_VERSION = "stage1_mir_nus_coco_validator_v3"
MIR_DATASET = "mirflickr25k"
NUS_DATASET = "nuswide"
COCO_DATASET = "mscoco"
MIR_SAMPLE_RE = re.compile(r"^mir_\d{5}$")
NUS_SAMPLE_RE = re.compile(r"^nus_\d{6}$")
COCO_SAMPLE_RE = re.compile(r"^coco_\d{12}$")
REQUIRED_FIELDS = {"sample_id", "dataset_name", "image_path", "text_source", "label_vector", "raw_index", "meta"}


def validate_stage1_preprocess(repo_root: Path, config_path: Path, dataset: str) -> dict[str, Any]:
    if dataset == MIR_DATASET:
        return _validate_mir(repo_root, config_path)
    if dataset == NUS_DATASET:
        return _validate_nus(repo_root, config_path)
    if dataset == COCO_DATASET:
        return _validate_coco(repo_root, config_path)
    raise ValueError(f"Stage 1 validator supports {MIR_DATASET}, {NUS_DATASET}, and {COCO_DATASET}; got {dataset}")


def _validate_mir(repo_root: Path, config_path: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = read_json(_resolve_repo_path(repo_root, config_path))
    dataset_config = config["datasets"][MIR_DATASET]
    processed_root = _resolve_repo_path(repo_root, Path(config["outputs"]["processed_root"])) / MIR_DATASET
    paths = _stage1_paths(processed_root)
    failures: list[str] = []
    output_presence = _check_output_presence(paths, failures)
    raw_rows: list[dict[str, Any]] = []
    filtered_rows: list[dict[str, Any]] = []
    if not failures:
        raw_rows = list(iter_jsonl(paths["manifest_raw"]))
        filtered_rows = list(iter_jsonl(paths["manifest_filtered"]))
        _check_counts(raw_rows, filtered_rows, dataset_config, failures)
        _check_manifest_rows(raw_rows, dataset_config, MIR_DATASET, MIR_SAMPLE_RE, "manifest_raw", failures)
        _check_manifest_rows(filtered_rows, dataset_config, MIR_DATASET, MIR_SAMPLE_RE, "manifest_filtered", failures)
        _check_filtered_text(filtered_rows, failures)
        _check_sample_ids(raw_rows, "manifest_raw", failures)
        _check_sample_ids(filtered_rows, "manifest_filtered", failures)
        _check_splits(filtered_rows, paths, config["split"], dataset_config, failures)
        _check_hashes(filtered_rows, paths, failures)
        _check_no_silent_fallback(paths, failures)
        mir_label_contract = _check_mir_label_positive_contract(filtered_rows, paths, config, failures)
    else:
        mir_label_contract = {}
    summary = _base_summary(MIR_DATASET, processed_root, output_presence, failures, raw_rows, filtered_rows, paths)
    summary.update(mir_label_contract)
    write_json(paths["validator_summary"], summary)
    return summary


def _validate_nus(repo_root: Path, config_path: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = read_json(_resolve_repo_path(repo_root, config_path))
    raw_roots = read_json(_resolve_repo_path(repo_root, Path(config["inputs"]["raw_roots_config"])))
    dataset_config = config["datasets"][NUS_DATASET]
    raw_root = _resolve_repo_path(repo_root, Path(raw_roots[NUS_DATASET]["raw_root"]))
    processed_root = _resolve_repo_path(repo_root, Path(config["outputs"]["processed_root"])) / NUS_DATASET
    paths = _stage1_paths(processed_root)
    failures: list[str] = []
    output_presence = _check_output_presence(paths, failures)
    raw_count = filtered_count = empty_tag_rows = 0
    concept_items: list[dict[str, Any]] = []
    final_tag_count = all_tags_row_count = 0
    filtered_rows: list[dict[str, Any]] = []
    if not failures:
        tag_vocab = _read_nus_final_tag_list(raw_root, int(dataset_config["expected_final_tag_count"]))
        final_tag_count = len(tag_vocab)
        image_index = _read_nus_image_index(repo_root, raw_root, int(dataset_config["expected_raw_count"]))
        all_tags_path = raw_root / "extracted" / "tags" / "All_Tags.txt"
        all_tags_row_count = _count_lines(all_tags_path)
        if all_tags_row_count != int(dataset_config["expected_raw_count"]):
            failures.append(f"All_Tags.txt row count mismatch: expected {dataset_config['expected_raw_count']}, got {all_tags_row_count}")
        concept_stats = _scan_nus_concepts(raw_root, int(dataset_config["expected_raw_count"]), int(dataset_config["expected_concept_count"]))
        concept_subset = sorted(concept_stats, key=lambda item: (-item["positive_count"], item["name"]))[: int(dataset_config["label_dimension"])]
        top_label_columns = _read_nus_label_columns(raw_root, concept_subset, int(dataset_config["expected_raw_count"]))
        concept_items = [{"name": item["name"], "positive_count": int(item["positive_count"])} for item in concept_subset]
        _check_nus_concept_outputs(paths, concept_items, failures)
        raw_count, empty_tag_rows, expected_filtered = _check_nus_raw_manifest(
            repo_root, raw_root, paths, image_index, tag_vocab, top_label_columns, concept_items, failures
        )
        filtered_rows = list(iter_jsonl(paths["manifest_filtered"]))
        filtered_count = len(filtered_rows)
        _check_nus_filtered_manifest(repo_root, filtered_rows, expected_filtered, dataset_config, failures)
        _check_splits(filtered_rows, paths, config["split"], dataset_config, failures)
        _check_hashes(filtered_rows, paths, failures)
        _check_no_silent_fallback(paths, failures)
        _check_no_kaggle_top10(paths, failures)
    summary = _base_summary(NUS_DATASET, processed_root, output_presence, failures, [], filtered_rows, paths)
    summary.update(
        {
            "manifest_raw_count": raw_count,
            "manifest_filtered_count": filtered_count,
            "concept_subset": concept_items,
            "concept_positive_counts": {item["name"]: item["positive_count"] for item in concept_items},
            "final_tag_list_tag_count": final_tag_count,
            "all_tags_row_count": all_tags_row_count,
            "empty_tag_row_count": empty_tag_rows,
        }
    )
    write_json(paths["validator_summary"], summary)
    return summary


def _validate_coco(repo_root: Path, config_path: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = read_json(_resolve_repo_path(repo_root, config_path))
    raw_roots = read_json(_resolve_repo_path(repo_root, Path(config["inputs"]["raw_roots_config"])))
    dataset_config = config["datasets"][COCO_DATASET]
    raw_root = _resolve_repo_path(repo_root, Path(raw_roots[COCO_DATASET]["raw_root"]))
    processed_root = _resolve_repo_path(repo_root, Path(config["outputs"]["processed_root"])) / COCO_DATASET
    paths = _stage1_paths(processed_root)
    failures: list[str] = []
    output_presence = _check_output_presence(paths, failures)
    raw_rows: list[dict[str, Any]] = []
    filtered_rows: list[dict[str, Any]] = []
    stats = {"caption_image_count": 0, "instance_image_count": 0, "category_count": 0, "zero_label_image_count": 0}
    category_order: list[dict[str, Any]] = []
    if not failures:
        expected_samples, stats, category_order = build_coco_samples(repo_root, raw_root, dataset_config)
        raw_rows = list(iter_jsonl(paths["manifest_raw"]))
        filtered_rows = list(iter_jsonl(paths["manifest_filtered"]))
        _check_coco_manifest(repo_root, raw_rows, expected_samples, dataset_config, "manifest_raw", failures)
        _check_coco_manifest(repo_root, filtered_rows, expected_samples, dataset_config, "manifest_filtered", failures)
        _check_coco_category_outputs(paths, category_order, stats, failures)
        _check_splits(filtered_rows, paths, config["split"], dataset_config, failures)
        _check_hashes(filtered_rows, paths, failures)
        _check_no_silent_fallback(paths, failures)
    summary = _base_summary(COCO_DATASET, processed_root, output_presence, failures, raw_rows, filtered_rows, paths)
    summary.update(
        {
            "caption_image_count": stats["caption_image_count"],
            "instance_image_count": stats["instance_image_count"],
            "category_count": stats["category_count"],
            "category_order": category_order,
            "zero_label_image_count": stats["zero_label_image_count"],
        }
    )
    write_json(paths["validator_summary"], summary)
    return summary


def _check_coco_manifest(repo_root: Path, rows: list[dict[str, Any]], expected_samples: list[dict[str, Any]], config: dict[str, Any], name: str, failures: list[str]) -> None:
    if len(rows) != int(config["expected_filtered_count"]):
        failures.append(f"{name} count mismatch: expected {config['expected_filtered_count']}, got {len(rows)}")
    expected_by_id = {sample["sample_id"]: sample for sample in expected_samples}
    ids = [row.get("sample_id") for row in rows]
    if len(ids) != len(set(ids)):
        failures.append(f"{name} sample_id is not unique")
    if ids != [sample["sample_id"] for sample in expected_samples]:
        failures.append(f"{name} sample_id order/content does not match COCO image_id ascending order")
    for row in rows:
        sample_id = row.get("sample_id")
        expected = expected_by_id.get(sample_id)
        if expected is None:
            failures.append(f"{name} unexpected COCO sample_id: {sample_id}")
            continue
        _check_coco_row(repo_root, row, expected, name, failures)


def _check_coco_row(repo_root: Path, row: dict[str, Any], expected: dict[str, Any], name: str, failures: list[str]) -> None:
    sample_id = expected["sample_id"]
    missing = REQUIRED_FIELDS - set(row)
    if missing:
        failures.append(f"{name} {sample_id} missing fields: {sorted(missing)}")
        return
    if row.get("sample_id") != sample_id or not COCO_SAMPLE_RE.match(str(row.get("sample_id"))):
        failures.append(f"{name} invalid sample_id: expected {sample_id}, got {row.get('sample_id')}")
    if row.get("dataset_name") != COCO_DATASET:
        failures.append(f"{name} invalid dataset_name for {sample_id}: {row.get('dataset_name')}")
    if row.get("image_path") != expected["image_path"] or not (repo_root / expected["image_path"]).is_file():
        failures.append(f"{name} image_path mismatch or missing for {sample_id}")
    if row.get("text_source") != expected["text_source"]:
        failures.append(f"{name} text_source mismatch for {sample_id}")
    vector = row.get("label_vector")
    if not isinstance(vector, list) or len(vector) != 80 or any(value not in (0, 1) for value in vector):
        failures.append(f"{name} label_vector is not 80-dim binary for {sample_id}")
    if vector != expected["label_vector"]:
        failures.append(f"{name} label_vector mismatch for {sample_id}")
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    expected_meta = expected["meta"]
    for key in ("caption_count", "caption_annotation_ids", "coco_split", "instance_annotation_count", "category_positive_count"):
        if meta.get(key) != expected_meta[key]:
            failures.append(f"{name} meta.{key} mismatch for {sample_id}")
    if int(meta.get("caption_count", 0)) < 1:
        failures.append(f"{name} caption_count < 1 for {sample_id}")
    if meta.get("caption_annotation_ids") != sorted(meta.get("caption_annotation_ids", [])):
        failures.append(f"{name} caption_annotation_ids are not sorted for {sample_id}")


def _check_coco_category_outputs(paths: dict[str, Path], category_order: list[dict[str, Any]], stats: dict[str, Any], failures: list[str]) -> None:
    manifest_meta = read_json(paths["manifest_meta"])
    preprocess_summary = read_json(paths["preprocess_summary"])
    for name, payload in (("manifest_meta", manifest_meta), ("preprocess_summary", preprocess_summary)):
        if payload.get("category_order") != category_order:
            failures.append(f"{name}.category_order does not match frozen COCO category id order")
        if payload.get("category_count") != stats["category_count"]:
            failures.append(f"{name}.category_count mismatch")
        if payload.get("zero_label_image_count") != stats["zero_label_image_count"]:
            failures.append(f"{name}.zero_label_image_count mismatch")


def _check_nus_raw_manifest(
    repo_root: Path,
    raw_root: Path,
    paths: dict[str, Path],
    image_index: list[dict[str, Any]],
    tag_vocab: list[str],
    top_label_columns: dict[str, bytearray],
    concept_items: list[dict[str, Any]],
    failures: list[str],
) -> tuple[int, int, dict[str, dict[str, Any]]]:
    vocab_index = {tag: index for index, tag in enumerate(tag_vocab)}
    concept_names = [item["name"] for item in concept_items]
    expected_filtered: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    empty_tag_rows = 0
    raw_count = 0
    all_tags_path = raw_root / "extracted" / "tags" / "All_Tags.txt"
    with all_tags_path.open("r", encoding="utf-8-sig") as all_tags_handle:
        for raw_count, (row, tag_line, image_record) in enumerate(zip(iter_jsonl(paths["manifest_raw"]), all_tags_handle, image_index), start=1):
            raw_index = int(image_record["raw_index"])
            expected_id = f"nus_{raw_index:06d}"
            _check_nus_row(repo_root, row, expected_id, image_record["repo_image_path"], failures, require_image_exists=False)
            if row.get("sample_id") in seen:
                failures.append(f"manifest_raw duplicate sample_id: {row.get('sample_id')}")
            seen.add(row.get("sample_id"))
            tags, _ = decode_nus_tag_line(tag_line, tag_vocab, vocab_index)
            expected_text = " ".join(tags)
            expected_vector = [int(top_label_columns[name][raw_count - 1]) for name in concept_names]
            empty_tag_rows += int(len(tags) == 0)
            if row.get("text_source") != expected_text:
                failures.append(f"NUS text_source mismatch for {expected_id}")
            if row.get("label_vector") != expected_vector:
                failures.append(f"NUS label_vector mismatch for {expected_id}")
            meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
            if meta.get("raw_tag_token_count") != len(tags):
                failures.append(f"NUS raw_tag_token_count mismatch for {expected_id}")
            if meta.get("raw_empty_tag_row") is not (len(tags) == 0):
                failures.append(f"NUS raw_empty_tag_row mismatch for {expected_id}")
            if sum(expected_vector) > 0:
                expected_filtered[expected_id] = {"text_source": expected_text, "label_vector": expected_vector, "image_path": image_record["repo_image_path"]}
    if raw_count != len(image_index):
        failures.append(f"NUS manifest_raw count mismatch against image_index: {raw_count} != {len(image_index)}")
    return raw_count, empty_tag_rows, expected_filtered


def _check_nus_filtered_manifest(repo_root: Path, rows: list[dict[str, Any]], expected: dict[str, dict[str, Any]], config: dict[str, Any], failures: list[str]) -> None:
    if len(rows) != int(config["expected_filtered_count"]):
        failures.append(f"manifest_filtered count mismatch: expected {config['expected_filtered_count']}, got {len(rows)}")
    ids = [row.get("sample_id") for row in rows]
    if ids != sorted(expected):
        failures.append("NUS manifest_filtered sample_id order/content does not match sum(label_vector)>0 filter")
    if len(ids) != len(set(ids)):
        failures.append("NUS manifest_filtered sample_id is not unique")
    for row in rows:
        sample_id = row.get("sample_id")
        expected_row = expected.get(sample_id)
        if expected_row is None:
            failures.append(f"NUS unexpected filtered sample_id: {sample_id}")
            continue
        _check_nus_row(repo_root, row, sample_id, expected_row["image_path"], failures, require_image_exists=True)
        if row.get("text_source") != expected_row["text_source"]:
            failures.append(f"NUS filtered text_source mismatch for {sample_id}")
        if row.get("label_vector") != expected_row["label_vector"]:
            failures.append(f"NUS filtered label_vector mismatch for {sample_id}")


def _check_nus_row(repo_root: Path, row: dict[str, Any], expected_id: str, expected_image_path: str, failures: list[str], require_image_exists: bool) -> None:
    missing = REQUIRED_FIELDS - set(row)
    if missing:
        failures.append(f"NUS row {expected_id} missing fields: {sorted(missing)}")
        return
    if row.get("sample_id") != expected_id or not NUS_SAMPLE_RE.match(str(row.get("sample_id"))):
        failures.append(f"NUS invalid sample_id: expected {expected_id}, got {row.get('sample_id')}")
    if row.get("dataset_name") != NUS_DATASET:
        failures.append(f"NUS invalid dataset_name for {expected_id}: {row.get('dataset_name')}")
    if row.get("image_path") != expected_image_path:
        failures.append(f"NUS image_path mismatch for {expected_id}")
    if require_image_exists and not (repo_root / expected_image_path).is_file():
        failures.append(f"NUS filtered image_path missing for {expected_id}: {expected_image_path}")
    vector = row.get("label_vector")
    if not isinstance(vector, list) or len(vector) != 10 or any(value not in (0, 1) for value in vector):
        failures.append(f"NUS label_vector is not 10-dim binary for {expected_id}")
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    if "raw_empty_tag_row" not in meta:
        failures.append(f"NUS missing meta.raw_empty_tag_row for {expected_id}")
    if "raw_tag_token_count" not in meta:
        failures.append(f"NUS missing meta.raw_tag_token_count for {expected_id}")
    if meta.get("text_source_protocol") != "final_tag_list_projected_binary_decode_v1":
        failures.append(f"NUS invalid text_source_protocol for {expected_id}")


def _check_nus_concept_outputs(paths: dict[str, Path], expected_items: list[dict[str, Any]], failures: list[str]) -> None:
    manifest_meta = read_json(paths["manifest_meta"])
    preprocess_summary = read_json(paths["preprocess_summary"])
    for name, payload in (("manifest_meta", manifest_meta), ("preprocess_summary", preprocess_summary)):
        if payload.get("concept_subset") != expected_items:
            failures.append(f"{name}.concept_subset does not match frozen top-10 concept order")
        counts = {item["name"]: item["positive_count"] for item in expected_items}
        if payload.get("concept_positive_counts") != counts:
            failures.append(f"{name}.concept_positive_counts does not match frozen top-10 counts")


def _check_no_kaggle_top10(paths: dict[str, Path], failures: list[str]) -> None:
    manifest_meta = read_json(paths["manifest_meta"])
    preprocess_summary = read_json(paths["preprocess_summary"])
    if manifest_meta.get("deprecated_kaggle_top10_used") is not False:
        failures.append("manifest_meta.deprecated_kaggle_top10_used is not false")
    if preprocess_summary.get("deprecated_kaggle_top10_used") is not False:
        failures.append("preprocess_summary.deprecated_kaggle_top10_used is not false")


def _check_output_presence(paths: dict[str, Path], failures: list[str]) -> dict[str, bool]:
    output_presence = {name: path.exists() and path.is_file() for name, path in paths.items() if name != "validator_summary"}
    failures.extend(f"missing output file: {name}={paths[name]}" for name, exists in output_presence.items() if not exists)
    output_presence["validator_summary"] = True
    return output_presence


def _check_counts(raw_rows: list[dict[str, Any]], filtered_rows: list[dict[str, Any]], config: dict[str, Any], failures: list[str]) -> None:
    if len(raw_rows) != int(config["expected_raw_count"]):
        failures.append(f"manifest_raw count mismatch: expected {config['expected_raw_count']}, got {len(raw_rows)}")
    if len(filtered_rows) != int(config["expected_filtered_count"]):
        failures.append(f"manifest_filtered count mismatch: expected {config['expected_filtered_count']}, got {len(filtered_rows)}")


def _check_manifest_rows(rows: list[dict[str, Any]], config: dict[str, Any], dataset: str, sample_re: re.Pattern[str], name: str, failures: list[str]) -> None:
    label_dimension = int(config["label_dimension"])
    for index, row in enumerate(rows, start=1):
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            failures.append(f"{name} row {index} missing fields: {sorted(missing)}")
            continue
        sample_id = row["sample_id"]
        if not isinstance(sample_id, str) or not sample_re.match(sample_id):
            failures.append(f"{name} row {index} invalid sample_id: {sample_id}")
        if row["dataset_name"] != dataset:
            failures.append(f"{name} row {index} invalid dataset_name: {row['dataset_name']}")
        vector = row["label_vector"]
        if not isinstance(vector, list) or len(vector) != label_dimension or any(value not in (0, 1) for value in vector):
            failures.append(f"{name} row {index} label_vector is not {label_dimension}-dim binary")
        meta = row["meta"] if isinstance(row.get("meta"), dict) else {}
        if "annotation_positive_count" not in meta:
            failures.append(f"{name} row {index} missing meta.annotation_positive_count")
        if "raw_tag_token_count" not in meta:
            failures.append(f"{name} row {index} missing meta.raw_tag_token_count")
        if isinstance(vector, list) and "annotation_positive_count" in meta and int(meta["annotation_positive_count"]) != sum(vector):
            failures.append(f"{name} row {index} annotation_positive_count mismatch")


def _check_filtered_text(rows: list[dict[str, Any]], failures: list[str]) -> None:
    empty_ids = [row.get("sample_id") for row in rows if not str(row.get("text_source", "")).strip()]
    if empty_ids:
        failures.append(f"empty text found in manifest_filtered: {empty_ids[:5]}")


def _check_sample_ids(rows: list[dict[str, Any]], name: str, failures: list[str]) -> None:
    ids = [row.get("sample_id") for row in rows]
    if len(ids) != len(set(ids)):
        failures.append(f"{name} sample_id is not unique")


def _check_splits(filtered_rows: list[dict[str, Any]], paths: dict[str, Path], split_config: dict[str, Any], dataset_config: dict[str, Any], failures: list[str]) -> None:
    query_ids = _read_lines(paths["query_ids"])
    retrieval_ids = _read_lines(paths["retrieval_ids"])
    train_ids = _read_lines(paths["train_ids"])
    if len(query_ids) != int(split_config["query_count"]):
        failures.append(f"query count mismatch: expected {split_config['query_count']}, got {len(query_ids)}")
    if len(retrieval_ids) != int(dataset_config["expected_retrieval_count"]):
        failures.append(f"retrieval count mismatch: expected {dataset_config['expected_retrieval_count']}, got {len(retrieval_ids)}")
    if len(train_ids) != int(split_config["train_count"]):
        failures.append(f"train count mismatch: expected {split_config['train_count']}, got {len(train_ids)}")
    if set(query_ids) & set(retrieval_ids):
        failures.append("query and retrieval splits overlap")
    if not set(train_ids).issubset(set(retrieval_ids)):
        failures.append("train split is not a retrieval subset")
    expected = make_split(
        [row["sample_id"] for row in filtered_rows],
        seed=int(split_config["seed"]),
        query_count=int(split_config["query_count"]),
        train_count=int(split_config["train_count"]),
    )
    if query_ids != expected["query_ids"] or retrieval_ids != expected["retrieval_ids"] or train_ids != expected["train_ids"]:
        failures.append("split ids do not match formal seed=0 permutation")


def _check_hashes(filtered_rows: list[dict[str, Any]], paths: dict[str, Path], failures: list[str]) -> None:
    hashes = read_json(paths["order_hashes"])
    expected = {
        "sample_id_order_sha256": hash_lines(sorted(row["sample_id"] for row in filtered_rows)),
        "manifest_filtered_order_sha256": hash_lines(row["sample_id"] for row in filtered_rows),
        "query_ids_sha256": hash_lines(_read_lines(paths["query_ids"])),
        "retrieval_ids_sha256": hash_lines(_read_lines(paths["retrieval_ids"])),
        "train_ids_sha256": hash_lines(_read_lines(paths["train_ids"])),
    }
    for key, value in expected.items():
        if hashes.get(key) != value:
            failures.append(f"order_hashes mismatch for {key}: expected {value}, got {hashes.get(key)}")


def _check_mir_label_positive_contract(
    filtered_rows: list[dict[str, Any]],
    paths: dict[str, Path],
    config: dict[str, Any],
    failures: list[str],
) -> dict[str, Any]:
    preprocess_summary = read_json(paths["preprocess_summary"])
    manifest_meta = read_json(paths["manifest_meta"])
    expected_policy = config["datasets"][MIR_DATASET]["filter_policy"]
    if expected_policy != "mir_pragmatic_high_signal_label_positive_v2":
        failures.append(f"MIR filtering_policy is not mir_pragmatic_high_signal_label_positive_v2: {expected_policy}")
    if preprocess_summary.get("filter_policy") != expected_policy:
        failures.append("preprocess_summary.filter_policy does not match MIR config")
    filter_stats = manifest_meta.get("filter_stats", {}) if isinstance(manifest_meta.get("filter_stats"), dict) else {}
    if filter_stats.get("filter_policy") != expected_policy:
        failures.append("manifest_meta.filter_stats.filter_policy does not match MIR config")

    query_ids = _read_lines(paths["query_ids"])
    retrieval_ids = _read_lines(paths["retrieval_ids"])
    train_ids = _read_lines(paths["train_ids"])
    by_id = {str(row["sample_id"]): row for row in filtered_rows}
    label_masks = {sample_id: _label_mask(row) for sample_id, row in by_id.items()}
    zero_filtered = sum(mask == 0 for mask in label_masks.values())
    zero_query = sum(label_masks[sample_id] == 0 for sample_id in query_ids)
    zero_train = sum(label_masks[sample_id] == 0 for sample_id in train_ids)
    zero_retrieval = sum(label_masks[sample_id] == 0 for sample_id in retrieval_ids)
    retrieval_masks = [label_masks[sample_id] for sample_id in retrieval_ids]
    no_relevant = sum(1 for sample_id in query_ids if not _has_relevant(label_masks[sample_id], retrieval_masks))

    if zero_filtered != 0:
        failures.append(f"zero_label_filtered_count must be 0, got {zero_filtered}")
    if zero_query != 0:
        failures.append(f"zero_label_query_count must be 0, got {zero_query}")
    if zero_train != 0:
        failures.append(f"zero_label_train_count must be 0, got {zero_train}")
    if zero_retrieval != 0:
        failures.append(f"zero_label_retrieval_count must be 0, got {zero_retrieval}")
    if no_relevant != 0:
        failures.append(f"query_with_no_relevant_retrieval_count must be 0, got {no_relevant}")

    return {
        "filtering_policy": expected_policy,
        "zero_label_filtered_count": zero_filtered,
        "zero_label_query_count": zero_query,
        "zero_label_train_count": zero_train,
        "zero_label_retrieval_count": zero_retrieval,
        "query_with_no_relevant_retrieval_count": no_relevant,
        "query_with_no_relevant_retrieval_rate": no_relevant / len(query_ids) if query_ids else 0.0,
    }


def _label_mask(row: dict[str, Any]) -> int:
    mask = 0
    for index, value in enumerate(row.get("label_vector", [])):
        if int(value):
            mask |= 1 << index
    return mask


def _has_relevant(query_mask: int, retrieval_masks: list[int]) -> bool:
    if query_mask == 0:
        return False
    return any(query_mask & mask for mask in retrieval_masks)


def _check_no_silent_fallback(paths: dict[str, Path], failures: list[str]) -> None:
    manifest_meta = read_json(paths["manifest_meta"])
    preprocess_summary = read_json(paths["preprocess_summary"])
    if manifest_meta.get("no_silent_fallback") is not True:
        failures.append("manifest_meta.no_silent_fallback is not true")
    if preprocess_summary.get("silent_fallback_used") is not False:
        failures.append("preprocess_summary.silent_fallback_used is not false")


def _base_summary(dataset: str, processed_root: Path, output_presence: dict[str, bool], failures: list[str], raw_rows: list[dict[str, Any]], filtered_rows: list[dict[str, Any]], paths: dict[str, Path]) -> dict[str, Any]:
    substage_by_dataset = {
        MIR_DATASET: "stage1a_mirflickr25k",
        NUS_DATASET: "stage1b_nuswide",
        COCO_DATASET: "stage1c_mscoco",
    }
    return {
        "stage": "stage1",
        "substage": substage_by_dataset[dataset],
        "validator_version": STAGE1_VALIDATOR_VERSION,
        "generated_at_utc": _utc_now(),
        "dataset_name": dataset,
        "processed_root": str(processed_root),
        "passed": len(failures) == 0,
        "failure_count": len(failures),
        "failure_reason": failures,
        "manifest_raw_count": len(raw_rows),
        "manifest_filtered_count": len(filtered_rows),
        "query_count": _line_count(paths["query_ids"]) if paths["query_ids"].exists() else 0,
        "retrieval_count": _line_count(paths["retrieval_ids"]) if paths["retrieval_ids"].exists() else 0,
        "train_count": _line_count(paths["train_ids"]) if paths["train_ids"].exists() else 0,
        "output_file_presence": output_presence,
        "silent_fallback_used": False,
    }


def _read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle if line.rstrip("\n")]


def _line_count(path: Path) -> int:
    return len(_read_lines(path))


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig") as handle:
        return sum(1 for _ in handle)


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


def _resolve_repo_path(repo_root: Path, path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
    resolved.relative_to(repo_root)
    return resolved


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

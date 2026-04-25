from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from src.datasets.validators.raw_preflight import (
    EXPECTED as PREFLIGHT_EXPECTED,
    IMAGE_EXTENSIONS,
    _count_files,
    _dataset_root,
    _ensure_within_repo,
    _load_config,
    _resolve_repo_path,
    check_mirflickr25k,
    check_mscoco,
    check_nuswide,
)

VALIDATOR_VERSION = "stage0r-4_raw_validator_v1"
DISALLOWED_GENERATED_FILENAMES = {
    "manifest.json",
    "raw_manifest.json",
    "dataset_manifest.json",
    "split.json",
    "splits.json",
    "train_split.json",
    "val_split.json",
    "test_split.json",
    "query_split.json",
    "database_split.json",
    "retrieval_split.json",
    "x_i.npy",
    "x_t.npy",
    "a.npy",
    "r.npy",
    "se.npy",
    "c.npy",
    "s.npy",
}


def run_stage0_raw_validator(repo_root: Path, config_path: Path, output_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config_full_path = _resolve_repo_path(repo_root, config_path)
    output_root_full_path = _resolve_repo_path(repo_root, output_root)
    _ensure_within_repo(repo_root, output_root_full_path)
    config = _load_config(config_full_path)
    generated_at = _utc_now()

    audits = {
        "mirflickr25k": validate_mirflickr25k(repo_root, config["mirflickr25k"], generated_at),
        "mscoco": validate_mscoco(repo_root, config["mscoco"], generated_at),
        "nuswide": validate_nuswide(repo_root, config["nuswide"], generated_at),
    }
    dataset_results: dict[str, Any] = {}
    for dataset, audit in audits.items():
        raw_root = Path(audit["raw_root"])
        dataset_summary = _dataset_summary(audit)
        _write_json(raw_root / "raw_audit.json", audit)
        _write_json(raw_root / "raw_validator_summary.json", dataset_summary)
        dataset_results[dataset] = dataset_summary

    failed_datasets = [name for name, result in dataset_results.items() if not result["passed"]]
    blocking_reasons = {name: result["failure_reason"] for name, result in dataset_results.items() if result["failure_reason"]}
    summary = {
        "stage": "stage0",
        "substage": "stage0r-4",
        "validator_version": VALIDATOR_VERSION,
        "generated_at_utc": generated_at,
        "config_path": str(config_full_path),
        "output_root": str(output_root_full_path),
        "all_raw_validators_passed": not failed_datasets,
        "dataset_results": dataset_results,
        "failed_datasets": failed_datasets,
        "blocking_reasons": blocking_reasons,
        "stage1_allowed": False,
        "stage1_allowed_reason": "Stage 0 CLIP weights are not yet validated",
        "no_silent_fallback": True,
        "clip_processed": False,
        "stage1_entered": False,
    }
    output_root_full_path.mkdir(parents=True, exist_ok=True)
    summary_json_path = output_root_full_path / "raw_validator_summary.json"
    summary_markdown_path = output_root_full_path / "raw_validator_summary.md"
    _write_json(summary_json_path, summary)
    summary_markdown_path.write_text(_render_markdown(summary), encoding="utf-8")
    summary["summary_json_path"] = str(summary_json_path)
    summary["summary_markdown_path"] = str(summary_markdown_path)
    return summary


def validate_mirflickr25k(repo_root: Path, config: dict[str, Any], generated_at: str) -> dict[str, Any]:
    raw_root = _dataset_root(repo_root, config, "mirflickr25k")
    preflight = check_mirflickr25k(repo_root, config, generated_at)
    failures = list(preflight["failure_reason"])
    generated_scan = _scan_disallowed_generated_files(raw_root)
    if generated_scan["detected"]:
        failures.append(f"Generated manifest/split/features/semantic files detected: {generated_scan['files']}")
    required_counts = {
        "mir_raw_image_count": PREFLIGHT_EXPECTED["mir_images"],
        "mir_tag_count": PREFLIGHT_EXPECTED["mir_tags"],
        "mir_exif_count": PREFLIGHT_EXPECTED["mir_tags"],
        "mir_annotation_txt_count": PREFLIGHT_EXPECTED["mir_annotations"],
    }
    observed_counts = {
        "mir_raw_image_count": preflight["image_file_count"],
        "mir_tag_count": preflight["tag_file_count"],
        "mir_exif_count": preflight["exif_file_count"],
        "mir_annotation_txt_count": preflight["annotation_txt_file_count"],
    }
    _append_count_failures(failures, required_counts, observed_counts)
    path_closure = {
        "image_count_closed": observed_counts["mir_raw_image_count"] == required_counts["mir_raw_image_count"],
        "tag_count_closed": observed_counts["mir_tag_count"] == required_counts["mir_tag_count"],
        "exif_count_closed": observed_counts["mir_exif_count"] == required_counts["mir_exif_count"],
        "annotation_count_closed": observed_counts["mir_annotation_txt_count"] == required_counts["mir_annotation_txt_count"],
        "readme_exists": preflight["readme_exists"],
    }
    key_files = {
        "mirflickr25k.zip": raw_root / "mirflickr25k.zip",
        "mirflickr25k_annotations_v080.zip": raw_root / "mirflickr25k_annotations_v080.zip",
        "README.txt": raw_root / "extracted" / "README.txt",
    }
    return _base_audit(
        dataset="mirflickr25k",
        source_protocol="mirflickr25k_raw_v1",
        generated_at=generated_at,
        raw_root=raw_root,
        required_files=_required_from_preflight(preflight),
        required_counts=required_counts,
        observed_counts=observed_counts,
        path_closure=path_closure,
        key_files=key_files,
        failures=failures,
        extra={
            "mir_raw_image_count": observed_counts["mir_raw_image_count"],
            "mir_tag_count": observed_counts["mir_tag_count"],
            "mir_exif_count": observed_counts["mir_exif_count"],
            "mir_annotation_txt_count": observed_counts["mir_annotation_txt_count"],
            "mir_archive_present": preflight["mirflickr25k_zip_exists"] and preflight["mirflickr25k_annotations_v080_zip_exists"],
            "generated_files_scan": generated_scan,
        },
    )


def validate_mscoco(repo_root: Path, config: dict[str, Any], generated_at: str) -> dict[str, Any]:
    raw_root = _dataset_root(repo_root, config, "mscoco")
    preflight = check_mscoco(repo_root, config, generated_at)
    failures = list(preflight["failure_reason"])
    generated_scan = _scan_disallowed_generated_files(raw_root)
    if generated_scan["detected"]:
        failures.append(f"Generated manifest/split/features/semantic files detected: {generated_scan['files']}")
    instances_category_counts = [
        check["category_count"]
        for name, check in preflight["json_checks"].items()
        if name.startswith("instances")
    ]
    required_counts = {
        "coco_train_image_count": PREFLIGHT_EXPECTED["coco_train"],
        "coco_val_image_count": PREFLIGHT_EXPECTED["coco_val"],
        "coco_total_image_count": PREFLIGHT_EXPECTED["coco_train"] + PREFLIGHT_EXPECTED["coco_val"],
        "coco_instances_category_count": PREFLIGHT_EXPECTED["coco_categories"],
    }
    observed_counts = {
        "coco_train_image_count": preflight["train2014_image_file_count"],
        "coco_val_image_count": preflight["val2014_image_file_count"],
        "coco_total_image_count": preflight["train2014_image_file_count"] + preflight["val2014_image_file_count"],
        "coco_instances_category_count": instances_category_counts,
    }
    _append_count_failures(failures, {k: v for k, v in required_counts.items() if k != "coco_instances_category_count"}, observed_counts)
    if any(count != PREFLIGHT_EXPECTED["coco_categories"] for count in instances_category_counts):
        failures.append(f"coco_instances_category_count mismatch: expected 80, got {instances_category_counts}")
    path_closure = {
        "train_count_closed": observed_counts["coco_train_image_count"] == required_counts["coco_train_image_count"],
        "val_count_closed": observed_counts["coco_val_image_count"] == required_counts["coco_val_image_count"],
        "total_count_closed": observed_counts["coco_total_image_count"] == required_counts["coco_total_image_count"],
        "json_readable_count": sum(1 for item in preflight["json_checks"].values() if item["readable"]),
        "json_expected_count": 4,
    }
    key_files = {
        "train2014.zip": raw_root / "train2014.zip",
        "val2014.zip": raw_root / "val2014.zip",
        "annotations_trainval2014.zip": raw_root / "annotations_trainval2014.zip",
        "captions_train2014.json": raw_root / "extracted" / "annotations" / "captions_train2014.json",
        "captions_val2014.json": raw_root / "extracted" / "annotations" / "captions_val2014.json",
        "instances_train2014.json": raw_root / "extracted" / "annotations" / "instances_train2014.json",
        "instances_val2014.json": raw_root / "extracted" / "annotations" / "instances_val2014.json",
    }
    return _base_audit(
        dataset="mscoco",
        source_protocol="mscoco_2014_raw_v1",
        generated_at=generated_at,
        raw_root=raw_root,
        required_files=_required_from_preflight(preflight),
        required_counts=required_counts,
        observed_counts=observed_counts,
        path_closure=path_closure,
        key_files=key_files,
        failures=failures,
        extra={
            "coco_train_image_count": observed_counts["coco_train_image_count"],
            "coco_val_image_count": observed_counts["coco_val_image_count"],
            "coco_total_image_count": observed_counts["coco_total_image_count"],
            "coco_json_readability": preflight["json_checks"],
            "coco_instances_category_count": instances_category_counts,
            "generated_files_scan": generated_scan,
        },
    )


def validate_nuswide(repo_root: Path, config: dict[str, Any], generated_at: str) -> dict[str, Any]:
    raw_root = _dataset_root(repo_root, config, "nuswide")
    preflight = check_nuswide(repo_root, config, generated_at)
    failures = list(preflight["failure_reason"])
    generated_scan = _scan_disallowed_generated_files(raw_root)
    if generated_scan["detected"]:
        failures.append(f"Generated manifest/split/features/semantic files detected: {generated_scan['files']}")
    required_counts = {
        "nus_all_tags_line_count": PREFLIGHT_EXPECTED["nus_all_tags_rows"],
        "nus_final_tag_list_line_count": PREFLIGHT_EXPECTED["nus_final_tag_list_rows"],
        "nus_image_index_line_count": PREFLIGHT_EXPECTED["nus_image_index_rows"],
        "nus_indexed_image_missing_count": 0,
        "nus_duplicate_raw_index_count": 0,
        "nus_duplicate_image_relative_path_count": 0,
    }
    observed_counts = {
        "nus_all_tags_line_count": preflight["all_tags_txt_line_count"],
        "nus_final_tag_list_line_count": preflight["final_tag_list_txt_line_count"],
        "nus_image_index_line_count": preflight["image_index_line_count"],
        "nus_indexed_image_missing_count": preflight["missing_indexed_image_file_count"],
        "nus_duplicate_raw_index_count": preflight["duplicate_raw_index_count"],
        "nus_duplicate_image_relative_path_count": preflight["duplicate_image_relative_path_count"],
        "nus_image_like_count": preflight["image_like_count"],
    }
    _append_count_failures(failures, required_counts, observed_counts)
    if preflight["image_like_count"] < PREFLIGHT_EXPECTED["nus_images_min"]:
        failures.append(f"nus image-like count below minimum: {preflight['image_like_count']}")
    path_closure = {
        "images_dir_exists": preflight["images_dir_exists"],
        "image_index_exists": preflight["image_index_tsv_exists"],
        "image_index_line_count_closed": preflight["image_index_line_count"] == PREFLIGHT_EXPECTED["nus_image_index_rows"],
        "raw_index_contiguous": preflight["image_index_raw_index_contiguous"],
        "all_indexed_images_exist": preflight["missing_indexed_image_file_count"] == 0,
        "duplicate_raw_index_count": preflight["duplicate_raw_index_count"],
        "duplicate_image_relative_path_count": preflight["duplicate_image_relative_path_count"],
        "first_5_image_index_lines": preflight["image_index_check"]["first_5_lines"],
    }
    key_files = {
        "NUS-WIDE.zip": raw_root / "NUS-WIDE.zip",
        "NUS_WID_Tags.zip": raw_root / "NUS_WID_Tags.zip",
        "Final_Tag_List.txt": raw_root / "extracted" / "tags" / "Final_Tag_List.txt",
        "All_Tags.txt": raw_root / "extracted" / "tags" / "All_Tags.txt",
        "AllTags81.txt": raw_root / "extracted" / "tags" / "AllTags81.txt",
        "AllTags1k.txt": raw_root / "extracted" / "tags" / "AllTags1k.txt",
        "TagList1k.txt": raw_root / "extracted" / "tags" / "TagList1k.txt",
        "image_index.tsv": raw_root / "image_index.tsv",
    }
    return _base_audit(
        dataset="nuswide",
        source_protocol="original_ra_nus_image_index_v1",
        generated_at=generated_at,
        raw_root=raw_root,
        required_files=_required_from_preflight(preflight),
        required_counts=required_counts,
        observed_counts=observed_counts,
        path_closure=path_closure,
        key_files=key_files,
        failures=failures,
        extra={
            "nus_all_tags_line_count": observed_counts["nus_all_tags_line_count"],
            "nus_final_tag_list_line_count": observed_counts["nus_final_tag_list_line_count"],
            "nus_image_index_line_count": observed_counts["nus_image_index_line_count"],
            "nus_indexed_image_missing_count": observed_counts["nus_indexed_image_missing_count"],
            "nus_duplicate_raw_index_count": observed_counts["nus_duplicate_raw_index_count"],
            "nus_duplicate_image_relative_path_count": observed_counts["nus_duplicate_image_relative_path_count"],
            "nus_source_protocol": preflight["source_protocol"],
            "image_index_check": preflight["image_index_check"],
            "uses_img_tc10_as_formal_input": False,
            "uses_targets_onehot_tc10_as_formal_input": False,
            "uses_database_test_split_as_formal_split": False,
            "deprecated_leftovers_detected": preflight["deprecated_leftovers_detected"],
            "generated_files_scan": generated_scan,
        },
    )


def _base_audit(
    dataset: str,
    source_protocol: str,
    generated_at: str,
    raw_root: Path,
    required_files: dict[str, dict[str, Any]],
    required_counts: dict[str, Any],
    observed_counts: dict[str, Any],
    path_closure: dict[str, Any],
    key_files: dict[str, Path],
    failures: list[str],
    extra: dict[str, Any],
) -> dict[str, Any]:
    audit = {
        "dataset": dataset,
        "source_protocol": source_protocol,
        "stage": "stage0",
        "substage": "stage0r-4",
        "validator_version": VALIDATOR_VERSION,
        "generated_at_utc": generated_at,
        "raw_root": str(raw_root),
        "passed": len(failures) == 0,
        "failure_count": len(failures),
        "failure_reason": failures,
        "required_files": required_files,
        "required_counts": required_counts,
        "observed_counts": observed_counts,
        "path_closure": path_closure,
        "sha256_or_size_for_key_files": _size_for_key_files(key_files),
        "silent_fallback_used": False,
        "stage1_entered": False,
    }
    audit.update(extra)
    return audit


def _dataset_summary(audit: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "dataset": audit["dataset"],
        "source_protocol": audit["source_protocol"],
        "passed": audit["passed"],
        "failure_count": audit["failure_count"],
        "failure_reason": audit["failure_reason"],
        "required_files": audit["required_files"],
        "required_counts": audit["required_counts"],
        "observed_counts": audit["observed_counts"],
        "path_closure": audit["path_closure"],
        "sha256_or_size_for_key_files": audit["sha256_or_size_for_key_files"],
        "validator_version": audit["validator_version"],
    }
    for key in (
        "mir_raw_image_count",
        "mir_tag_count",
        "mir_exif_count",
        "mir_annotation_txt_count",
        "mir_archive_present",
        "coco_train_image_count",
        "coco_val_image_count",
        "coco_total_image_count",
        "coco_json_readability",
        "coco_instances_category_count",
        "nus_all_tags_line_count",
        "nus_final_tag_list_line_count",
        "nus_image_index_line_count",
        "nus_indexed_image_missing_count",
        "nus_duplicate_raw_index_count",
        "nus_duplicate_image_relative_path_count",
        "nus_source_protocol",
        "uses_img_tc10_as_formal_input",
        "uses_targets_onehot_tc10_as_formal_input",
        "uses_database_test_split_as_formal_split",
        "deprecated_leftovers_detected",
    ):
        if key in audit:
            summary[key] = audit[key]
    return summary


def _required_from_preflight(preflight: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return preflight.get("required_paths", {})


def _size_for_key_files(files: dict[str, Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, path in files.items():
        result[name] = {
            "path": str(path),
            "exists": path.exists() and path.is_file(),
            "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
            "sha256": None,
            "hash_policy": "size_only_for_large_raw_files",
        }
    return result


def _append_count_failures(failures: list[str], required: dict[str, int], observed: dict[str, Any]) -> None:
    for name, expected in required.items():
        actual = observed.get(name)
        if actual != expected:
            failures.append(f"{name} mismatch: expected {expected}, got {actual}")


def _scan_disallowed_generated_files(raw_root: Path) -> dict[str, Any]:
    files: list[str] = []
    if raw_root.exists() and raw_root.is_dir():
        for path in raw_root.rglob("*"):
            if path.is_file() and path.name.lower() in DISALLOWED_GENERATED_FILENAMES:
                files.append(str(path))
    return {"detected": bool(files), "files": files}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Stage 0 Raw Validator Summary",
        "",
        f"- Generated at UTC: `{summary['generated_at_utc']}`",
        f"- Validator version: `{summary['validator_version']}`",
        f"- All raw validators passed: `{str(summary['all_raw_validators_passed']).lower()}`",
        f"- Stage 1 allowed: `{str(summary['stage1_allowed']).lower()}`",
        f"- Stage 1 allowed reason: `{summary['stage1_allowed_reason']}`",
        "",
        "## Dataset Results",
        "",
    ]
    for dataset, result in summary["dataset_results"].items():
        lines.extend(
            [
                f"### {dataset}",
                "",
                f"- Source protocol: `{result['source_protocol']}`",
                f"- Passed: `{str(result['passed']).lower()}`",
                f"- Failure count: `{result['failure_count']}`",
            ]
        )
        if result["failure_reason"]:
            lines.append("- Failures:")
            for failure in result["failure_reason"]:
                lines.append(f"  - {failure}")
        else:
            lines.append("- Failures: none")
        lines.append("")
    return "\n".join(lines)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

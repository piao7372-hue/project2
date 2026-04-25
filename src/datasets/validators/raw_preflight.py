from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
EXPECTED = {
    "mir_images": 25000,
    "mir_tags": 25000,
    "mir_annotations": 38,
    "coco_train": 82783,
    "coco_val": 40504,
    "coco_categories": 80,
    "nus_all_tags_rows": 269648,
    "nus_final_tag_list_rows": 5018,
    "nus_image_index_rows": 269648,
    "nus_images_min": 269648,
}


class RawPreflightError(ValueError):
    pass


def run_raw_preflight(repo_root: Path, config_path: Path, output_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config_full_path = _resolve_repo_path(repo_root, config_path)
    output_root_full_path = _resolve_repo_path(repo_root, output_root)
    _ensure_within_repo(repo_root, output_root_full_path)

    config = _load_config(config_full_path)
    generated_at = _utc_now()

    results = {
        "mirflickr25k": check_mirflickr25k(repo_root, config["mirflickr25k"], generated_at),
        "mscoco": check_mscoco(repo_root, config["mscoco"], generated_at),
        "nuswide": check_nuswide(repo_root, config["nuswide"], generated_at),
    }

    for dataset, result in results.items():
        dataset_dir = output_root_full_path / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        _write_json(dataset_dir / "raw_preflight.json", result)

    summary = {
        "stage": "stage0",
        "substage": "stage0b-5",
        "generated_at_utc": generated_at,
        "config_path": str(config_full_path),
        "output_root": str(output_root_full_path),
        "all_preflight_passed": all(result["formal_raw_ready"] for result in results.values()),
        "datasets": results,
    }
    summary_path = output_root_full_path / "raw_preflight_summary.json"
    _write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def check_mirflickr25k(repo_root: Path, config: dict[str, Any], generated_at: str) -> dict[str, Any]:
    raw_root = _dataset_root(repo_root, config, "mirflickr25k")
    extracted = raw_root / "extracted"
    required = {
        "mirflickr25k_zip": raw_root / "mirflickr25k.zip",
        "annotations_zip": raw_root / "mirflickr25k_annotations_v080.zip",
        "images_dir": extracted / "images",
        "meta_tags_dir": extracted / "meta" / "tags",
        "meta_exif_dir": extracted / "meta" / "exif",
        "annotations_dir": extracted / "annotations",
        "readme": extracted / "README.txt",
    }
    existence = _path_existence(required)

    image_count = _count_files(required["images_dir"], IMAGE_EXTENSIONS)
    tag_count = _count_files(required["meta_tags_dir"])
    exif_count = _count_files(required["meta_exif_dir"])
    annotation_txt_count = _count_files(required["annotations_dir"], {".txt"})

    extracted_failures: list[str] = []
    _require_exists(extracted_failures, existence, "images_dir")
    _require_exists(extracted_failures, existence, "meta_tags_dir")
    _require_exists(extracted_failures, existence, "meta_exif_dir")
    _require_exists(extracted_failures, existence, "annotations_dir")
    _require_exists(extracted_failures, existence, "readme")
    _expect_count(extracted_failures, "MIRFlickr image count", image_count, EXPECTED["mir_images"])
    _expect_count(extracted_failures, "MIRFlickr tag count", tag_count, EXPECTED["mir_tags"])
    _expect_count(
        extracted_failures,
        "MIRFlickr annotation txt count",
        annotation_txt_count,
        EXPECTED["mir_annotations"],
    )

    archive_missing = [
        item
        for item in ("mirflickr25k_zip", "annotations_zip")
        if not existence[item]["exists"]
    ]
    formal_failures = [
        f"Missing required archive: {name} -> {existence[name]['path']}"
        for name in archive_missing
    ]
    formal_failures.extend(extracted_failures)

    extracted_content_usable = len(extracted_failures) == 0
    formal_raw_ready = len(formal_failures) == 0

    return {
        "dataset": "mirflickr25k",
        "stage": "stage0",
        "substage": "stage0b-5",
        "generated_at_utc": generated_at,
        "raw_root": str(raw_root),
        "raw_root_exists": raw_root.exists(),
        "required_paths": existence,
        "image_file_count": image_count,
        "expected_image_file_count": EXPECTED["mir_images"],
        "tag_file_count": tag_count,
        "expected_tag_file_count": EXPECTED["mir_tags"],
        "exif_file_count": exif_count,
        "annotation_txt_file_count": annotation_txt_count,
        "expected_annotation_txt_file_count": EXPECTED["mir_annotations"],
        "readme_exists": existence["readme"]["exists"],
        "mirflickr25k_zip_exists": existence["mirflickr25k_zip"]["exists"],
        "mirflickr25k_annotations_v080_zip_exists": existence["annotations_zip"]["exists"],
        "archive_missing": len(archive_missing) > 0,
        "archive_missing_items": archive_missing,
        "extracted_content_usable": extracted_content_usable,
        "formal_raw_ready": formal_raw_ready,
        "v3_raw_layout_closed": formal_raw_ready,
        "failure_reason": formal_failures,
    }


def check_mscoco(repo_root: Path, config: dict[str, Any], generated_at: str) -> dict[str, Any]:
    raw_root = _dataset_root(repo_root, config, "mscoco")
    annotations_dir = raw_root / "extracted" / "annotations"
    required = {
        "train2014_zip": raw_root / "train2014.zip",
        "val2014_zip": raw_root / "val2014.zip",
        "annotations_trainval2014_zip": raw_root / "annotations_trainval2014.zip",
        "train2014_dir": raw_root / "extracted" / "train2014",
        "val2014_dir": raw_root / "extracted" / "val2014",
        "captions_train2014_json": annotations_dir / "captions_train2014.json",
        "captions_val2014_json": annotations_dir / "captions_val2014.json",
        "instances_train2014_json": annotations_dir / "instances_train2014.json",
        "instances_val2014_json": annotations_dir / "instances_val2014.json",
    }
    existence = _path_existence(required)

    train_count = _count_files(required["train2014_dir"], IMAGE_EXTENSIONS)
    val_count = _count_files(required["val2014_dir"], IMAGE_EXTENSIONS)
    json_checks = {
        name: _check_coco_json(path)
        for name, path in required.items()
        if name.endswith("_json")
    }

    extracted_failures: list[str] = []
    _require_exists(extracted_failures, existence, "train2014_dir")
    _require_exists(extracted_failures, existence, "val2014_dir")
    for name in json_checks:
        _require_exists(extracted_failures, existence, name)
    _expect_count(extracted_failures, "MSCOCO train2014 image count", train_count, EXPECTED["coco_train"])
    _expect_count(extracted_failures, "MSCOCO val2014 image count", val_count, EXPECTED["coco_val"])

    for name, check in json_checks.items():
        if not check["readable"]:
            extracted_failures.append(f"MSCOCO JSON is not readable: {name}: {check['error']}")
            continue
        if not check["has_images"] or not check["has_annotations"]:
            extracted_failures.append(f"MSCOCO JSON missing required top-level fields: {name}")
        if name.startswith("instances"):
            if not check["has_categories"]:
                extracted_failures.append(f"MSCOCO instances JSON missing top-level categories: {name}")
            elif check["category_count"] != EXPECTED["coco_categories"]:
                extracted_failures.append(
                    f"MSCOCO instances categories mismatch: {name} expected "
                    f"{EXPECTED['coco_categories']}, got {check['category_count']}."
                )

    archive_missing = [
        item
        for item in ("train2014_zip", "val2014_zip", "annotations_trainval2014_zip")
        if not existence[item]["exists"]
    ]
    formal_failures = [
        f"Missing required archive: {name} -> {existence[name]['path']}"
        for name in archive_missing
    ]
    formal_failures.extend(extracted_failures)

    extracted_content_usable = len(extracted_failures) == 0
    formal_raw_ready = len(formal_failures) == 0

    return {
        "dataset": "mscoco",
        "stage": "stage0",
        "substage": "stage0b-5",
        "generated_at_utc": generated_at,
        "raw_root": str(raw_root),
        "raw_root_exists": raw_root.exists(),
        "required_paths": existence,
        "train2014_image_file_count": train_count,
        "expected_train2014_image_file_count": EXPECTED["coco_train"],
        "val2014_image_file_count": val_count,
        "expected_val2014_image_file_count": EXPECTED["coco_val"],
        "json_checks": json_checks,
        "instances_categories_expected_count": EXPECTED["coco_categories"],
        "train2014_zip_exists": existence["train2014_zip"]["exists"],
        "val2014_zip_exists": existence["val2014_zip"]["exists"],
        "annotations_trainval2014_zip_exists": existence["annotations_trainval2014_zip"]["exists"],
        "archive_missing": len(archive_missing) > 0,
        "archive_missing_items": archive_missing,
        "extracted_content_usable": extracted_content_usable,
        "formal_raw_ready": formal_raw_ready,
        "v3_raw_layout_closed": formal_raw_ready,
        "failure_reason": formal_failures,
    }


def check_nuswide(repo_root: Path, config: dict[str, Any], generated_at: str) -> dict[str, Any]:
    raw_root = _dataset_root(repo_root, config, "nuswide")
    source_protocol = config.get("source_protocol")
    extracted = raw_root / "extracted"
    tags = extracted / "tags"
    images_dir = raw_root / "images"
    image_index_path = raw_root / "image_index.tsv"
    kaggle_top10 = raw_root / "kaggle_top10"
    required = {
        "NUS-WIDE.zip": raw_root / "NUS-WIDE.zip",
        "NUS_WID_Tags.zip": raw_root / "NUS_WID_Tags.zip",
        "extracted/Groundtruth": extracted / "Groundtruth",
        "extracted/ConceptsList": extracted / "ConceptsList",
        "extracted/tags/Final_Tag_List.txt": tags / "Final_Tag_List.txt",
        "extracted/tags/All_Tags.txt": tags / "All_Tags.txt",
        "extracted/tags/AllTags81.txt": tags / "AllTags81.txt",
        "extracted/tags/AllTags1k.txt": tags / "AllTags1k.txt",
        "extracted/tags/TagList1k.txt": tags / "TagList1k.txt",
        "images": images_dir,
        "image_index.tsv": image_index_path,
    }
    existence = _path_existence(required)

    image_like_count = _count_files(images_dir, IMAGE_EXTENSIONS)
    final_tag_list_rows = _count_lines(required["extracted/tags/Final_Tag_List.txt"])
    all_tags_rows = _count_lines(required["extracted/tags/All_Tags.txt"])
    image_index_check = _check_original_ra_image_index(image_index_path, images_dir)
    deprecated_leftovers = _deprecated_nus_leftovers(kaggle_top10)

    failures: list[str] = []
    if source_protocol != "original_ra_nus_image_index_v1":
        failures.append(
            "NUS source_protocol mismatch: expected original_ra_nus_image_index_v1, "
            f"got {source_protocol!r}."
        )
    for name in required:
        _require_exists(failures, existence, name)
    _expect_count(
        failures,
        "Final_Tag_List.txt line count",
        final_tag_list_rows,
        EXPECTED["nus_final_tag_list_rows"],
    )
    _expect_count(failures, "All_Tags.txt line count", all_tags_rows, EXPECTED["nus_all_tags_rows"])
    if image_like_count < EXPECTED["nus_images_min"]:
        failures.append(
            "NUS images image-like file count is below minimum: "
            f"expected at least {EXPECTED['nus_images_min']}, got {image_like_count}."
        )
    if not image_index_check["passed"]:
        failures.extend(image_index_check["failures"])

    formal_raw_ready = len(failures) == 0
    return {
        "dataset": "nuswide",
        "stage": "stage0",
        "substage": "stage0r-3",
        "source_protocol": source_protocol,
        "expected_source_protocol": "original_ra_nus_image_index_v1",
        "generated_at_utc": generated_at,
        "raw_root": str(raw_root),
        "raw_root_exists": raw_root.exists(),
        "required_paths": existence,
        "NUS-WIDE.zip_exists": existence["NUS-WIDE.zip"]["exists"],
        "NUS_WID_Tags.zip_exists": existence["NUS_WID_Tags.zip"]["exists"],
        "groundtruth_dir_exists": existence["extracted/Groundtruth"]["exists"],
        "concepts_list_dir_exists": existence["extracted/ConceptsList"]["exists"],
        "final_tag_list_txt_exists": existence["extracted/tags/Final_Tag_List.txt"]["exists"],
        "final_tag_list_txt_line_count": final_tag_list_rows,
        "final_tag_list_txt_expected_line_count": EXPECTED["nus_final_tag_list_rows"],
        "all_tags_txt_exists": existence["extracted/tags/All_Tags.txt"]["exists"],
        "all_tags_txt_line_count": all_tags_rows,
        "all_tags_txt_expected_line_count": EXPECTED["nus_all_tags_rows"],
        "alltags81_txt_exists": existence["extracted/tags/AllTags81.txt"]["exists"],
        "alltags1k_txt_exists": existence["extracted/tags/AllTags1k.txt"]["exists"],
        "taglist1k_txt_exists": existence["extracted/tags/TagList1k.txt"]["exists"],
        "images_dir_exists": existence["images"]["exists"],
        "image_like_count": image_like_count,
        "image_like_expected_min_count": EXPECTED["nus_images_min"],
        "image_index_tsv_exists": existence["image_index.tsv"]["exists"],
        "image_index_check": image_index_check,
        "image_index_line_count": image_index_check["line_count"],
        "image_index_expected_line_count": EXPECTED["nus_image_index_rows"],
        "image_index_all_rows_field_count_2": image_index_check["all_rows_field_count_2"],
        "image_index_raw_index_contiguous": image_index_check["raw_index_contiguous"],
        "duplicate_raw_index_count": image_index_check["duplicate_raw_index_count"],
        "duplicate_image_relative_path_count": image_index_check["duplicate_image_relative_path_count"],
        "missing_indexed_image_file_count": image_index_check["missing_image_file_count"],
        "uses_img_tc10_as_formal_input": False,
        "uses_targets_onehot_tc10_as_formal_input": False,
        "uses_database_test_split_as_formal_split": False,
        "deprecated_leftovers_detected": deprecated_leftovers,
        "official_text_source_ready": (
            existence["extracted/tags/Final_Tag_List.txt"]["exists"]
            and existence["extracted/tags/All_Tags.txt"]["exists"]
            and final_tag_list_rows == EXPECTED["nus_final_tag_list_rows"]
            and all_tags_rows == EXPECTED["nus_all_tags_rows"]
        ),
        "formal_raw_ready": formal_raw_ready,
        "original_ra_raw_layout_closed": formal_raw_ready,
        "failure_reason": failures,
    }


def _check_original_ra_image_index(image_index_path: Path, images_dir: Path) -> dict[str, Any]:
    result = {
        "path": str(image_index_path),
        "exists": image_index_path.exists(),
        "checked": False,
        "passed": False,
        "line_count": 0,
        "expected_line_count": EXPECTED["nus_image_index_rows"],
        "field_count_per_row_expected": 2,
        "all_rows_field_count_2": False,
        "invalid_field_count_rows": 0,
        "raw_index_contiguous": False,
        "duplicate_raw_index_count": 0,
        "duplicate_image_relative_path_count": 0,
        "missing_image_file_count": 0,
        "first_5_lines": [],
        "first_20_missing_image_examples": [],
        "first_20_invalid_examples": [],
        "path_separator_normalized": True,
        "failures": [],
    }
    if not image_index_path.exists() or not image_index_path.is_file():
        result["failures"].append(f"Missing required image_index.tsv: {image_index_path}")
        return result
    if not images_dir.exists() or not images_dir.is_dir():
        result["failures"].append(f"Missing required images directory for image_index.tsv validation: {images_dir}")
        return result

    raw_indexes: set[int] = set()
    image_relative_paths: set[str] = set()
    invalid_field_rows = 0
    duplicate_raw_index_count = 0
    duplicate_image_relative_path_count = 0
    missing_image_file_count = 0
    raw_index_contiguous = True
    path_separator_normalized = True

    with image_index_path.open("r", encoding="utf-8-sig", errors="strict") as handle:
        for expected_raw_index, line in enumerate(handle):
            stripped = line.rstrip("\n").rstrip("\r")
            if expected_raw_index < 5:
                result["first_5_lines"].append(stripped)
            result["line_count"] += 1
            parts = stripped.split("\t")
            if len(parts) != 2:
                invalid_field_rows += 1
                if len(result["first_20_invalid_examples"]) < 20:
                    result["first_20_invalid_examples"].append(
                        {"line_number": expected_raw_index + 1, "reason": "field_count", "value": stripped}
                    )
                continue

            raw_index_text, image_relative_path = parts
            try:
                raw_index = int(raw_index_text)
            except ValueError:
                raw_index_contiguous = False
                if len(result["first_20_invalid_examples"]) < 20:
                    result["first_20_invalid_examples"].append(
                        {"line_number": expected_raw_index + 1, "reason": "raw_index_not_int", "value": raw_index_text}
                    )
                continue

            if raw_index != expected_raw_index:
                raw_index_contiguous = False
                if len(result["first_20_invalid_examples"]) < 20:
                    result["first_20_invalid_examples"].append(
                        {
                            "line_number": expected_raw_index + 1,
                            "reason": "raw_index_not_contiguous",
                            "expected": expected_raw_index,
                            "actual": raw_index,
                        }
                    )
            if raw_index in raw_indexes:
                duplicate_raw_index_count += 1
            raw_indexes.add(raw_index)

            if "\\" in image_relative_path:
                path_separator_normalized = False
            relative_path = Path(image_relative_path)
            if relative_path.is_absolute() or ".." in relative_path.parts or not image_relative_path:
                if len(result["first_20_invalid_examples"]) < 20:
                    result["first_20_invalid_examples"].append(
                        {
                            "line_number": expected_raw_index + 1,
                            "reason": "unsafe_or_empty_image_relative_path",
                            "value": image_relative_path,
                        }
                    )
                continue

            rel_key = image_relative_path.replace("\\", "/").lower()
            if rel_key in image_relative_paths:
                duplicate_image_relative_path_count += 1
            image_relative_paths.add(rel_key)

            image_path = (images_dir / relative_path).resolve()
            try:
                image_path.relative_to(images_dir.resolve())
            except ValueError:
                if len(result["first_20_invalid_examples"]) < 20:
                    result["first_20_invalid_examples"].append(
                        {
                            "line_number": expected_raw_index + 1,
                            "reason": "image_path_escapes_images_dir",
                            "value": image_relative_path,
                        }
                    )
                continue
            if not image_path.exists() or not image_path.is_file():
                missing_image_file_count += 1
                if len(result["first_20_missing_image_examples"]) < 20:
                    result["first_20_missing_image_examples"].append(
                        {"raw_index": raw_index, "image_relative_path": image_relative_path}
                    )

    result["checked"] = True
    result["all_rows_field_count_2"] = invalid_field_rows == 0
    result["invalid_field_count_rows"] = invalid_field_rows
    result["raw_index_contiguous"] = raw_index_contiguous and result["line_count"] == EXPECTED["nus_image_index_rows"]
    result["duplicate_raw_index_count"] = duplicate_raw_index_count
    result["duplicate_image_relative_path_count"] = duplicate_image_relative_path_count
    result["missing_image_file_count"] = missing_image_file_count
    result["path_separator_normalized"] = path_separator_normalized

    if result["line_count"] != EXPECTED["nus_image_index_rows"]:
        result["failures"].append(
            f"image_index.tsv line count mismatch: expected {EXPECTED['nus_image_index_rows']}, got {result['line_count']}."
        )
    if invalid_field_rows:
        result["failures"].append(f"image_index.tsv has rows whose field count is not 2: {invalid_field_rows}.")
    if not result["raw_index_contiguous"]:
        result["failures"].append("image_index.tsv raw_index is not contiguous from 0 to 269647.")
    if duplicate_raw_index_count:
        result["failures"].append(f"image_index.tsv duplicate raw_index count is {duplicate_raw_index_count}.")
    if duplicate_image_relative_path_count:
        result["failures"].append(
            f"image_index.tsv duplicate image_relative_path count is {duplicate_image_relative_path_count}."
        )
    if missing_image_file_count:
        result["failures"].append(
            f"image_index.tsv references missing files under data/raw/nuswide/images: {missing_image_file_count}."
        )
    if not path_separator_normalized:
        result["failures"].append("image_index.tsv contains backslash path separators; expected normalized '/'.")

    result["passed"] = len(result["failures"]) == 0
    return result


def _deprecated_nus_leftovers(kaggle_top10: Path) -> list[dict[str, Any]]:
    names = ["img_tc10.txt", "targets_onehot_tc10.txt", "database_img.txt", "test_img.txt"]
    leftovers = []
    for name in names:
        path = kaggle_top10 / name
        leftovers.append(
            {
                "name": f"kaggle_top10/{name}",
                "path": str(path),
                "exists": path.exists() and path.is_file(),
                "status": "deprecated_leftover",
                "used_as_formal_input": False,
            }
        )
    return leftovers

def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing raw roots config: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        config = json.load(handle)
    expected = {"mirflickr25k", "mscoco", "nuswide"}
    actual = set(config)
    if actual != expected:
        raise RawPreflightError(f"Expected dataset keys {sorted(expected)}, got {sorted(actual)}")
    return config


def _dataset_root(repo_root: Path, config: dict[str, Any], dataset: str) -> Path:
    raw_root_value = config.get("raw_root")
    if not raw_root_value:
        raise RawPreflightError(f"Missing raw_root for {dataset}")
    raw_root = _resolve_repo_path(repo_root, Path(raw_root_value))
    _ensure_within_repo(repo_root, raw_root)
    return raw_root


def _path_existence(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "path": str(path),
            "exists": path.exists(),
            "is_file": path.is_file(),
            "is_dir": path.is_dir(),
        }
        for name, path in paths.items()
    }


def _require_exists(failures: list[str], existence: dict[str, dict[str, Any]], name: str) -> None:
    if not existence[name]["exists"]:
        failures.append(f"Missing required path: {name} -> {existence[name]['path']}")


def _count_files(path: Path, extensions: set[str] | None = None) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    count = 0
    for child in path.rglob("*"):
        if child.is_file() and (extensions is None or child.suffix.lower() in extensions):
            count += 1
    return count


def _count_lines(path: Path) -> int | None:
    if not path.exists() or not path.is_file():
        return None
    count = 0
    last_byte = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            count += chunk.count(b"\n")
            last_byte = chunk[-1:]
    if path.stat().st_size > 0 and last_byte not in (b"\n", b"\r"):
        count += 1
    return count


def _expect_count(failures: list[str], name: str, actual: int | None, expected: int) -> None:
    if actual is None:
        return
    if actual != expected:
        failures.append(f"{name} mismatch: expected {expected}, got {actual}.")


def _check_coco_json(path: Path) -> dict[str, Any]:
    result = {
        "path": str(path),
        "exists": path.exists(),
        "readable": False,
        "error": None,
        "top_level_fields": [],
        "has_images": False,
        "has_annotations": False,
        "has_categories": False,
        "image_count": None,
        "unique_image_id_count": None,
        "annotation_count": None,
        "category_count": None,
    }
    if not path.exists() or not path.is_file():
        result["error"] = "missing"
        return result
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except Exception as exc:  # pragma: no cover - captured in output
        result["error"] = repr(exc)
        return result

    result["readable"] = True
    if not isinstance(data, dict):
        result["error"] = "top-level JSON value is not an object"
        return result

    result["top_level_fields"] = sorted(data.keys())
    images = data.get("images")
    annotations = data.get("annotations")
    categories = data.get("categories")
    result["has_images"] = isinstance(images, list)
    result["has_annotations"] = isinstance(annotations, list)
    result["has_categories"] = isinstance(categories, list)
    if isinstance(images, list):
        image_ids = [entry.get("id") for entry in images if isinstance(entry, dict) and "id" in entry]
        result["image_count"] = len(images)
        result["unique_image_id_count"] = len(set(image_ids))
    if isinstance(annotations, list):
        result["annotation_count"] = len(annotations)
    if isinstance(categories, list):
        result["category_count"] = len(categories)
    return result


def _check_onehot_width(path: Path, expected_width: int) -> dict[str, Any]:
    result = {
        "path": str(path),
        "checked": False,
        "expected_width": expected_width,
        "all_rows_expected_width": False,
        "invalid_row_count": None,
        "invalid_row_examples": [],
    }
    if not path.exists() or not path.is_file():
        return result

    invalid_count = 0
    examples = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            width = len(stripped.replace(",", " ").split()) if stripped else 0
            if width != expected_width:
                invalid_count += 1
                if len(examples) < 20:
                    examples.append({"line": line_number, "width": width})
    result["checked"] = True
    result["all_rows_expected_width"] = invalid_count == 0
    result["invalid_row_count"] = invalid_count
    result["invalid_row_examples"] = examples
    return result


def _check_nus_images(img_list_path: Path, image_root: Path) -> dict[str, Any]:
    result = {
        "img_list_path": str(img_list_path),
        "image_root": str(image_root),
        "checked": False,
        "row_count": None,
        "unique_path_count": None,
        "duplicate_count": None,
        "all_images_exist": False,
        "missing_image_count": None,
        "missing_image_examples": [],
        "path_match_rules_tried": [
            "exact relative path under kaggle_top10/images",
            "normalized path after removing images/ prefix",
        ],
        "path_match_rule_counts": {},
    }
    if not img_list_path.exists() or not img_list_path.is_file() or not image_root.exists():
        return result

    seen: set[str] = set()
    duplicate_count = 0
    missing_count = 0
    missing_examples = []
    row_count = 0
    rule_counts = {
        "exact relative path under kaggle_top10/images": 0,
        "normalized path after removing images/ prefix": 0,
    }

    with img_list_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            rel = line.strip().replace("\\", "/")
            if not rel:
                continue
            row_count += 1
            if rel in seen:
                duplicate_count += 1
            else:
                seen.add(rel)

            matched_rule = _match_nus_image_rule(rel, image_root)
            if matched_rule is None:
                missing_count += 1
                if len(missing_examples) < 20:
                    missing_examples.append(rel)
            else:
                rule_counts[matched_rule] += 1

    result["checked"] = True
    result["row_count"] = row_count
    result["unique_path_count"] = len(seen)
    result["duplicate_count"] = duplicate_count
    result["all_images_exist"] = missing_count == 0
    result["missing_image_count"] = missing_count
    result["missing_image_examples"] = missing_examples
    result["path_match_rule_counts"] = rule_counts
    return result


def _match_nus_image_rule(rel: str, image_root: Path) -> str | None:
    exact = image_root / rel
    if exact.exists() and exact.is_file():
        return "exact relative path under kaggle_top10/images"
    normalized = rel[7:] if rel.lower().startswith("images/") else rel
    if normalized != rel:
        candidate = image_root / normalized
        if candidate.exists() and candidate.is_file():
            return "normalized path after removing images/ prefix"
    return None


def _resolve_repo_path(repo_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _ensure_within_repo(repo_root: Path, path: Path) -> None:
    path.relative_to(repo_root)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
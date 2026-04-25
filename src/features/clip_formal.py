from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizerFast

from src.features.stage2_baseline import compute_stage2_baseline
from src.utils.jsonl import iter_jsonl, read_json, write_json


SUPPORTED_STAGE2_DATASETS = {"mirflickr25k", "nuswide", "mscoco"}


def run_stage2_features(repo_root: Path, config_path: Path, dataset: str) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = read_json(_resolve_repo_path(repo_root, config_path))
    if dataset not in SUPPORTED_STAGE2_DATASETS or dataset not in config["datasets"]:
        raise ValueError("Stage 2 currently supports mirflickr25k, nuswide, and mscoco only")
    dataset_config = config["datasets"][dataset]
    processed_root = _resolve_repo_path(repo_root, Path(config["inputs"]["processed_root"])) / dataset
    paths = _stage2_input_paths(processed_root)
    output_dir = processed_root / "feature_cache" / config["outputs"]["feature_cache_dirname"]
    _ensure_within(repo_root, output_dir)
    _require_stage2_inputs(paths)

    manifest_meta = read_json(paths["manifest_meta"])
    order_hashes = read_json(paths["order_hashes"])
    rows = list(iter_jsonl(paths["manifest_filtered"]))
    query_ids = _read_lines(paths["query_ids"])
    retrieval_ids = _read_lines(paths["retrieval_ids"])
    train_ids = _read_lines(paths["train_ids"])
    _validate_stage2_inputs(rows, query_ids, retrieval_ids, train_ids, manifest_meta, dataset_config, dataset)

    model, image_processor, tokenizer = _load_clip(repo_root, config)
    feature_dim = int(model.config.projection_dim)
    if feature_dim != 512:
        raise RuntimeError(f"CLIP projection_dim must be 512, got {feature_dim}")

    output_dir.mkdir(parents=True, exist_ok=True)
    x_i = _extract_image_features(
        repo_root=repo_root,
        rows=rows,
        model=model,
        image_processor=image_processor,
        device=config["runtime"]["device"],
        batch_size=int(config["batches"]["image_batch_size"]),
        feature_dim=feature_dim,
    )
    x_t = _extract_text_features(
        rows=rows,
        model=model,
        tokenizer=tokenizer,
        device=config["runtime"]["device"],
        batch_size=int(config["batches"]["text_batch_size"]),
        feature_dim=feature_dim,
    )
    np.save(output_dir / "X_I.npy", x_i)
    np.save(output_dir / "X_T.npy", x_t)

    meta = _meta_payload(
        dataset=dataset,
        config=config,
        rows=rows,
        query_ids=query_ids,
        retrieval_ids=retrieval_ids,
        train_ids=train_ids,
        order_hashes=order_hashes,
        model=model,
        image_processor=image_processor,
        output_dir=output_dir,
    )
    write_json(output_dir / "meta.json", meta)
    baseline_summary = compute_stage2_baseline(
        dataset=dataset,
        feature_set_id=config["feature_set_id"],
        output_dir=output_dir,
        rows=rows,
        query_ids=query_ids,
        retrieval_ids=retrieval_ids,
        config=config,
        meta=meta,
    )
    return {
        "dataset": dataset,
        "feature_set_id": config["feature_set_id"],
        "output_dir": str(output_dir),
        "filtered_count": len(rows),
        "query_count": len(query_ids),
        "retrieval_count": len(retrieval_ids),
        "x_i_shape": list(x_i.shape),
        "x_t_shape": list(x_t.shape),
        "x_i_dtype": str(x_i.dtype),
        "x_t_dtype": str(x_t.dtype),
        "x_i_norm_min": float(np.linalg.norm(x_i, axis=1).min()),
        "x_i_norm_max": float(np.linalg.norm(x_i, axis=1).max()),
        "x_t_norm_min": float(np.linalg.norm(x_t, axis=1).min()),
        "x_t_norm_max": float(np.linalg.norm(x_t, axis=1).max()),
        "baseline_summary": baseline_summary,
    }


def _load_clip(repo_root: Path, config: dict[str, Any]) -> tuple[CLIPModel, CLIPImageProcessor, CLIPTokenizerFast]:
    runtime = config["runtime"]
    clip_config = config["clip"]
    if runtime["device"] != "cuda:0":
        raise RuntimeError(f"formal Stage 2 device must be cuda:0, got {runtime['device']}")
    if runtime["dtype"] != "float32":
        raise RuntimeError(f"formal Stage 2 dtype must be float32, got {runtime['dtype']}")
    if runtime.get("amp_enabled") is not False:
        raise RuntimeError("formal Stage 2 requires AMP disabled")
    if clip_config["backbone_id"] != "openai/clip-vit-base-patch32":
        raise RuntimeError(f"unsupported backbone_id: {clip_config['backbone_id']}")
    if clip_config.get("local_files_only") is not True:
        raise RuntimeError("formal Stage 2 requires local_files_only=true")
    if not torch.cuda.is_available():
        raise RuntimeError("formal Stage 2 requires CUDA; CPU fallback is forbidden")

    model_path = _resolve_repo_path(repo_root, Path(clip_config["model_local_path"]))
    if not model_path.is_dir():
        raise FileNotFoundError(f"local CLIP model directory missing: {model_path}")
    model = CLIPModel.from_pretrained(str(model_path), local_files_only=True)
    image_processor = CLIPImageProcessor.from_pretrained(str(model_path), local_files_only=True)
    tokenizer = CLIPTokenizerFast.from_pretrained(str(model_path), local_files_only=True)
    model.eval()
    model.float()
    model.to(torch.device(runtime["device"]))
    _verify_image_processor(image_processor)
    return model, image_processor, tokenizer


def _extract_image_features(
    repo_root: Path,
    rows: list[dict[str, Any]],
    model: CLIPModel,
    image_processor: CLIPImageProcessor,
    device: str,
    batch_size: int,
    feature_dim: int,
) -> np.ndarray:
    output = np.empty((len(rows), feature_dim), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            images = [_load_rgb_image(repo_root, rows[index], index + 1) for index in range(start, end)]
            inputs = image_processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device=device, dtype=torch.float32)
            features = model.get_image_features(pixel_values=pixel_values)
            features = torch.nn.functional.normalize(features.float(), p=2, dim=1)
            output[start:end] = features.detach().cpu().numpy().astype(np.float32, copy=False)
            for image in images:
                image.close()
    _check_feature_array(output, "X_I")
    return output


def _extract_text_features(
    rows: list[dict[str, Any]],
    model: CLIPModel,
    tokenizer: CLIPTokenizerFast,
    device: str,
    batch_size: int,
    feature_dim: int,
) -> np.ndarray:
    output = np.empty((len(rows), feature_dim), dtype=np.float32)
    texts = []
    for index, row in enumerate(rows, start=1):
        text = row.get("text_source")
        if not isinstance(text, str):
            raise RuntimeError(f"manifest row {index} has missing text_source")
        texts.append(text)
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            inputs = tokenizer(
                texts[start:end],
                max_length=77,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            features = model.get_text_features(**inputs)
            features = torch.nn.functional.normalize(features.float(), p=2, dim=1)
            output[start:end] = features.detach().cpu().numpy().astype(np.float32, copy=False)
    _check_feature_array(output, "X_T")
    return output


def _load_rgb_image(repo_root: Path, row: dict[str, Any], row_number: int) -> Image.Image:
    image_path = row.get("image_path")
    if not isinstance(image_path, str) or not image_path:
        raise RuntimeError(f"manifest row {row_number} has missing image_path")
    path = _resolve_repo_path(repo_root, Path(image_path))
    if not path.is_file():
        raise FileNotFoundError(f"manifest image_path missing for row {row_number}: {path}")
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"failed to open image for row {row_number}: {path}") from exc


def _validate_stage2_inputs(
    rows: list[dict[str, Any]],
    query_ids: list[str],
    retrieval_ids: list[str],
    train_ids: list[str],
    manifest_meta: dict[str, Any],
    dataset_config: dict[str, Any],
    dataset: str,
) -> None:
    expected_filtered = int(dataset_config["expected_filtered_count"])
    if len(rows) != expected_filtered:
        raise RuntimeError(f"manifest_filtered count mismatch: expected {expected_filtered}, got {len(rows)}")
    if manifest_meta.get("filtered_count") != expected_filtered:
        raise RuntimeError("manifest_meta.filtered_count does not match Stage 2 config")
    if len(query_ids) != int(dataset_config["expected_query_count"]):
        raise RuntimeError("query_ids count mismatch")
    if len(retrieval_ids) != int(dataset_config["expected_retrieval_count"]):
        raise RuntimeError("retrieval_ids count mismatch")
    if len(train_ids) != int(dataset_config["expected_train_count"]):
        raise RuntimeError("train_ids count mismatch")
    sample_ids = [str(row.get("sample_id")) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("manifest_filtered sample_id is not unique")
    sample_set = set(sample_ids)
    if not set(query_ids).issubset(sample_set):
        raise RuntimeError("query_ids contains ids outside manifest_filtered")
    if not set(retrieval_ids).issubset(sample_set):
        raise RuntimeError("retrieval_ids contains ids outside manifest_filtered")
    if not set(train_ids).issubset(set(retrieval_ids)):
        raise RuntimeError("train_ids is not a subset of retrieval_ids")
    if set(query_ids) & set(retrieval_ids):
        raise RuntimeError("query_ids and retrieval_ids overlap")
    label_dim = int(dataset_config["label_dimension"])
    sample_prefix = f"{dataset_config['sample_id_prefix']}_"
    for index, row in enumerate(rows, start=1):
        if row.get("dataset_name") != dataset:
            raise RuntimeError(f"manifest row {index} has wrong dataset_name: {row.get('dataset_name')}")
        if not isinstance(row.get("sample_id"), str) or not row["sample_id"].startswith(sample_prefix):
            raise RuntimeError(f"manifest row {index} has invalid sample_id")
        if not isinstance(row.get("text_source"), str):
            raise RuntimeError(f"manifest row {index} has missing text_source")
        vector = row.get("label_vector")
        if not isinstance(vector, list) or len(vector) != label_dim or any(value not in (0, 1) for value in vector):
            raise RuntimeError(f"manifest row {index} has invalid label_vector")


def _meta_payload(
    dataset: str,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    query_ids: list[str],
    retrieval_ids: list[str],
    train_ids: list[str],
    order_hashes: dict[str, str],
    model: CLIPModel,
    image_processor: CLIPImageProcessor,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "feature_set_id": config["feature_set_id"],
        "backbone_id": config["clip"]["backbone_id"],
        "model_local_path": config["clip"]["model_local_path"],
        "local_files_only": config["clip"]["local_files_only"],
        "device": config["runtime"]["device"],
        "dtype": config["runtime"]["dtype"],
        "feature_dim": int(model.config.projection_dim),
        "filtered_count": len(rows),
        "query_count": len(query_ids),
        "retrieval_count": len(retrieval_ids),
        "train_count": len(train_ids),
        "image_batch_size": int(config["batches"]["image_batch_size"]),
        "text_batch_size": int(config["batches"]["text_batch_size"]),
        "manifest_filtered_order_sha256": order_hashes["manifest_filtered_order_sha256"],
        "sample_id_order_sha256": order_hashes["sample_id_order_sha256"],
        "query_ids_sha256": order_hashes["query_ids_sha256"],
        "retrieval_ids_sha256": order_hashes["retrieval_ids_sha256"],
        "train_ids_sha256": order_hashes["train_ids_sha256"],
        "image_preprocess_protocol": _image_preprocess_protocol(image_processor),
        "text_tokenizer_protocol": {
            "tokenizer": config["clip"]["backbone_id"],
            "max_length": 77,
            "padding": "max_length",
            "truncation": True,
            "return_attention_mask": True,
            "source": "manifest_filtered.text_source",
            "empty_text_preserved": True,
        },
        "model_eval": not model.training,
        "torch_no_grad": True,
        "amp_enabled": False,
        "silent_fallback_used": False,
        "bad_sample_skip_used": False,
        "zero_vector_padding_used": False,
        "output_dir": str(output_dir),
        "generated_at_utc": _utc_now(),
    }


def _verify_image_processor(image_processor: CLIPImageProcessor) -> None:
    protocol = _image_preprocess_protocol(image_processor)
    if protocol["size"] != {"shortest_edge": 224}:
        raise RuntimeError(f"CLIP image processor size mismatch: {protocol['size']}")
    if protocol["crop_size"] != {"height": 224, "width": 224}:
        raise RuntimeError(f"CLIP image processor crop_size mismatch: {protocol['crop_size']}")
    if protocol["resample"] not in ("3", "Resampling.BICUBIC", "BICUBIC"):
        raise RuntimeError(f"CLIP image processor resample is not bicubic: {protocol['resample']}")
    if protocol["do_resize"] is not True or protocol["do_center_crop"] is not True or protocol["do_normalize"] is not True:
        raise RuntimeError("CLIP image processor resize/crop/normalize flags are not formal")


def _image_preprocess_protocol(image_processor: CLIPImageProcessor) -> dict[str, Any]:
    return {
        "protocol": "RGB -> shortest side resize 224 -> bicubic -> center crop 224x224 -> CLIP mean/std normalize",
        "manual_rgb_convert": True,
        "do_resize": bool(image_processor.do_resize),
        "size": dict(image_processor.size),
        "resample": _resample_name(image_processor.resample),
        "do_center_crop": bool(image_processor.do_center_crop),
        "crop_size": dict(image_processor.crop_size),
        "do_normalize": bool(image_processor.do_normalize),
        "image_mean": [float(value) for value in image_processor.image_mean],
        "image_std": [float(value) for value in image_processor.image_std],
    }


def _resample_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if name:
        return name
    return str(int(value)) if isinstance(value, int) else str(value)


def _check_feature_array(array: np.ndarray, name: str) -> None:
    if array.dtype != np.float32:
        raise RuntimeError(f"{name} dtype must be float32, got {array.dtype}")
    if not np.isfinite(array).all():
        raise RuntimeError(f"{name} contains NaN or Inf")
    norms = np.linalg.norm(array, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
        raise RuntimeError(f"{name} rows are not L2 normalized")


def _stage2_input_paths(processed_root: Path) -> dict[str, Path]:
    return {
        "manifest_filtered": processed_root / "manifest" / "manifest_filtered.jsonl",
        "manifest_meta": processed_root / "manifest" / "manifest_meta.json",
        "query_ids": processed_root / "splits" / "query_ids.txt",
        "retrieval_ids": processed_root / "splits" / "retrieval_ids.txt",
        "train_ids": processed_root / "splits" / "train_ids.txt",
        "order_hashes": processed_root / "reports" / "order_hashes.json",
    }


def _require_stage2_inputs(paths: dict[str, Path]) -> None:
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"required Stage 2 input missing: {name}={path}")


def _read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle if line.rstrip("\n")]


def _resolve_repo_path(repo_root: Path, path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
    resolved.relative_to(repo_root)
    return resolved


def _ensure_within(repo_root: Path, path: Path) -> None:
    path.resolve().relative_to(repo_root)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

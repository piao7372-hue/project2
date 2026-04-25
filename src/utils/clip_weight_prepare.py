from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any


BACKBONE_ID = "openai/clip-vit-base-patch32"
MODEL_LOCAL_REL = Path("models") / "clip" / "openai_clip-vit-base-patch32"
VALIDATOR_VERSION = "stage0c-1_clip_weight_prepare_v1"


class ClipWeightPrepareError(RuntimeError):
    pass


def prepare_stage0_clip_weights(repo_root: Path, config_path: Path, output_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config_full_path = _resolve_repo_path(repo_root, config_path)
    output_root_full = _resolve_repo_path(repo_root, output_root)
    model_local_path = (repo_root / MODEL_LOCAL_REL).resolve()
    _ensure_within_repo(repo_root, config_full_path)
    _ensure_within_repo(repo_root, output_root_full)
    _ensure_within_repo(repo_root, model_local_path)

    generated_at = _utc_now()
    failures: list[str] = []
    online_download_used = False
    model_load_ok = False
    processor_load_ok = False
    model_config_projection_dim = None
    vision_config_hidden_size = None
    text_config_hidden_size = None
    torch_dtype = None
    device_for_validation = "cpu"

    config = _load_json(config_full_path)
    clip_config = dict(config.get("clip", {}))
    if clip_config.get("backbone_id") != BACKBONE_ID:
        failures.append(f"clip.backbone_id must be {BACKBONE_ID!r}, got {clip_config.get('backbone_id')!r}")
    else:
        try:
            config = _set_clip_config(config, allow_online_download=True)
            _write_json(config_full_path, config)
            model_local_path.mkdir(parents=True, exist_ok=True)
            _set_hf_cache_env(model_local_path)
            model_load_ok, processor_load_ok = _validate_local(model_local_path)
            if not (model_load_ok and processor_load_ok):
                online_download_used = True
                _download_and_save(model_local_path)
                model_load_ok, processor_load_ok = _validate_local(model_local_path)
            if not model_load_ok:
                failures.append("CLIPModel local_files_only validation failed")
            if not processor_load_ok:
                failures.append("CLIPProcessor local_files_only validation failed")
            if model_load_ok:
                dims = _read_model_config(model_local_path)
                model_config_projection_dim = dims["model_config_projection_dim"]
                vision_config_hidden_size = dims["vision_config_hidden_size"]
                text_config_hidden_size = dims["text_config_hidden_size"]
                torch_dtype = dims["torch_dtype"]
                device_for_validation = dims["device_for_validation"]
        except Exception as exc:
            failures.append(repr(exc))
        finally:
            final_config = _load_json(config_full_path)
            final_config = _set_clip_config(final_config, allow_online_download=False)
            _write_json(config_full_path, final_config)

    output_root_full.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": "stage0",
        "substage": "stage0c-1",
        "validator_version": VALIDATOR_VERSION,
        "generated_at_utc": generated_at,
        "backbone_id": BACKBONE_ID,
        "model_local_path": str(MODEL_LOCAL_REL).replace("\\", "/"),
        "model_local_abs_path": str(model_local_path),
        "local_files_only": True,
        "use_safetensors": True,
        "model_load_ok": model_load_ok,
        "processor_load_ok": processor_load_ok,
        "model_config_projection_dim": model_config_projection_dim,
        "vision_config_hidden_size": vision_config_hidden_size,
        "text_config_hidden_size": text_config_hidden_size,
        "torch_dtype": torch_dtype,
        "device_for_validation": device_for_validation,
        "online_download_used": online_download_used,
        "allow_online_download_after_run": _load_json(config_full_path)["clip"].get("allow_online_download"),
        "failure_reason": failures,
        "feature_extraction_performed": False,
        "stage1_entered": False,
        "forbidden_model_switch_used": False,
        "silent_fallback_used": False,
    }
    summary_json = output_root_full / "clip_prepare_summary.json"
    summary_md = output_root_full / "clip_prepare_summary.md"
    _write_json(summary_json, summary)
    summary_md.write_text(_render_markdown(summary), encoding="utf-8")
    summary["summary_json_path"] = str(summary_json)
    summary["summary_markdown_path"] = str(summary_md)
    return summary


def _download_and_save(model_local_path: Path) -> None:
    from transformers import CLIPModel, CLIPProcessor

    cache_dir = model_local_path / "_download_cache"
    model = CLIPModel.from_pretrained(BACKBONE_ID, cache_dir=str(cache_dir), local_files_only=False, use_safetensors=True)
    processor = CLIPProcessor.from_pretrained(BACKBONE_ID, cache_dir=str(cache_dir), local_files_only=False)
    model.save_pretrained(model_local_path)
    processor.save_pretrained(model_local_path)


def _validate_local(model_local_path: Path) -> tuple[bool, bool]:
    from transformers import CLIPModel, CLIPProcessor

    model_ok = False
    processor_ok = False
    try:
        CLIPModel.from_pretrained(model_local_path, local_files_only=True, use_safetensors=True)
        model_ok = True
    except Exception:
        model_ok = False
    try:
        CLIPProcessor.from_pretrained(model_local_path, local_files_only=True)
        processor_ok = True
    except Exception:
        processor_ok = False
    return model_ok, processor_ok


def _read_model_config(model_local_path: Path) -> dict[str, Any]:
    from transformers import CLIPModel

    model = CLIPModel.from_pretrained(model_local_path, local_files_only=True, use_safetensors=True)
    first_param = next(model.parameters())
    return {
        "model_config_projection_dim": getattr(model.config, "projection_dim", None),
        "vision_config_hidden_size": getattr(model.config.vision_config, "hidden_size", None),
        "text_config_hidden_size": getattr(model.config.text_config, "hidden_size", None),
        "torch_dtype": str(first_param.dtype),
        "device_for_validation": str(first_param.device),
    }


def _set_hf_cache_env(model_local_path: Path) -> None:
    cache_root = model_local_path / "_hf_home"
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "transformers")


def _set_clip_config(config: dict[str, Any], allow_online_download: bool) -> dict[str, Any]:
    clip = dict(config.get("clip", {}))
    clip["backbone_id"] = BACKBONE_ID
    clip["model_local_path"] = str(MODEL_LOCAL_REL).replace("\\", "/")
    clip["allow_online_download"] = allow_online_download
    config["clip"] = clip
    return config


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Stage 0 CLIP Prepare Summary",
        "",
        f"- Backbone: `{summary['backbone_id']}`",
        f"- Model local path: `{summary['model_local_path']}`",
        f"- Local files only validation: `{str(summary['local_files_only']).lower()}`",
        f"- use_safetensors: `{str(summary['use_safetensors']).lower()}`",
        f"- Model load OK: `{str(summary['model_load_ok']).lower()}`",
        f"- Processor load OK: `{str(summary['processor_load_ok']).lower()}`",
        f"- Online download used: `{str(summary['online_download_used']).lower()}`",
        f"- allow_online_download after run: `{str(summary['allow_online_download_after_run']).lower()}`",
        f"- Projection dim: `{summary['model_config_projection_dim']}`",
        f"- Vision hidden size: `{summary['vision_config_hidden_size']}`",
        f"- Text hidden size: `{summary['text_config_hidden_size']}`",
        f"- Torch dtype: `{summary['torch_dtype']}`",
        f"- Validation device: `{summary['device_for_validation']}`",
        "",
        "## Failures",
        "",
    ]
    if summary["failure_reason"]:
        for failure in summary["failure_reason"]:
            lines.append(f"- {failure}")
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _resolve_repo_path(repo_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _ensure_within_repo(repo_root: Path, path: Path) -> None:
    path.relative_to(repo_root)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

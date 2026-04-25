from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


REQUIRED_CONFIG_FIELDS = (
    ("runtime", "formal_python_path"),
    ("runtime", "require_formal_python"),
    ("runtime", "allow_codex_runtime_for_stage0a_only"),
    ("clip", "backbone_id"),
    ("clip", "model_local_path"),
    ("clip", "allow_online_download"),
)

PACKAGE_SPECS = {
    "torch": {"dists": ("torch",), "module": "torch"},
    "torchvision": {"dists": ("torchvision",), "module": "torchvision"},
    "transformers": {"dists": ("transformers",), "module": "transformers"},
    "faiss": {"dists": ("faiss-cpu", "faiss-gpu", "faiss"), "module": "faiss"},
    "numpy": {"dists": ("numpy",), "module": "numpy"},
    "pillow": {"dists": ("pillow",), "module": "PIL"},
    "scipy": {"dists": ("scipy",), "module": "scipy"},
}

FORMAL_CLIP_BACKBONE = "openai/clip-vit-base-patch32"


@dataclass(frozen=True)
class EnvironmentLockResult:
    lock_json_path: Path
    lock_markdown_path: Path
    stage0a_ready: bool
    formal_dependency_ready: bool
    missing_packages: list[str]


def run_environment_lock(repo_root: Path, config_path: Path) -> EnvironmentLockResult:
    repo_root = repo_root.resolve()
    config_full_path = (repo_root / config_path).resolve()
    config = _load_json(config_full_path)
    _validate_config(config)

    runtime_check = _validate_runtime(config)
    lock = _build_lock(repo_root, config_full_path, config, runtime_check)

    lock_json_path = (repo_root / config["outputs"]["lock_json_path"]).resolve()
    lock_markdown_path = (repo_root / config["outputs"]["lock_markdown_path"]).resolve()
    _ensure_within_repo(repo_root, lock_json_path)
    _ensure_within_repo(repo_root, lock_markdown_path)

    lock_json_path.parent.mkdir(parents=True, exist_ok=True)
    lock_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    lock_json_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lock_markdown_path.write_text(_render_markdown(lock), encoding="utf-8")

    missing_packages = [
        name for name, entry in lock["packages"].items() if not entry["installed"]
    ]
    return EnvironmentLockResult(
        lock_json_path=lock_json_path,
        lock_markdown_path=lock_markdown_path,
        stage0a_ready=bool(lock["checks"]["stage0a_ready"]),
        formal_dependency_ready=bool(lock["checks"]["formal_dependency_ready"]),
        missing_packages=missing_packages,
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_config(config: dict[str, Any]) -> None:
    missing = [
        ".".join(field_path)
        for field_path in REQUIRED_CONFIG_FIELDS
        if not _has_nested_key(config, field_path)
    ]
    if missing:
        raise ValueError("Missing Stage 0 config fields: " + ", ".join(missing))

    if config["clip"]["backbone_id"] != FORMAL_CLIP_BACKBONE:
        raise ValueError(
            "Invalid CLIP backbone. Expected "
            f"{FORMAL_CLIP_BACKBONE}, got {config['clip']['backbone_id']}"
        )

    bool_fields = (
        ("runtime", "require_formal_python"),
        ("runtime", "allow_codex_runtime_for_stage0a_only"),
        ("clip", "allow_online_download"),
        ("execution_policy", "fail_fast"),
        ("execution_policy", "allow_silent_fallback"),
        ("execution_policy", "allow_bad_sample_skip"),
        ("execution_policy", "allow_zero_vector_padding"),
        ("execution_policy", "allow_model_switch"),
        ("execution_policy", "allow_git_commit"),
        ("execution_policy", "allow_git_push"),
    )
    for field_path in bool_fields:
        if _has_nested_key(config, field_path) and not isinstance(
            _get_nested(config, field_path), bool
        ):
            raise TypeError("Expected boolean config field: " + ".".join(field_path))

    forbidden_true = (
        ("execution_policy", "allow_silent_fallback"),
        ("execution_policy", "allow_bad_sample_skip"),
        ("execution_policy", "allow_zero_vector_padding"),
        ("execution_policy", "allow_model_switch"),
        ("execution_policy", "allow_git_commit"),
        ("execution_policy", "allow_git_push"),
    )
    enabled_forbidden = [
        ".".join(field_path)
        for field_path in forbidden_true
        if _has_nested_key(config, field_path) and _get_nested(config, field_path)
    ]
    if enabled_forbidden:
        raise ValueError("Forbidden execution policy enabled: " + ", ".join(enabled_forbidden))


def _validate_runtime(config: dict[str, Any]) -> dict[str, Any]:
    formal_path = Path(config["runtime"]["formal_python_path"]).resolve()
    current_path = Path(sys.executable).resolve()
    formal_exists = formal_path.exists()
    running_formal = _same_path(formal_path, current_path)
    codex_exception_allowed = bool(config["runtime"]["allow_codex_runtime_for_stage0a_only"])
    require_formal = bool(config["runtime"]["require_formal_python"])
    codex_runtime = ".cache" in str(current_path).lower() and "codex-runtimes" in str(
        current_path
    ).lower()
    codex_exception_used = (
        require_formal
        and not running_formal
        and codex_exception_allowed
        and config.get("substage") == "stage0a"
        and codex_runtime
    )

    if not formal_exists:
        raise FileNotFoundError(f"Formal Python path does not exist: {formal_path}")
    if require_formal and not running_formal and not codex_exception_used:
        raise RuntimeError(
            "Stage 0 config requires the formal Python runtime. "
            f"current={current_path}; formal={formal_path}"
        )

    return {
        "formal_python_path": str(formal_path),
        "current_python_path": str(current_path),
        "formal_python_path_exists": formal_exists,
        "running_with_formal_python": running_formal,
        "codex_stage0a_exception_used": codex_exception_used,
    }


def _build_lock(
    repo_root: Path,
    config_path: Path,
    config: dict[str, Any],
    runtime_check: dict[str, Any],
) -> dict[str, Any]:
    packages = _collect_packages()
    missing_packages = [name for name, entry in packages.items() if not entry["installed"]]
    clip_local_path = config["clip"]["model_local_path"]
    clip_local_exists = bool(clip_local_path) and Path(clip_local_path).exists()
    formal_dependency_ready = not missing_packages

    lock = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "stage": config["stage"],
        "substage": config["substage"],
        "repo_root": str(repo_root),
        "formal_spec": _file_identity(Path(config["formal_spec_path"])),
        "config": {
            "path": str(config_path),
            "sha256": _sha256_file(config_path),
            "runtime": config["runtime"],
            "clip": config["clip"],
            "execution_policy": config.get("execution_policy", {}),
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "version_info": list(sys.version_info[:5]),
            "implementation": platform.python_implementation(),
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
        },
        "os": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "packages": packages,
        "cuda": _collect_cuda(),
        "nvidia_smi": _collect_nvidia_smi(),
        "checks": {
            **runtime_check,
            "clip_backbone_locked": config["clip"]["backbone_id"] == FORMAL_CLIP_BACKBONE,
            "clip_online_download_allowed": bool(config["clip"]["allow_online_download"]),
            "clip_model_local_path_exists": clip_local_exists,
            "formal_dependency_ready": formal_dependency_ready,
            "stage0a_ready": (
                runtime_check["formal_python_path_exists"]
                and (
                    runtime_check["running_with_formal_python"]
                    or runtime_check["codex_stage0a_exception_used"]
                )
                and config["clip"]["backbone_id"] == FORMAL_CLIP_BACKBONE
            ),
        },
        "notes": {
            "clip_download_attempted": False,
            "raw_data_download_attempted": False,
            "nus_audit_attempted": False,
            "stage1_or_later_code_created": False,
            "missing_packages": missing_packages,
        },
    }
    return lock


def _collect_packages() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, spec in PACKAGE_SPECS.items():
        version = _first_distribution_version(spec["dists"])
        import_status = _module_import_status(spec["module"])
        result[name] = {
            "installed": version is not None or import_status["importable"],
            "version": version,
            "module": spec["module"],
            "module_importable": import_status["importable"],
            "import_error": import_status["error"],
        }
    return result


def _collect_cuda() -> dict[str, Any]:
    status: dict[str, Any] = {
        "torch_importable": False,
        "torch_cuda_version": None,
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "error": None,
    }
    try:
        torch = importlib.import_module("torch")
        status["torch_importable"] = True
        status["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        status["cuda_available"] = bool(torch.cuda.is_available())
        status["device_count"] = int(torch.cuda.device_count())
        devices = []
        for index in range(status["device_count"]):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                }
            )
        status["devices"] = devices
    except Exception as exc:
        status["error"] = repr(exc)
    return status


def _collect_nvidia_smi() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return {"available": False, "error": "nvidia-smi not found", "gpus": []}
    except subprocess.TimeoutExpired:
        return {"available": False, "error": "nvidia-smi timed out", "gpus": []}

    if completed.returncode != 0:
        return {
            "available": False,
            "error": completed.stderr.strip(),
            "gpus": [],
        }

    gpus = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            gpus.append(
                {
                    "name": parts[0],
                    "memory_total_mib": parts[1],
                    "driver_version": parts[2],
                }
            )
    return {"available": True, "error": None, "gpus": gpus}


def _render_markdown(lock: dict[str, Any]) -> str:
    packages = lock["packages"]
    package_rows = [
        "| Package | Installed | Version | Importable |",
        "| --- | --- | --- | --- |",
    ]
    for name in PACKAGE_SPECS:
        entry = packages[name]
        package_rows.append(
            "| {name} | {installed} | {version} | {importable} |".format(
                name=name,
                installed=str(entry["installed"]).lower(),
                version=entry["version"] or "",
                importable=str(entry["module_importable"]).lower(),
            )
        )

    cuda = lock["cuda"]
    devices = cuda["devices"]
    if devices:
        gpu_lines = [
            f"- cuda:{device['index']} {device['name']} "
            f"({device['total_memory_bytes']} bytes)"
            for device in devices
        ]
    else:
        gpu_lines = ["- No CUDA device reported by torch."]

    missing = lock["notes"]["missing_packages"]
    missing_text = ", ".join(missing) if missing else "none"

    lines = [
        "# Environment Lock",
        "",
        f"- Generated UTC: `{lock['generated_at_utc']}`",
        f"- Stage: `{lock['stage']}`",
        f"- Substage: `{lock['substage']}`",
        f"- Formal spec path: `{lock['formal_spec']['path']}`",
        f"- Formal spec sha256: `{lock['formal_spec']['sha256']}`",
        f"- Config path: `{lock['config']['path']}`",
        f"- Config sha256: `{lock['config']['sha256']}`",
        "",
        "## Runtime",
        "",
        f"- Formal Python: `{lock['checks']['formal_python_path']}`",
        f"- Current Python: `{lock['checks']['current_python_path']}`",
        f"- Running with formal Python: `{str(lock['checks']['running_with_formal_python']).lower()}`",
        f"- Stage 0A Codex runtime exception used: `{str(lock['checks']['codex_stage0a_exception_used']).lower()}`",
        f"- Python version: `{lock['python']['version'].splitlines()[0]}`",
        "",
        "## Operating System",
        "",
        f"- Platform: `{lock['os']['platform']}`",
        f"- Machine: `{lock['os']['machine']}`",
        "",
        "## Packages",
        "",
        *package_rows,
        "",
        "## CUDA",
        "",
        f"- Torch CUDA version: `{cuda['torch_cuda_version']}`",
        f"- CUDA available: `{str(cuda['cuda_available']).lower()}`",
        f"- Device count: `{cuda['device_count']}`",
        *gpu_lines,
        "",
        "## CLIP",
        "",
        f"- Backbone: `{lock['config']['clip']['backbone_id']}`",
        f"- Local path: `{lock['config']['clip']['model_local_path']}`",
        f"- Local path exists: `{str(lock['checks']['clip_model_local_path_exists']).lower()}`",
        f"- Online download allowed: `{str(lock['checks']['clip_online_download_allowed']).lower()}`",
        f"- Download attempted: `{str(lock['notes']['clip_download_attempted']).lower()}`",
        "",
        "## Stage 0A Result",
        "",
        f"- Stage 0A ready: `{str(lock['checks']['stage0a_ready']).lower()}`",
        f"- Formal dependency ready: `{str(lock['checks']['formal_dependency_ready']).lower()}`",
        f"- Missing packages: `{missing_text}`",
        f"- Raw data download attempted: `{str(lock['notes']['raw_data_download_attempted']).lower()}`",
        f"- NUS audit attempted: `{str(lock['notes']['nus_audit_attempted']).lower()}`",
        f"- Stage 1 or later code created: `{str(lock['notes']['stage1_or_later_code_created']).lower()}`",
        "",
    ]
    return "\n".join(lines)


def _first_distribution_version(names: tuple[str, ...]) -> str | None:
    for name in names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def _module_import_status(module_name: str) -> dict[str, Any]:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        return {"importable": False, "error": repr(exc)}
    return {"importable": True, "error": None}


def _file_identity(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing formal spec file: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _has_nested_key(data: dict[str, Any], field_path: tuple[str, ...]) -> bool:
    current: Any = data
    for part in field_path:
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True


def _get_nested(data: dict[str, Any], field_path: tuple[str, ...]) -> Any:
    current: Any = data
    for part in field_path:
        current = current[part]
    return current


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left)) == os.path.normcase(str(right))


def _ensure_within_repo(repo_root: Path, target: Path) -> None:
    target.relative_to(repo_root)

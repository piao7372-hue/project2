from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any


SPEC_DOCUMENT = "kan_engineering_original_RA_NUS.pdf"
NUS_PROTOCOL = "original_ra_nus_image_index_v1"
CLIP_BACKBONE = "openai/clip-vit-base-patch32"
CLIP_LOCAL_PATH = "models/clip/openai_clip-vit-base-patch32"
STAGE1_ARTIFACT_NAMES = {
    "manifest_raw.jsonl",
    "manifest_filtered.jsonl",
    "manifest_meta.json",
    "query_ids.txt",
    "retrieval_ids.txt",
    "train_ids.txt",
    "X_I.npy",
    "X_T.npy",
    "A.npy",
    "R.npy",
    "Se.npy",
    "C.npy",
    "S.npy",
}
IGNORE_REQUIRED = [
    "data/raw/probe.file",
    "data/processed/probe.file",
    "outputs/probe.file",
    "models/clip/probe.file",
    "probe.npy",
    "probe.npz",
    "probe.pt",
    "probe.pth",
    "probe.ckpt",
    "cache/probe.file",
    "logs/probe.file",
    "checkpoint/probe.file",
    "checkpoints/probe.file",
    "ckpts/probe.file",
]
SOURCE_NOT_IGNORED = [
    "scripts/validate_stage0_raw.py",
    "src/datasets/validators/raw_validator.py",
    "scripts/prepare_stage0_clip.py",
    "src/utils/clip_weight_prepare.py",
]


def run_stage0_final_gate(repo_root: Path, output_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_root_full = _resolve_repo_path(repo_root, output_root)
    _ensure_within_repo(repo_root, output_root_full)
    generated_at = _utc_now()

    spec = _check_spec(repo_root)
    env = _check_environment_lock(repo_root)
    raw = _check_raw_validator(repo_root)
    clip = _check_clip(repo_root)
    git_gate = _check_git_ignore(repo_root)
    absence = _check_stage1_absence(repo_root)

    blocker_reason: list[str] = []
    for label, check in (
        ("environment_lock", env),
        ("raw_validator", raw),
        ("clip_validation", clip),
        ("engineering_spec", spec),
        ("git_ignore", git_gate),
        ("stage1_artifact_absence", absence),
    ):
        if not check["passed"]:
            blocker_reason.append(f"{label}: {check.get('failure_reason', [])}")

    stage0_complete = not blocker_reason
    summary = {
        "stage": "stage0",
        "substage": "stage0f-1",
        "generated_at_utc": generated_at,
        "stage0_environment_lock_passed": env["passed"],
        "stage0_raw_validator_passed": raw["passed"],
        "stage0_clip_validation_passed": clip["passed"],
        "engineering_spec_compliance_passed": spec["passed"],
        "git_ignore_gate_passed": git_gate["passed"],
        "stage1_artifact_absence_passed": absence["passed"],
        "faiss_missing": env["faiss_missing"],
        "faiss_status": env["faiss_status"],
        "stage0_complete": stage0_complete,
        "stage1_allowed": False,
        "stage1_allowed_reason": "Stage 1 requires explicit user authorization",
        "blocker_reason": blocker_reason,
        "engineering_spec_compliance": spec,
        "environment_lock": env,
        "raw_validator": raw,
        "clip_validation": clip,
        "git_ignore_gate": git_gate,
        "stage1_artifact_absence": absence,
        "stage1_entered": False,
        "stage2_entered": False,
    }

    output_root_full.mkdir(parents=True, exist_ok=True)
    summary_path = output_root_full / "stage0_final_gate_summary.json"
    md_path = output_root_full / "stage0_final_gate_summary.md"
    compliance_path = output_root_full / "engineering_spec_compliance.json"
    _write_json(summary_path, summary)
    _write_json(compliance_path, spec)
    md_path.write_text(_render_markdown(summary), encoding="utf-8")
    summary["summary_json_path"] = str(summary_path)
    summary["summary_markdown_path"] = str(md_path)
    summary["engineering_spec_compliance_path"] = str(compliance_path)
    return summary


def _check_spec(repo_root: Path) -> dict[str, Any]:
    failures: list[str] = []
    stage0 = _load_json(repo_root / "configs" / "stages" / "stage0_formal.json")
    roots = _load_json(repo_root / "configs" / "datasets" / "raw_roots.json")
    status_path = repo_root / "docs" / "project_status.md"
    status_text = status_path.read_text(encoding="utf-8", errors="replace") if status_path.exists() else ""
    engineering_spec = stage0.get("engineering_spec", {})
    spec_document = engineering_spec.get("document")
    nus_protocol = engineering_spec.get("nus_protocol")
    root_protocol = roots.get("nuswide", {}).get("source_protocol")
    old_kaggle_protocol_used_as_formal = (
        nus_protocol == "kaggle_top10_formal_v3" or root_protocol == "kaggle_top10_formal_v3"
    )
    if spec_document != SPEC_DOCUMENT:
        failures.append(f"spec document mismatch: {spec_document!r}")
    if nus_protocol != NUS_PROTOCOL:
        failures.append(f"engineering_spec.nus_protocol mismatch: {nus_protocol!r}")
    if root_protocol != NUS_PROTOCOL:
        failures.append(f"raw_roots nuswide.source_protocol mismatch: {root_protocol!r}")
    if old_kaggle_protocol_used_as_formal:
        failures.append("kaggle_top10_formal_v3 is still used as formal protocol")
    if "Do not enter Stage 1" not in status_text and "Stage 1" not in status_text:
        failures.append("project_status.md does not mention Stage 1 gate")
    return {
        "passed": not failures,
        "failure_reason": failures,
        "spec_document": spec_document,
        "nus_protocol": nus_protocol,
        "raw_roots_nus_protocol": root_protocol,
        "old_kaggle_protocol_used_as_formal": old_kaggle_protocol_used_as_formal,
        "stage": "Stage 0",
        "stage1_forbidden": True,
        "stage2_forbidden": True,
        "compliance_passed": not failures,
    }


def _check_environment_lock(repo_root: Path) -> dict[str, Any]:
    lock_md = repo_root / "docs" / "environment_lock.md"
    lock_json = repo_root / "outputs" / "env" / "environment.lock.json"
    failures: list[str] = []
    if not lock_md.exists():
        failures.append(f"missing {lock_md}")
    if not lock_json.exists():
        failures.append(f"missing {lock_json}")
        lock = {}
    else:
        try:
            lock = _load_json(lock_json)
        except Exception as exc:
            failures.append(f"environment.lock.json unreadable: {exc!r}")
            lock = {}
    packages = lock.get("packages", {})
    cuda = lock.get("cuda", {})
    nvidia = lock.get("nvidia_smi", {})
    faiss = packages.get("faiss", {})
    faiss_missing = not bool(faiss.get("installed"))
    required_packages = ["torch", "torchvision", "transformers", "numpy", "pillow", "scipy"]
    for name in required_packages:
        if not packages.get(name, {}).get("installed"):
            failures.append(f"required package missing in environment lock: {name}")
    return {
        "passed": not failures,
        "failure_reason": failures,
        "environment_lock_md_exists": lock_md.exists(),
        "environment_lock_json_exists": lock_json.exists(),
        "python_version": lock.get("python", {}).get("version"),
        "torch_version": packages.get("torch", {}).get("version"),
        "torchvision_version": packages.get("torchvision", {}).get("version"),
        "transformers_version": packages.get("transformers", {}).get("version"),
        "numpy_version": packages.get("numpy", {}).get("version"),
        "pillow_version": packages.get("pillow", {}).get("version"),
        "scipy_version": packages.get("scipy", {}).get("version"),
        "cuda_available": cuda.get("cuda_available"),
        "gpu_name": _gpu_name(cuda, nvidia),
        "faiss_status": faiss,
        "faiss_missing": faiss_missing,
        "faiss_missing_is_stage0_blocker": False,
        "faiss_note": "faiss is not installed and was not installed; this is not a Stage 0 final gate blocker for raw/CLIP validation.",
    }


def _check_raw_validator(repo_root: Path) -> dict[str, Any]:
    failures: list[str] = []
    paths = [
        repo_root / "data" / "raw" / "mirflickr25k" / "raw_audit.json",
        repo_root / "data" / "raw" / "mirflickr25k" / "raw_validator_summary.json",
        repo_root / "data" / "raw" / "mscoco" / "raw_audit.json",
        repo_root / "data" / "raw" / "mscoco" / "raw_validator_summary.json",
        repo_root / "data" / "raw" / "nuswide" / "raw_audit.json",
        repo_root / "data" / "raw" / "nuswide" / "raw_validator_summary.json",
        repo_root / "outputs" / "stage0_raw_validator" / "raw_validator_summary.json",
    ]
    exists = {str(path): path.exists() for path in paths}
    for path in paths:
        if not path.exists():
            failures.append(f"missing raw validator artifact: {path}")
    summary = {}
    if (repo_root / "outputs" / "stage0_raw_validator" / "raw_validator_summary.json").exists():
        summary = _load_json(repo_root / "outputs" / "stage0_raw_validator" / "raw_validator_summary.json")
        if summary.get("all_raw_validators_passed") is not True:
            failures.append("all_raw_validators_passed is not true")
        results = summary.get("dataset_results", {})
        for dataset in ("mirflickr25k", "mscoco", "nuswide"):
            if results.get(dataset, {}).get("passed") is not True:
                failures.append(f"{dataset} raw validator did not pass")
        nus = results.get("nuswide", {})
        if nus.get("source_protocol") != NUS_PROTOCOL and nus.get("nus_source_protocol") != NUS_PROTOCOL:
            failures.append(f"NUS raw validator protocol mismatch: {nus.get('source_protocol')!r} / {nus.get('nus_source_protocol')!r}")
        if nus.get("uses_img_tc10_as_formal_input") is not False:
            failures.append("NUS uses_img_tc10_as_formal_input is not false")
        if nus.get("uses_targets_onehot_tc10_as_formal_input") is not False:
            failures.append("NUS uses_targets_onehot_tc10_as_formal_input is not false")
        if nus.get("uses_database_test_split_as_formal_split") is not False:
            failures.append("NUS uses_database_test_split_as_formal_split is not false")
    return {
        "passed": not failures,
        "failure_reason": failures,
        "artifact_exists": exists,
        "all_raw_validators_passed": summary.get("all_raw_validators_passed"),
        "dataset_results": summary.get("dataset_results", {}),
    }


def _check_clip(repo_root: Path) -> dict[str, Any]:
    failures: list[str] = []
    model_dir = repo_root / CLIP_LOCAL_PATH
    summary_path = repo_root / "outputs" / "stage0_clip" / "clip_prepare_summary.json"
    stage0 = _load_json(repo_root / "configs" / "stages" / "stage0_formal.json")
    clip_config = stage0.get("clip", {})
    if not model_dir.exists():
        failures.append(f"missing CLIP model directory: {model_dir}")
    if not summary_path.exists():
        failures.append(f"missing CLIP prepare summary: {summary_path}")
        summary = {}
    else:
        summary = _load_json(summary_path)
    expected_config = {
        "backbone_id": CLIP_BACKBONE,
        "model_local_path": CLIP_LOCAL_PATH,
        "allow_online_download": False,
    }
    for key, expected in expected_config.items():
        if clip_config.get(key) != expected:
            failures.append(f"stage0_formal clip.{key} mismatch: expected {expected!r}, got {clip_config.get(key)!r}")
    checks = {
        "backbone_id": summary.get("backbone_id") == CLIP_BACKBONE,
        "model_local_path": summary.get("model_local_path") == CLIP_LOCAL_PATH,
        "local_files_only": summary.get("local_files_only") is True,
        "model_load_ok": summary.get("model_load_ok") is True,
        "processor_load_ok": summary.get("processor_load_ok") is True,
        "model_config_projection_dim": summary.get("model_config_projection_dim") == 512,
        "allow_online_download_after_run": summary.get("allow_online_download_after_run") is False,
    }
    for name, ok in checks.items():
        if not ok:
            failures.append(f"CLIP validation check failed: {name}")
    return {
        "passed": not failures,
        "failure_reason": failures,
        "model_dir_exists": model_dir.exists(),
        "summary_exists": summary_path.exists(),
        "clip_config": clip_config,
        "summary": summary,
    }


def _check_git_ignore(repo_root: Path) -> dict[str, Any]:
    failures: list[str] = []
    ignored = _run_git(repo_root, ["check-ignore", *IGNORE_REQUIRED])
    ignored_paths = {line.strip().replace("\\", "/") for line in ignored["stdout"].splitlines() if line.strip()}
    for path in IGNORE_REQUIRED:
        if path not in ignored_paths:
            failures.append(f"required ignored path is not ignored: {path}")
    source_results: dict[str, Any] = {}
    for path in SOURCE_NOT_IGNORED:
        result = _run_git(repo_root, ["check-ignore", "-v", path], allow_failure=True)
        source_results[path] = {"stdout": result["stdout"], "returncode": result["returncode"]}
        if result["stdout"].strip():
            failures.append(f"source file is incorrectly ignored: {path}: {result['stdout'].strip()}")
    return {
        "passed": not failures,
        "failure_reason": failures,
        "required_ignore_stdout": ignored["stdout"],
        "required_ignore_stderr": ignored["stderr"],
        "source_not_ignored_results": source_results,
    }


def _check_stage1_absence(repo_root: Path) -> dict[str, Any]:
    failures: list[str] = []
    data_processed = repo_root / "data" / "processed"
    if data_processed.exists():
        failures.append(f"data/processed exists: {data_processed}")
    found: list[str] = []
    for path in repo_root.rglob("*"):
        if path.is_file() and path.name in STAGE1_ARTIFACT_NAMES:
            found.append(str(path))
    if found:
        failures.append(f"Stage 1+ artifacts found: {found}")
    return {
        "passed": not failures,
        "failure_reason": failures,
        "data_processed_exists": data_processed.exists(),
        "stage1_artifacts_found": found,
    }


def _run_git(repo_root: Path, args: list[str], allow_failure: bool = False) -> dict[str, Any]:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0 and not allow_failure:
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    return {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


def _gpu_name(cuda: dict[str, Any], nvidia: dict[str, Any]) -> str | None:
    devices = cuda.get("devices") or []
    if devices:
        return devices[0].get("name")
    gpus = nvidia.get("gpus") or []
    if gpus:
        return gpus[0].get("name")
    return None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Stage 0 Final Gate Summary",
        "",
        f"- Stage 0 environment lock passed: `{str(summary['stage0_environment_lock_passed']).lower()}`",
        f"- Stage 0 raw validator passed: `{str(summary['stage0_raw_validator_passed']).lower()}`",
        f"- Stage 0 CLIP validation passed: `{str(summary['stage0_clip_validation_passed']).lower()}`",
        f"- Engineering spec compliance passed: `{str(summary['engineering_spec_compliance_passed']).lower()}`",
        f"- Git ignore gate passed: `{str(summary['git_ignore_gate_passed']).lower()}`",
        f"- Stage 1 artifact absence passed: `{str(summary['stage1_artifact_absence_passed']).lower()}`",
        f"- faiss missing: `{str(summary['faiss_missing']).lower()}`",
        f"- Stage 0 complete: `{str(summary['stage0_complete']).lower()}`",
        f"- Stage 1 allowed: `{str(summary['stage1_allowed']).lower()}`",
        f"- Stage 1 allowed reason: `{summary['stage1_allowed_reason']}`",
        "",
        "## Blockers",
        "",
    ]
    if summary["blocker_reason"]:
        for blocker in summary["blocker_reason"]:
            lines.append(f"- {blocker}")
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

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.wrappers.cross_modal_hash_net import CrossModalHashNet
from src.utils.jsonl import iter_jsonl, read_json, write_json


FORMAL_STAGE4_CONFIG = REPO_ROOT / "configs" / "stages" / "stage4_model.json"
SUPPORTED_STAGE4_FORWARD_DATASETS = {"mirflickr25k", "nuswide", "mscoco"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 4 untrained forward sanity.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--all-bits", action="store_true")
    parser.add_argument("--config", default="configs/stages/stage4_model.json")
    return parser.parse_args()


def enforce_formal_python() -> None:
    config = read_json(FORMAL_STAGE4_CONFIG)
    expected = Path(config["runtime"]["python"]).resolve()
    current = Path(sys.executable).resolve()
    if os.path.normcase(str(expected)) != os.path.normcase(str(current)):
        raise RuntimeError(f"Stage 4 requires formal Python: current={current}; expected={expected}")


def main() -> int:
    enforce_formal_python()
    args = parse_args()
    summary = run_stage4_forward(REPO_ROOT, Path(args.config), args.dataset, args.all_bits)
    print(f"dataset={summary['dataset']}")
    print(f"cache_id={summary['cache_id']}")
    print(f"hash_bits={summary['hash_bits']}")
    for bit, bit_summary in summary["bits"].items():
        print(f"bit={bit}")
        print(f"output_dir={bit_summary['output_dir']}")
        print(f"H_I_shape={bit_summary['H_I_shape']}")
        print(f"H_T_shape={bit_summary['H_T_shape']}")
        print(f"B_I_shape={bit_summary['B_I_shape']}")
        print(f"B_T_shape={bit_summary['B_T_shape']}")
        print(f"H_I_range={bit_summary['H_I_range']}")
        print(f"H_T_range={bit_summary['H_T_range']}")
        print(f"B_unique={bit_summary['B_unique']}")
    print("stage4_forward_completed=true")
    return 0


def run_stage4_forward(repo_root: Path, config_path: Path, dataset: str, all_bits: bool) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = read_json(_resolve_repo_path(repo_root, config_path))
    _check_stage4_forward_dataset(dataset, config)
    _check_config(config)
    if not all_bits:
        raise RuntimeError("Stage 4 forward requires --all-bits")
    _check_runtime(config)

    dataset_config = config["datasets"][dataset]
    processed_root = _resolve_repo_path(repo_root, Path(config["inputs"]["processed_root"])) / dataset
    paths = _stage4_paths(processed_root, config)
    context = _load_context(paths, config, dataset_config, dataset)
    X_I_train, X_T_train = _load_train_features(paths, dataset_config, context["train_indices"])

    device = torch.device(str(config["runtime"]["device"]))
    bits = [int(value) for value in config["hash_bits"]]
    bit_summaries: dict[str, Any] = {}
    for bit in bits:
        torch.manual_seed(int(config["runtime"]["seed"]))
        torch.cuda.manual_seed_all(int(config["runtime"]["seed"]))
        model = CrossModalHashNet(
            input_dim=int(dataset_config["input_feature_dim"]),
            d_z=int(dataset_config["d_z"]),
            bit=bit,
            cheby_order=int(dataset_config["cheby_order"]),
            tree_prototypes=[int(value) for value in dataset_config["tree_prototypes"]],
            graph_k=int(dataset_config["graph_k_train"]),
            beta_tree_injection=float(dataset_config["beta_tree_injection"]),
        ).to(device)
        model.eval()
        X_I_tensor = torch.from_numpy(X_I_train).to(device=device, dtype=torch.float32)
        X_T_tensor = torch.from_numpy(X_T_train).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            output = model(X_I_tensor, X_T_tensor, bit=bit)
        bit_dir = paths["cache_root"] / str(bit)
        bit_dir.mkdir(parents=True, exist_ok=True)
        H_I = output["H_I"].detach().cpu().numpy().astype(np.float32, copy=False)
        H_T = output["H_T"].detach().cpu().numpy().astype(np.float32, copy=False)
        B_I = output["B_I"].detach().cpu().numpy().astype(np.int8, copy=False)
        B_T = output["B_T"].detach().cpu().numpy().astype(np.int8, copy=False)
        _check_forward_arrays(H_I, H_T, B_I, B_T, int(dataset_config["expected_train_count"]), bit)
        np.save(bit_dir / "H_I.npy", H_I)
        np.save(bit_dir / "H_T.npy", H_T)
        np.save(bit_dir / "B_I.npy", B_I)
        np.save(bit_dir / "B_T.npy", B_T)
        meta = _stage4_meta(config, dataset_config, context, output, dataset, bit)
        write_json(bit_dir / "meta.json", meta)
        bit_summaries[str(bit)] = {
            "output_dir": str(bit_dir),
            "H_I_shape": list(H_I.shape),
            "H_T_shape": list(H_T.shape),
            "B_I_shape": list(B_I.shape),
            "B_T_shape": list(B_T.shape),
            "H_I_range": [float(np.min(H_I)), float(np.max(H_I))],
            "H_T_range": [float(np.min(H_T)), float(np.max(H_T))],
            "B_unique": sorted(int(value) for value in np.unique(np.concatenate([B_I.reshape(-1), B_T.reshape(-1)]))),
        }
        del model, X_I_tensor, X_T_tensor, output
        torch.cuda.empty_cache()

    return {
        "stage": "stage4",
        "substage": _stage4_substage(dataset),
        "dataset": dataset,
        "cache_id": config["cache_id"],
        "hash_bits": bits,
        "bits": bit_summaries,
    }


def _load_context(
    paths: dict[str, Path],
    config: dict[str, Any],
    dataset_config: dict[str, Any],
    dataset: str,
) -> dict[str, Any]:
    required = [
        paths["manifest_filtered"],
        paths["train_ids"],
        paths["order_hashes"],
        paths["x_i"],
        paths["x_t"],
        paths["stage2_meta"],
        paths["s"],
        paths["stage3_meta"],
        paths["stage3_diagnostics"],
    ]
    for path in required:
        if not path.exists():
            raise RuntimeError(f"missing Stage 4 input: {path}")
    rows = list(iter_jsonl(paths["manifest_filtered"]))
    train_ids = _read_lines(paths["train_ids"])
    order_hashes = read_json(paths["order_hashes"])
    stage2_meta = read_json(paths["stage2_meta"])
    stage3_meta = read_json(paths["stage3_meta"])
    stage3_diagnostics = read_json(paths["stage3_diagnostics"])
    expected_train = int(dataset_config["expected_train_count"])
    if len(train_ids) != expected_train:
        raise RuntimeError(f"train_count mismatch: expected {expected_train}, got {len(train_ids)}")
    sample_ids = [str(row["sample_id"]) for row in rows]
    expected_hashes = {
        "sample_id_order_sha256": hash_lines(sorted(sample_ids)),
        "manifest_filtered_order_sha256": hash_lines(sample_ids),
        "train_ids_sha256": hash_lines(train_ids),
    }
    for key, value in expected_hashes.items():
        if order_hashes.get(key) != value:
            raise RuntimeError(f"Stage 1 order_hashes mismatch for {key}")
    train_indices = _train_indices(rows, train_ids)
    S = np.load(paths["s"], mmap_mode="r")
    if S.shape != (expected_train, expected_train):
        raise RuntimeError(f"S shape mismatch: expected {[expected_train, expected_train]}, got {list(S.shape)}")
    if S.dtype != np.float32:
        raise RuntimeError(f"S dtype mismatch: expected float32, got {S.dtype}")
    if not np.isfinite(S).all():
        raise RuntimeError("S contains NaN or Inf")

    stage1_hash_match = (
        stage2_meta.get("manifest_filtered_order_sha256") == order_hashes.get("manifest_filtered_order_sha256")
        and stage2_meta.get("train_ids_sha256") == order_hashes.get("train_ids_sha256")
        and stage3_meta.get("stage1_manifest_filtered_order_sha256") == order_hashes.get("manifest_filtered_order_sha256")
        and stage3_meta.get("stage1_train_ids_sha256") == order_hashes.get("train_ids_sha256")
    )
    stage2_hash_match = (
        stage3_meta.get("stage2_manifest_filtered_order_sha256") == stage2_meta.get("manifest_filtered_order_sha256")
        and stage3_meta.get("stage2_feature_set_id") == stage2_meta.get("feature_set_id")
        and stage2_meta.get("feature_set_id") == config["feature_set_id"]
    )
    stage3_hash_match = (
        stage3_meta.get("semantic_set_id") == config["semantic_set_id"]
        and stage3_meta.get("train_count") == expected_train
        and stage3_meta.get("matrix_shape") == [expected_train, expected_train]
        and bool(stage3_diagnostics.get("semantic_validator_passed")) is True
    )
    if not stage1_hash_match:
        raise RuntimeError("Stage 1 hash/meta consistency check failed")
    if not stage2_hash_match:
        raise RuntimeError("Stage 2 hash/meta consistency check failed")
    if not stage3_hash_match:
        raise RuntimeError("Stage 3 hash/meta consistency check failed")
    return {
        "rows": rows,
        "train_ids": train_ids,
        "train_indices": train_indices,
        "order_hashes": order_hashes,
        "stage2_meta": stage2_meta,
        "stage3_meta": stage3_meta,
        "stage3_diagnostics": stage3_diagnostics,
        "hash_checks": {
            "stage1_hash_match": True,
            "stage2_hash_match": True,
            "stage3_hash_match": True,
            "train_mapping_verified": True,
        },
    }


def _load_train_features(
    paths: dict[str, Path],
    dataset_config: dict[str, Any],
    train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_i = np.load(paths["x_i"], mmap_mode="r")
    x_t = np.load(paths["x_t"], mmap_mode="r")
    input_dim = int(dataset_config["input_feature_dim"])
    if x_i.ndim != 2 or x_i.shape[1] != input_dim:
        raise RuntimeError(f"X_I feature shape invalid: {list(x_i.shape)}")
    if x_t.ndim != 2 or x_t.shape[1] != input_dim:
        raise RuntimeError(f"X_T feature shape invalid: {list(x_t.shape)}")
    if x_i.dtype != np.float32 or x_t.dtype != np.float32:
        raise RuntimeError("Stage 4 requires float32 Stage 2 features")
    X_I_train = np.asarray(x_i[train_indices], dtype=np.float32)
    X_T_train = np.asarray(x_t[train_indices], dtype=np.float32)
    if X_I_train.shape != (int(dataset_config["expected_train_count"]), input_dim):
        raise RuntimeError(f"X_I_train shape invalid: {list(X_I_train.shape)}")
    if X_T_train.shape != (int(dataset_config["expected_train_count"]), input_dim):
        raise RuntimeError(f"X_T_train shape invalid: {list(X_T_train.shape)}")
    if not np.isfinite(X_I_train).all() or not np.isfinite(X_T_train).all():
        raise RuntimeError("Stage 4 train features contain NaN or Inf")
    return X_I_train, X_T_train


def _stage4_meta(
    config: dict[str, Any],
    dataset_config: dict[str, Any],
    context: dict[str, Any],
    output: dict[str, Any],
    dataset: str,
    bit: int,
) -> dict[str, Any]:
    order_hashes = context["order_hashes"]
    stage2_meta = context["stage2_meta"]
    stage3_meta = context["stage3_meta"]
    return {
        "dataset": dataset,
        "bit": bit,
        "stage4_model_id": config["stage4_model_id"],
        "cache_id": config["cache_id"],
        "feature_set_id": config["feature_set_id"],
        "semantic_set_id": config["semantic_set_id"],
        "train_count": int(dataset_config["expected_train_count"]),
        "input_feature_dim": int(dataset_config["input_feature_dim"]),
        "d_z": int(dataset_config["d_z"]),
        "cheby_order": int(dataset_config["cheby_order"]),
        "tree_levels": int(dataset_config["tree_levels"]),
        "tree_prototypes": [int(value) for value in dataset_config["tree_prototypes"]],
        "graph_k_train": int(dataset_config["graph_k_train"]),
        "beta_tree_injection": float(dataset_config["beta_tree_injection"]),
        "hash_bits_all": [int(value) for value in config["hash_bits"]],
        "stage1_hashes": {
            "manifest_filtered_order_sha256": order_hashes["manifest_filtered_order_sha256"],
            "sample_id_order_sha256": order_hashes["sample_id_order_sha256"],
            "train_ids_sha256": order_hashes["train_ids_sha256"],
            "query_ids_sha256": order_hashes.get("query_ids_sha256"),
            "retrieval_ids_sha256": order_hashes.get("retrieval_ids_sha256"),
        },
        "stage2_hashes": {
            "feature_set_id": stage2_meta["feature_set_id"],
            "manifest_filtered_order_sha256": stage2_meta["manifest_filtered_order_sha256"],
            "sample_id_order_sha256": stage2_meta["sample_id_order_sha256"],
            "train_ids_sha256": stage2_meta["train_ids_sha256"],
        },
        "stage3_hashes": {
            "semantic_set_id": stage3_meta["semantic_set_id"],
            "stage1_manifest_filtered_order_sha256": stage3_meta["stage1_manifest_filtered_order_sha256"],
            "stage1_train_ids_sha256": stage3_meta["stage1_train_ids_sha256"],
            "stage2_manifest_filtered_order_sha256": stage3_meta["stage2_manifest_filtered_order_sha256"],
            "train_ids_sha256": stage3_meta["train_ids_sha256"],
            "train_indices_sha256": stage3_meta["train_indices_sha256"],
        },
        "hash_checks": context["hash_checks"],
        "random_seed": int(config["runtime"]["seed"]),
        "device": str(config["runtime"]["device"]),
        "dtype": str(config["runtime"]["dtype"]),
        "model_eval_mode": True,
        "stage4_forward_is_untrained_sanity": True,
        "not_final_retrieval_result": True,
        "generated_at_utc": _utc_now(),
        "tree_diagnostics": _jsonable(output["tree_diagnostics"]),
        "graph_diagnostics": _jsonable(output["graph_diagnostics"]),
        "dev_only_tree_candidates_recorded_not_run": True,
    }


def _check_forward_arrays(H_I: np.ndarray, H_T: np.ndarray, B_I: np.ndarray, B_T: np.ndarray, n: int, bit: int) -> None:
    expected = (n, bit)
    for name, H in [("H_I", H_I), ("H_T", H_T)]:
        if H.shape != expected:
            raise RuntimeError(f"{name} shape mismatch: expected {expected}, got {H.shape}")
        if H.dtype != np.float32:
            raise RuntimeError(f"{name} dtype must be float32")
        if not np.isfinite(H).all():
            raise RuntimeError(f"{name} contains NaN or Inf")
        if np.min(H) < -1.000001 or np.max(H) > 1.000001:
            raise RuntimeError(f"{name} outside [-1, 1]")
    for name, H, B in [("B_I", H_I, B_I), ("B_T", H_T, B_T)]:
        if B.shape != expected:
            raise RuntimeError(f"{name} shape mismatch: expected {expected}, got {B.shape}")
        if B.dtype != np.int8:
            raise RuntimeError(f"{name} dtype must be int8")
        if not set(int(value) for value in np.unique(B)).issubset({-1, 1}):
            raise RuntimeError(f"{name} contains values outside {{-1,+1}}")
        if not np.array_equal(B, np.where(H >= 0.0, 1, -1).astype(np.int8)):
            raise RuntimeError(f"{name} sign rule check failed")


def _check_runtime(config: dict[str, Any]) -> None:
    runtime = config["runtime"]
    if runtime["device"] != "cuda:0":
        raise RuntimeError("Stage 4 forward formal run requires cuda:0")
    if runtime["dtype"] != "float32":
        raise RuntimeError("Stage 4 forward formal run requires float32")
    if runtime.get("amp_enabled") is not False:
        raise RuntimeError("Stage 4 forward does not allow AMP")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is not allowed")


def _check_config(config: dict[str, Any]) -> None:
    for dataset, profile in config["datasets"].items():
        if int(profile["tree_levels"]) != len(profile["tree_prototypes"]):
            raise RuntimeError(f"{dataset} tree_levels must match tree_prototypes length")
        if int(profile["tree_levels"]) >= 5:
            raise RuntimeError(f"{dataset} main tree_levels must be < 5")
    for dataset, candidates in config.get("dev_only_tree_candidates", {}).items():
        for candidate in candidates:
            if int(candidate["tree_levels"]) != len(candidate["tree_prototypes"]):
                raise RuntimeError(f"{dataset} candidate tree_levels must match tree_prototypes length")
            if int(candidate["tree_levels"]) >= 5:
                raise RuntimeError(f"{dataset} candidate tree_levels must be < 5")
    if config.get("execution_policy", {}).get("run_dev_only_tree_candidates_in_stage4_forward") is not False:
        raise RuntimeError("Stage 4 forward must not run dev-only tree candidates")


def _check_stage4_forward_dataset(dataset: str, config: dict[str, Any]) -> None:
    allowed = set(config.get("execution_policy", {}).get("stage4_forward_allowed_datasets", []))
    if dataset not in SUPPORTED_STAGE4_FORWARD_DATASETS or dataset not in allowed:
        raise RuntimeError(f"Stage 4 forward is authorized for mirflickr25k/nuswide/mscoco only; got {dataset}")


def _stage4_substage(dataset: str) -> str:
    if dataset == "nuswide":
        return "stage4b_nuswide_untrained_forward"
    if dataset == "mirflickr25k":
        return "stage4b_mirflickr25k_regression"
    if dataset == "mscoco":
        return "stage4c_mscoco_untrained_forward"
    raise RuntimeError(f"unsupported Stage 4 substage dataset: {dataset}")


def _train_indices(rows: list[dict[str, Any]], train_ids: list[str]) -> np.ndarray:
    sample_ids = [str(row["sample_id"]) for row in rows]
    id_to_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    indices: list[int] = []
    for sample_id in train_ids:
        index = id_to_index.get(sample_id)
        if index is None:
            raise RuntimeError(f"train_id not present in manifest_filtered: {sample_id}")
        indices.append(index)
    return np.asarray(indices, dtype=np.int64)


def _stage4_paths(processed_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    feature_dir = processed_root / "feature_cache" / str(config["feature_set_id"])
    semantic_dir = processed_root / "semantic_cache" / str(config["semantic_set_id"])
    return {
        "manifest_filtered": processed_root / "manifest" / "manifest_filtered.jsonl",
        "train_ids": processed_root / "splits" / "train_ids.txt",
        "order_hashes": processed_root / "reports" / "order_hashes.json",
        "x_i": feature_dir / "X_I.npy",
        "x_t": feature_dir / "X_T.npy",
        "stage2_meta": feature_dir / "meta.json",
        "s": semantic_dir / "S.npy",
        "stage3_meta": semantic_dir / "meta.json",
        "stage3_diagnostics": semantic_dir / "semantic_diagnostics.json",
        "cache_root": processed_root / "model_cache" / str(config["cache_id"]),
    }


def _read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def hash_lines(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(str(line).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _resolve_repo_path(repo_root: Path, path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
    resolved.relative_to(repo_root)
    return resolved


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())

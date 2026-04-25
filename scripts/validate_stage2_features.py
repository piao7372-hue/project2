from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.validators.stage2_validator import validate_stage2_features
from src.utils.jsonl import read_json


FORMAL_STAGE2_CONFIG = REPO_ROOT / "configs" / "stages" / "stage2_features.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate formal Stage 2 feature outputs.")
    parser.add_argument("--dataset", required=True, choices=["mirflickr25k", "nuswide", "mscoco"])
    parser.add_argument("--config", default="configs/stages/stage2_features.json")
    return parser.parse_args()


def enforce_formal_python() -> None:
    config = read_json(FORMAL_STAGE2_CONFIG)
    expected = Path(config["runtime"]["formal_python_path"]).resolve()
    current = Path(sys.executable).resolve()
    if os.path.normcase(str(expected)) != os.path.normcase(str(current)):
        raise RuntimeError(f"Stage 2 validator requires formal Python: current={current}; expected={expected}")


def main() -> int:
    enforce_formal_python()
    args = parse_args()
    summary = validate_stage2_features(REPO_ROOT, Path(args.config), args.dataset)
    print(f"dataset={summary['dataset']}")
    print(f"feature_set_id={summary['feature_set_id']}")
    print(f"feature_cache_dir={summary['feature_cache_dir']}")
    print(f"X_I_shape={summary['x_i_shape']}")
    print(f"X_T_shape={summary['x_t_shape']}")
    print(f"X_I_dtype={summary['x_i_dtype']}")
    print(f"X_T_dtype={summary['x_t_dtype']}")
    print(f"X_I_norm_range={summary['x_i_norm_range']}")
    print(f"X_T_norm_range={summary['x_t_norm_range']}")
    print(f"meta_hashes_match_stage1={str(summary['meta_hashes_match_stage1']).lower()}")
    print(f"baseline_completed={str(summary['baseline_completed']).lower()}")
    print(f"paired_cosine_mean={summary['paired_cosine_mean']}")
    print(f"random_cosine_mean={summary['random_cosine_mean']}")
    print(f"cosine_gap_mean={summary['cosine_gap_mean']}")
    print(f"paired_cosine_median={summary['paired_cosine_median']}")
    print(f"random_cosine_median={summary['random_cosine_median']}")
    print(f"cosine_gap_median={summary['cosine_gap_median']}")
    print(f"clip_i2t_map_at_50={summary['clip_i2t_map_at_50']}")
    print(f"clip_t2i_map_at_50={summary['clip_t2i_map_at_50']}")
    print(f"passed={str(summary['passed']).lower()}")
    for failure in summary["failure_reason"]:
        print(f"failure={failure}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

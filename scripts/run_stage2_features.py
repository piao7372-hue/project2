from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.clip_formal import run_stage2_features
from src.utils.jsonl import read_json


FORMAL_STAGE2_CONFIG = REPO_ROOT / "configs" / "stages" / "stage2_features.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run formal Stage 2 feature extraction.")
    parser.add_argument("--dataset", required=True, choices=["mirflickr25k", "nuswide", "mscoco"])
    parser.add_argument("--config", default="configs/stages/stage2_features.json")
    return parser.parse_args()


def enforce_formal_python() -> None:
    config = read_json(FORMAL_STAGE2_CONFIG)
    expected = Path(config["runtime"]["formal_python_path"]).resolve()
    current = Path(sys.executable).resolve()
    if os.path.normcase(str(expected)) != os.path.normcase(str(current)):
        raise RuntimeError(f"Stage 2 requires formal Python: current={current}; expected={expected}")


def main() -> int:
    enforce_formal_python()
    args = parse_args()
    summary = run_stage2_features(REPO_ROOT, Path(args.config), args.dataset)
    print(f"dataset={summary['dataset']}")
    print(f"feature_set_id={summary['feature_set_id']}")
    print(f"output_dir={summary['output_dir']}")
    print(f"filtered_count={summary['filtered_count']}")
    print(f"query_count={summary['query_count']}")
    print(f"retrieval_count={summary['retrieval_count']}")
    print(f"X_I_shape={summary['x_i_shape']}")
    print(f"X_T_shape={summary['x_t_shape']}")
    print(f"X_I_dtype={summary['x_i_dtype']}")
    print(f"X_T_dtype={summary['x_t_dtype']}")
    print(f"X_I_norm_min={summary['x_i_norm_min']}")
    print(f"X_I_norm_max={summary['x_i_norm_max']}")
    print(f"X_T_norm_min={summary['x_t_norm_min']}")
    print(f"X_T_norm_max={summary['x_t_norm_max']}")
    baseline = summary["baseline_summary"]
    print(f"baseline_completed={str(baseline['baseline_completed']).lower()}")
    print(f"paired_cosine_mean={baseline['paired_cosine_mean']}")
    print(f"random_cosine_mean={baseline['random_cosine_mean']}")
    print(f"cosine_gap_mean={baseline['cosine_gap_mean']}")
    print(f"paired_cosine_median={baseline['paired_cosine_median']}")
    print(f"random_cosine_median={baseline['random_cosine_median']}")
    print(f"cosine_gap_median={baseline['cosine_gap_median']}")
    print(f"clip_i2t_map_at_50={baseline['clip_i2t_map_at_50']}")
    print(f"clip_t2i_map_at_50={baseline['clip_t2i_map_at_50']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

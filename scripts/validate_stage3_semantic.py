from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.validators.stage3_validator import validate_stage3_semantic
from src.utils.jsonl import read_json


FORMAL_STAGE3_CONFIG = REPO_ROOT / "configs" / "stages" / "stage3_semantic.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate formal Stage 3 semantic relation outputs.")
    parser.add_argument("--dataset", required=True, choices=["mirflickr25k", "nuswide", "mscoco"])
    parser.add_argument("--config", default="configs/stages/stage3_semantic.json")
    return parser.parse_args()


def enforce_formal_python() -> None:
    config = read_json(FORMAL_STAGE3_CONFIG)
    expected = Path(config["runtime"]["formal_python_path"]).resolve()
    current = Path(sys.executable).resolve()
    if os.path.normcase(str(expected)) != os.path.normcase(str(current)):
        raise RuntimeError(f"Stage 3 validator requires formal Python: current={current}; expected={expected}")


def main() -> int:
    enforce_formal_python()
    args = parse_args()
    summary = validate_stage3_semantic(REPO_ROOT, Path(args.config), args.dataset)
    print(f"dataset={summary['dataset']}")
    print(f"semantic_set_id={summary['semantic_set_id']}")
    print(f"semantic_cache_dir={summary['semantic_cache_dir']}")
    print(f"train_count={summary['train_count']}")
    print(f"matrix_shape={summary['matrix_shape']}")
    print(f"lambda_ar_fusion={summary['lambda_ar_fusion']}")
    print(f"tau_confidence={summary['tau_confidence']}")
    print(f"topk_for_diagnostics={summary['topk_for_diagnostics']}")
    print(f"meta_hashes_match_stage1={str(summary['meta_hashes_match_stage1']).lower()}")
    print(f"meta_hashes_match_stage2={str(summary['meta_hashes_match_stage2']).lower()}")
    print(f"train_mapping_verified={str(summary['train_mapping_verified']).lower()}")
    print(f"omega_topk_diag_diagnostic_only={str(summary['omega_topk_diag_diagnostic_only']).lower()}")
    diagnostics = summary["diagnostics"]
    if diagnostics:
        print(f"diag_mean_s={diagnostics['diag_mean_s']}")
        print(f"offdiag_mean_s={diagnostics['offdiag_mean_s']}")
        print(f"diag_minus_offdiag_s={diagnostics['diag_minus_offdiag_s']}")
        print(f"paired_diag_quantile_in_row_median={diagnostics['paired_diag_quantile_in_row_median']}")
        print(f"paired_diag_quantile_in_col_median={diagnostics['paired_diag_quantile_in_col_median']}")
        print(f"diag_in_row_topk_rate={diagnostics['diag_in_row_topk_rate']}")
        print(f"diag_in_col_topk_rate={diagnostics['diag_in_col_topk_rate']}")
    print(f"passed={str(summary['passed']).lower()}")
    for failure in summary["failure_reason"]:
        print(f"failure={failure}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

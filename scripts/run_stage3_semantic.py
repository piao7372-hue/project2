from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.semantic.semantic_relation import run_stage3_semantic
from src.utils.jsonl import read_json


FORMAL_STAGE3_CONFIG = REPO_ROOT / "configs" / "stages" / "stage3_semantic.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run formal Stage 3 semantic relation construction.")
    parser.add_argument("--dataset", required=True, choices=["mirflickr25k", "nuswide", "mscoco"])
    parser.add_argument("--config", default="configs/stages/stage3_semantic.json")
    return parser.parse_args()


def enforce_formal_python() -> None:
    config = read_json(FORMAL_STAGE3_CONFIG)
    expected = Path(config["runtime"]["formal_python_path"]).resolve()
    current = Path(sys.executable).resolve()
    if os.path.normcase(str(expected)) != os.path.normcase(str(current)):
        raise RuntimeError(f"Stage 3 requires formal Python: current={current}; expected={expected}")


def main() -> int:
    enforce_formal_python()
    args = parse_args()
    summary = run_stage3_semantic(REPO_ROOT, Path(args.config), args.dataset)
    print(f"dataset={summary['dataset']}")
    print(f"semantic_set_id={summary['semantic_set_id']}")
    print(f"semantic_cache_dir={summary['semantic_cache_dir']}")
    print(f"train_count={summary['train_count']}")
    print(f"matrix_shape={summary['matrix_shape']}")
    print(f"lambda_ar_fusion={summary['lambda_ar_fusion']}")
    print(f"tau_confidence={summary['tau_confidence']}")
    print(f"topk_for_diagnostics={summary['topk_for_diagnostics']}")
    print(f"A_shape={summary['a_shape']}")
    print(f"R_shape={summary['r_shape']}")
    print(f"Se_shape={summary['se_shape']}")
    print(f"C_shape={summary['c_shape']}")
    print(f"S_shape={summary['s_shape']}")
    diagnostics = summary["diagnostics"]
    print(f"diag_mean_s={diagnostics['diag_mean_s']}")
    print(f"offdiag_mean_s={diagnostics['offdiag_mean_s']}")
    print(f"diag_minus_offdiag_s={diagnostics['diag_minus_offdiag_s']}")
    print(f"paired_diag_quantile_in_row_median={diagnostics['paired_diag_quantile_in_row_median']}")
    print(f"paired_diag_quantile_in_col_median={diagnostics['paired_diag_quantile_in_col_median']}")
    print(f"semantic_validator_passed={str(diagnostics['semantic_validator_passed']).lower()}")
    return 0 if diagnostics["semantic_validator_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

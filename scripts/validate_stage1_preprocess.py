from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.validators.stage1_validator import validate_stage1_preprocess
from src.utils.jsonl import read_json

FORMAL_STAGE1_CONFIG = REPO_ROOT / "configs" / "stages" / "stage1_preprocess.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate formal Stage 1 preprocessing outputs.")
    parser.add_argument("--dataset", required=True, choices=["mirflickr25k", "nuswide", "mscoco"])
    parser.add_argument("--config", default="configs/stages/stage1_preprocess.json")
    return parser.parse_args()


def enforce_formal_python() -> None:
    config = read_json(FORMAL_STAGE1_CONFIG)
    expected = Path(config["runtime"]["formal_python_path"]).resolve()
    current = Path(sys.executable).resolve()
    if os.path.normcase(str(expected)) != os.path.normcase(str(current)):
        raise RuntimeError(f"Stage 1 validator requires formal Python: current={current}; expected={expected}")


def main() -> int:
    enforce_formal_python()
    args = parse_args()
    summary = validate_stage1_preprocess(REPO_ROOT, Path(args.config), args.dataset)
    print(f"dataset={summary['dataset_name']}")
    print(f"validator_summary_path={REPO_ROOT / 'data' / 'processed' / args.dataset / 'reports' / 'validator_summary.json'}")
    print(f"manifest_raw_count={summary['manifest_raw_count']}")
    print(f"manifest_filtered_count={summary['manifest_filtered_count']}")
    print(f"query_count={summary['query_count']}")
    print(f"retrieval_count={summary['retrieval_count']}")
    print(f"train_count={summary['train_count']}")
    print(f"passed={str(summary['passed']).lower()}")
    for failure in summary["failure_reason"]:
        print(f"failure={failure}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

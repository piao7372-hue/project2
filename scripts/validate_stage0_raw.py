from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.validators.raw_validator import run_stage0_raw_validator

FORMAL_STAGE0_CONFIG = REPO_ROOT / "configs" / "stages" / "stage0_formal.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run formal Stage 0 raw validator.")
    parser.add_argument("--config", default="configs/datasets/raw_roots.json")
    parser.add_argument("--output-root", default="outputs/stage0_raw_validator")
    return parser.parse_args()


def enforce_formal_python() -> None:
    with FORMAL_STAGE0_CONFIG.open("r", encoding="utf-8-sig") as handle:
        config = json.load(handle)
    expected = Path(config["runtime"]["formal_python_path"]).resolve()
    current = Path(sys.executable).resolve()
    if os.path.normcase(str(expected)) != os.path.normcase(str(current)):
        raise RuntimeError(f"Stage 0R-4 requires formal Python: current={current}; expected={expected}")


def main() -> int:
    enforce_formal_python()
    args = parse_args()
    summary = run_stage0_raw_validator(REPO_ROOT, Path(args.config), Path(args.output_root))
    print(f"summary_json_path={summary['summary_json_path']}")
    print(f"summary_markdown_path={summary['summary_markdown_path']}")
    print(f"all_raw_validators_passed={str(summary['all_raw_validators_passed']).lower()}")
    for dataset, result in summary["dataset_results"].items():
        print(f"{dataset}: passed={str(result['passed']).lower()} failure_count={result['failure_count']}")
    print(f"stage1_allowed={str(summary['stage1_allowed']).lower()}")
    print(f"stage1_allowed_reason={summary['stage1_allowed_reason']}")
    return 0 if summary["all_raw_validators_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

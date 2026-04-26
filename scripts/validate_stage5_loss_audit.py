from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.validators.stage5_validator import validate_stage5_loss_audit
from src.utils.jsonl import read_json


FORMAL_STAGE5_CONFIG = REPO_ROOT / "configs" / "stages" / "stage5_loss.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Stage 5 loss audit outputs.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--all-bits", action="store_true")
    parser.add_argument("--config", default="configs/stages/stage5_loss.json")
    return parser.parse_args()


def enforce_formal_python() -> None:
    config = read_json(FORMAL_STAGE5_CONFIG)
    expected = Path(config["runtime"]["python"]).resolve()
    current = Path(sys.executable).resolve()
    if os.path.normcase(str(expected)) != os.path.normcase(str(current)):
        raise RuntimeError(f"Stage 5 validator requires formal Python: current={current}; expected={expected}")


def main() -> int:
    enforce_formal_python()
    args = parse_args()
    summary = validate_stage5_loss_audit(REPO_ROOT, Path(args.config), args.dataset, args.all_bits)
    print(f"dataset={summary['dataset']}")
    print(f"hash_bits={summary['hash_bits']}")
    print(f"beta_candidates={summary['beta_candidates']}")
    print(f"derived_profile_norm_risk={summary['derived_profile_norm_risk']}")
    for bit, bit_summary in summary["bits"].items():
        print(f"bit={bit}")
        print(f"passed={str(bit_summary['passed']).lower()}")
        print(f"failure_count={bit_summary['failure_count']}")
        for failure in bit_summary["failure_reason"]:
            print(f"failure={failure}")
    print(f"failure_count={summary['failure_count']}")
    for failure in summary["failure_reason"]:
        print(f"failure={failure}")
    print(f"passed={str(summary['passed']).lower()}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

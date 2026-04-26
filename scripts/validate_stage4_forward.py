from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.validators.stage4_validator import validate_stage4_forward
from src.utils.jsonl import read_json


FORMAL_STAGE4_CONFIG = REPO_ROOT / "configs" / "stages" / "stage4_model.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Stage 4 untrained forward sanity outputs.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--all-bits", action="store_true")
    parser.add_argument("--config", default="configs/stages/stage4_model.json")
    return parser.parse_args()


def enforce_formal_python() -> None:
    config = read_json(FORMAL_STAGE4_CONFIG)
    expected = Path(config["runtime"]["python"]).resolve()
    current = Path(sys.executable).resolve()
    if os.path.normcase(str(expected)) != os.path.normcase(str(current)):
        raise RuntimeError(f"Stage 4 validator requires formal Python: current={current}; expected={expected}")


def main() -> int:
    enforce_formal_python()
    args = parse_args()
    summary = validate_stage4_forward(REPO_ROOT, Path(args.config), args.dataset, args.all_bits)
    print(f"dataset={summary['dataset']}")
    print(f"cache_id={summary['cache_id']}")
    print(f"hash_bits={summary['hash_bits']}")
    for bit, bit_summary in summary["bits"].items():
        print(f"bit={bit}")
        print(f"passed={str(bit_summary['passed']).lower()}")
        health_i = bit_summary["bit_health"].get("image", {})
        health_t = bit_summary["bit_health"].get("text", {})
        print(f"image_unique_code_ratio={health_i.get('unique_code_ratio')}")
        print(f"text_unique_code_ratio={health_t.get('unique_code_ratio')}")
        print(f"image_constant_bit_ratio={health_i.get('constant_bit_ratio')}")
        print(f"text_constant_bit_ratio={health_t.get('constant_bit_ratio')}")
        print(f"tree_risk_level={bit_summary['tree_risk']['risk_level']}")
        print(f"failure_count={bit_summary['failure_count']}")
        for failure in bit_summary["failure_reason"]:
            print(f"failure={failure}")
    print(f"passed={str(summary['passed']).lower()}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

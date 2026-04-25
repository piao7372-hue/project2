from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.validators.raw_preflight import run_raw_preflight


FORMAL_STAGE0_CONFIG = REPO_ROOT / "configs" / "stages" / "stage0_formal.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 0 raw layout preflight.")
    parser.add_argument(
        "--config",
        default="configs/datasets/raw_roots.json",
        help="Raw root config JSON. No fallback paths are used.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/stage0_raw_preflight",
        help="Ignored output directory for raw preflight JSON products.",
    )
    return parser.parse_args()


def enforce_formal_python() -> None:
    with FORMAL_STAGE0_CONFIG.open("r", encoding="utf-8-sig") as handle:
        stage0_config = json.load(handle)
    expected = Path(stage0_config["runtime"]["formal_python_path"]).resolve()
    current = Path(sys.executable).resolve()
    if os.path.normcase(str(expected)) != os.path.normcase(str(current)):
        raise RuntimeError(
            "Stage 0 raw preflight must run with the formal Python interpreter. "
            f"current={current}; expected={expected}"
        )


def main() -> int:
    enforce_formal_python()
    args = parse_args()
    summary = run_raw_preflight(
        repo_root=REPO_ROOT,
        config_path=Path(args.config),
        output_root=Path(args.output_root),
    )
    print(f"summary_path={summary['summary_path']}")
    print(f"all_preflight_passed={str(summary['all_preflight_passed']).lower()}")
    for dataset in ("mirflickr25k", "mscoco", "nuswide"):
        result = summary["datasets"][dataset]
        print(
            f"{dataset}: passed={str(result['formal_raw_ready']).lower()} "
            f"failure_count={len(result['failure_reason'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


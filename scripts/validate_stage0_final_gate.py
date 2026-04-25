from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.stage0_final_gate import run_stage0_final_gate

FORMAL_STAGE0_CONFIG = REPO_ROOT / "configs" / "stages" / "stage0_formal.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 0 final gate checks without entering Stage 1.")
    parser.add_argument("--output-root", default="outputs/stage0_final_gate")
    return parser.parse_args()


def enforce_formal_python() -> None:
    with FORMAL_STAGE0_CONFIG.open("r", encoding="utf-8-sig") as handle:
        config = json.load(handle)
    expected = Path(config["runtime"]["formal_python_path"]).resolve()
    current = Path(sys.executable).resolve()
    if os.path.normcase(str(expected)) != os.path.normcase(str(current)):
        raise RuntimeError(f"Stage 0F-1 requires formal Python: current={current}; expected={expected}")


def main() -> int:
    enforce_formal_python()
    args = parse_args()
    summary = run_stage0_final_gate(REPO_ROOT, Path(args.output_root))
    print(f"summary_json_path={summary['summary_json_path']}")
    print(f"summary_markdown_path={summary['summary_markdown_path']}")
    print(f"engineering_spec_compliance_path={summary['engineering_spec_compliance_path']}")
    print(f"stage0_environment_lock_passed={str(summary['stage0_environment_lock_passed']).lower()}")
    print(f"stage0_raw_validator_passed={str(summary['stage0_raw_validator_passed']).lower()}")
    print(f"stage0_clip_validation_passed={str(summary['stage0_clip_validation_passed']).lower()}")
    print(f"engineering_spec_compliance_passed={str(summary['engineering_spec_compliance_passed']).lower()}")
    print(f"git_ignore_gate_passed={str(summary['git_ignore_gate_passed']).lower()}")
    print(f"stage1_artifact_absence_passed={str(summary['stage1_artifact_absence_passed']).lower()}")
    print(f"faiss_missing={str(summary['faiss_missing']).lower()}")
    print(f"stage0_complete={str(summary['stage0_complete']).lower()}")
    print(f"stage1_allowed={str(summary['stage1_allowed']).lower()}")
    print(f"stage1_allowed_reason={summary['stage1_allowed_reason']}")
    if summary["blocker_reason"]:
        for blocker in summary["blocker_reason"]:
            print(f"blocker={blocker}")
    return 0 if summary["stage0_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.clip_weight_prepare import prepare_stage0_clip_weights

FORMAL_STAGE0_CONFIG = REPO_ROOT / "configs" / "stages" / "stage0_formal.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and validate formal Stage 0 CLIP weights.")
    parser.add_argument("--config", default="configs/stages/stage0_formal.json")
    parser.add_argument("--output-root", default="outputs/stage0_clip")
    return parser.parse_args()


def enforce_formal_python() -> None:
    with FORMAL_STAGE0_CONFIG.open("r", encoding="utf-8-sig") as handle:
        config = json.load(handle)
    expected = Path(config["runtime"]["formal_python_path"]).resolve()
    current = Path(sys.executable).resolve()
    if os.path.normcase(str(expected)) != os.path.normcase(str(current)):
        raise RuntimeError(f"Stage 0C-1 requires formal Python: current={current}; expected={expected}")


def main() -> int:
    enforce_formal_python()
    args = parse_args()
    summary = prepare_stage0_clip_weights(REPO_ROOT, Path(args.config), Path(args.output_root))
    print(f"summary_json_path={summary['summary_json_path']}")
    print(f"summary_markdown_path={summary['summary_markdown_path']}")
    print(f"backbone_id={summary['backbone_id']}")
    print(f"model_local_path={summary['model_local_path']}")
    print(f"online_download_used={str(summary['online_download_used']).lower()}")
    print(f"model_load_ok={str(summary['model_load_ok']).lower()}")
    print(f"processor_load_ok={str(summary['processor_load_ok']).lower()}")
    print(f"allow_online_download_after_run={str(summary['allow_online_download_after_run']).lower()}")
    print(f"stage1_entered=false")
    if summary["failure_reason"]:
        for failure in summary["failure_reason"]:
            print(f"failure={failure}")
    return 0 if summary["model_load_ok"] and summary["processor_load_ok"] and not summary["failure_reason"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

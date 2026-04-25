from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.environment_lock import run_environment_lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Stage 0A environment lock.")
    parser.add_argument(
        "--config",
        default="configs/stages/stage0_formal.json",
        help="Path to the Stage 0 formal config JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_environment_lock(REPO_ROOT, Path(args.config))
    print(f"wrote_json={result.lock_json_path}")
    print(f"wrote_markdown={result.lock_markdown_path}")
    print(f"stage0a_ready={str(result.stage0a_ready).lower()}")
    print(f"formal_dependency_ready={str(result.formal_dependency_ready).lower()}")
    if result.missing_packages:
        print("missing_packages=" + ",".join(result.missing_packages))
    else:
        print("missing_packages=")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

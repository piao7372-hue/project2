from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.builders.stage1_preprocess import run_stage1_preprocess
from src.utils.jsonl import read_json

FORMAL_STAGE1_CONFIG = REPO_ROOT / "configs" / "stages" / "stage1_preprocess.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run formal Stage 1 preprocessing.")
    parser.add_argument("--dataset", required=True, choices=["mirflickr25k", "nuswide", "mscoco"])
    parser.add_argument("--config", default="configs/stages/stage1_preprocess.json")
    return parser.parse_args()


def enforce_formal_python() -> None:
    config = read_json(FORMAL_STAGE1_CONFIG)
    expected = Path(config["runtime"]["formal_python_path"]).resolve()
    current = Path(sys.executable).resolve()
    if os.path.normcase(str(expected)) != os.path.normcase(str(current)):
        raise RuntimeError(f"Stage 1 requires formal Python: current={current}; expected={expected}")


def main() -> int:
    enforce_formal_python()
    args = parse_args()
    config = read_json(REPO_ROOT / args.config)
    summary = run_stage1_preprocess(REPO_ROOT, Path(args.config), Path(config["inputs"]["raw_roots_config"]), args.dataset)
    print(f"dataset={summary['dataset']}")
    print(f"processed_root={summary['processed_root']}")
    print(f"manifest_raw_count={summary['raw_count']}")
    print(f"manifest_filtered_count={summary['filtered_count']}")
    if "empty_text_removed" in summary:
        print(f"empty_text_removed={summary['empty_text_removed']}")
    if "empty_tag_row_count" in summary:
        print(f"empty_tag_row_count={summary['empty_tag_row_count']}")
    if "concept_subset" in summary:
        print(f"concept_subset={summary['concept_subset']}")
        print(f"concept_positive_counts={summary['concept_positive_counts']}")
    if "final_tag_list_tag_count" in summary:
        print(f"final_tag_list_tag_count={summary['final_tag_list_tag_count']}")
    if "all_tags_row_count" in summary:
        print(f"all_tags_row_count={summary['all_tags_row_count']}")
    if "caption_image_count" in summary:
        print(f"caption_image_count={summary['caption_image_count']}")
    if "instance_image_count" in summary:
        print(f"instance_image_count={summary['instance_image_count']}")
    if "category_count" in summary:
        print(f"category_count={summary['category_count']}")
    if "zero_label_image_count" in summary:
        print(f"zero_label_image_count={summary['zero_label_image_count']}")
    print(f"query_count={summary['query_count']}")
    print(f"retrieval_count={summary['retrieval_count']}")
    print(f"train_count={summary['train_count']}")
    for key, value in summary["order_hashes"].items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

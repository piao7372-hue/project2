# Project Status

Last updated: 2026-04-25

## Formal Specification

- Path: `C:/Users/ASVS/Desktop/1/kan_engineering_original_RA_NUS.pdf`
- Status: only formal engineering specification for NUS raw protocol from Stage 0R-1 onward.
- `configs/stages/stage0_formal.json` declares:
  - `engineering_spec.document`: `kan_engineering_original_RA_NUS.pdf`
  - `engineering_spec.nus_protocol`: `original_ra_nus_image_index_v1`
  - `engineering_spec.portable_path_warning`: true
- `formal_spec_path` remains as a local human reference path, but runners must not depend on it as a hard portable input.

## Current Project Root

- Formal project root: `C:/Users/ASVS/Desktop/New project 2`
- The old `Documents/New project 2` path is not used and does not exist on this machine.

## Current Stage

- Stage: Stage 0F-1
- Scope: read-only Stage 0 final gate and engineering specification consistency check.
- Not in scope: data copy/move/delete, downloads, CLIP download, faiss installation, manifest/split generation, feature extraction, Stage 2 feature cache, semantic matrices, Stage 1+ code, Stage 1+ execution, git commit/push, or Stage 0 total garbage cleanup.

## Stage 0F-1 Products

- Created `scripts/validate_stage0_final_gate.py`.
- Created `src/utils/stage0_final_gate.py`.
- Modified `docs/project_status.md`.
- Generated ignored outputs:
  - `outputs/stage0_final_gate/stage0_final_gate_summary.json`
  - `outputs/stage0_final_gate/stage0_final_gate_summary.md`
  - `outputs/stage0_final_gate/engineering_spec_compliance.json`

## Final Gate Result

- `stage0_environment_lock_passed`: true.
- `stage0_raw_validator_passed`: true.
- `stage0_clip_validation_passed`: true.
- `engineering_spec_compliance_passed`: true.
- `git_ignore_gate_passed`: true.
- `stage1_artifact_absence_passed`: true.
- `stage0_complete`: true.
- `stage1_allowed`: false.
- `stage1_allowed_reason`: `Stage 1 requires explicit user authorization`.
- `blocker_reason`: none.

## Engineering Spec Compliance

- `spec_document`: `kan_engineering_original_RA_NUS.pdf`.
- `nus_protocol`: `original_ra_nus_image_index_v1`.
- `old_kaggle_protocol_used_as_formal`: false.
- `stage`: `Stage 0`.
- `stage1_forbidden`: true.
- `stage2_forbidden`: true.
- `compliance_passed`: true.

## Environment Lock Summary

- `python_version`: `3.9.25` CPython from formal deeplearning environment.
- `torch_version`: `2.5.1+cu121`.
- `torchvision_version`: `0.20.1+cu121`.
- `transformers_version`: `4.57.6`.
- `numpy_version`: `2.0.2`.
- `pillow_version`: `11.3.0`.
- `scipy_version`: `1.13.1`.
- `cuda_available`: true.
- `gpu_name`: `NVIDIA GeForce RTX 4080 Laptop GPU`.
- `faiss_missing`: true.
- `faiss_missing_is_stage0_blocker`: false, because Stage 0 raw/CLIP validation does not install or use faiss.

## Raw Validator Summary

- MIRFlickr25K passed: true; images 25000, tags 25000, exif 25000, annotation txt 38.
- MSCOCO passed: true; train images 82783, val images 40504, total images 123287.
- NUS-WIDE passed: true; protocol `original_ra_nus_image_index_v1`, image_index rows 269648, missing indexed images 0, duplicate raw_index 0, duplicate image_relative_path 0.
- Deprecated NUS Kaggle leftovers remain non-formal leftovers and were not used as formal inputs.

## CLIP Validation Summary

- `backbone_id`: `openai/clip-vit-base-patch32`.
- `model_local_path`: `models/clip/openai_clip-vit-base-patch32`.
- `local_files_only`: true.
- `model_load_ok`: true.
- `processor_load_ok`: true.
- `model_config_projection_dim`: 512.
- `allow_online_download_after_run`: false.

## Git And Artifact Boundary

- Required ignored paths for raw data, processed data, outputs, models, arrays, checkpoints, cache, and logs passed `git check-ignore`.
- Stage 0 source files are not ignored by `.gitignore`.
- `data/processed` does not exist.
- No Stage 1+ artifact names were found: manifest files, split ids, `X_I.npy`, `X_T.npy`, `A.npy`, `R.npy`, `Se.npy`, `C.npy`, or `S.npy`.

## Current Gate

- Stage 0 is complete according to the final gate.
- Stage 1 is still not allowed until explicitly authorized by the user.
- The next housekeeping step, if requested later, can be a Stage 0 total cleanup dry-run plan. It was not performed in Stage 0F-1.

## Stage 1 Status

- Stage 1 status: completed.
- Stage 1 datasets:
  - mirflickr25k: raw=25000, filtered=20015, query=2000, retrieval=18015, train=5000.
  - nuswide: raw=269648, filtered=186577, query=2000, retrieval=184577, train=5000.
  - mscoco: raw=123287, filtered=123287, query=2000, retrieval=121287, train=5000.

## Stage 1 Frozen Artifacts

- `manifest_raw.jsonl`
- `manifest_filtered.jsonl`
- `manifest_meta.json`
- `query_ids.txt`
- `retrieval_ids.txt`
- `train_ids.txt`
- `split_summary.json`
- `preprocess_summary.json`
- `validator_summary.json`
- `config_snapshot.json`
- `order_hashes.json`

## Stage 1 Validators

- `validate_stage1_preprocess.py --dataset mirflickr25k`: passed.
- `validate_stage1_preprocess.py --dataset nuswide`: passed.
- `validate_stage1_preprocess.py --dataset mscoco`: passed.

## Stage 1 Diagnostics

- MIR empty_text_removed=2128.
- NUS empty_tag_row_count=2005.
- NUS concept_subset=sky, clouds, person, water, animal, grass, buildings, window, plants, lake.
- COCO zero_label_image_count=1069.

## Stage 2 Boundary

- Stage 2 may consume Stage 1 manifest/split artifacts.
- Stage 2 must not rewrite sample_id, text_source, label_vector, or split.
- Stage 2 must not bypass Stage 1 by reading raw data directly.

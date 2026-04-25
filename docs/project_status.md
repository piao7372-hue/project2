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

- Stage 0: completed and pushed (`43e3cc1 stage0: finalize environment raw data and clip preparation`).
- Stage 1: completed and pushed (`f0b5d17 stage1: build manifests and freeze splits`).
- Current next stage: Stage 3A MIRFlickr-25K RA-like label-positive revision checkpoint locked; MIR Stage 1/2/3 rerun passed with `lambda_ar_fusion=0.70`, `tau_confidence=0.0075`; pending explicit authorization before Stage 3B.
- Stage 2A status: MIRFlickr-25K completed and validator passed.
- Stage 2B status: NUS-WIDE completed and validator passed.
- Stage 2C status: MSCOCO completed and validator passed.
- Current scope: Stage 3A MIRFlickr-25K RA-like preprocessing revision and semantic relation cache review.
- Not in scope: Stage 3B NUS-WIDE, Stage 3C MSCOCO, Stage 4, or training.

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
- `data/processed` contains ignored runtime artifacts and must not be staged or tracked as source.
- Stage 1 manifest and split artifacts are frozen runtime products under `data/processed`.
- Stage 2 feature cache files remain runtime products and must not be staged or tracked as source.

## Current Gate

- mAP-first execution plan: `docs/map_first_execution_plan.md`.
- Current immediate task: MIR Stage 3A RA-like label-positive preprocessing revision review.
- Stage 1/2 remain frozen unless the Pre-Stage3 readiness audit shows a hard failure.
- Stage 3 candidate selection must use unsupervised `S`-quality criteria first.
- Label-aware metrics are diagnostics only and must not become training
  supervision or the primary tuning target.
- Each dataset Stage 3 gate requires semantic validator, tau/lambda selection
  audit, semantic compatibility audit, and `S_II_star / S_TT_star`
  compatibility audit.
- Stage 4 may start only after MIR/NUS/COCO all have formal Stage 3 `S` locked.
- Stage 3B/3C not started.
- Stage 4 not started.
- Stage 0 is complete and pushed.
- Stage 1 pushed baseline exists; MIRFlickr-25K now has the RA-like label-positive revision checkpoint.
- Stage 2A MIRFlickr-25K is complete locally after the MIR revision and validator passed.
- Stage 2B NUS-WIDE is complete and validator passed.
- Stage 2C MSCOCO is complete and validator passed.
- Stage 3A MIRFlickr-25K semantic matrices completed locally after the MIR revision with `lambda_ar_fusion=0.70`, `tau_confidence=0.0075`, and validator passed.
- Stage 3B NUS-WIDE, Stage 3C MSCOCO, Stage 4, and training are not allowed until explicitly authorized.

## MIR Revision Checkpoint

- MIR Stage 1 revised:
  - filtering_policy = `mir_pragmatic_high_signal_label_positive_v2`
  - filtered = 20015
  - zero_label_filtered_count = 0
  - query_with_no_relevant_retrieval_count = 0
- MIR Stage 2 rerun:
  - clip_i2t_map_at_50 = 0.7806291322
  - clip_t2i_map_at_50 = 0.7872354135
  - validator passed
- MIR Stage 3A locked:
  - lambda_ar_fusion = 0.70
  - tau_confidence = 0.0075
  - compatibility risk_level = low
  - validator passed
- NUS/COCO:
  - preprocessing consistency audited
  - keep frozen
  - Stage 3B/3C not started
  - Stage 4 not started

## Stage 2 Status

- Stage 2A / MIRFlickr-25K: rerun after MIR RA-like label-positive Stage 1 revision; `validate_stage2_features.py --dataset mirflickr25k` passed.
- Stage 2B / NUS-WIDE: completed; `validate_stage2_features.py --dataset nuswide` passed.
- Stage 2C / MSCOCO: completed; `validate_stage2_features.py --dataset mscoco` passed.
- Stage 2 feature_set_id: `clip_vit_b32_formal_v1`.
- MIR feature shapes: `X_I=(20015, 512)`, `X_T=(20015, 512)`, dtype `float32`.
- NUS feature shapes: `X_I=(186577, 512)`, `X_T=(186577, 512)`, dtype `float32`.
- COCO feature shapes: `X_I=(123287, 512)`, `X_T=(123287, 512)`, dtype `float32`.
- Stage 3A / MIRFlickr-25K semantic_set_id: `se_c_s_formal_v1`; rerun after MIR RA-like label-positive Stage 1/2 revision with `lambda_ar_fusion=0.70`, `tau_confidence=0.0075`, completed locally, and `validate_stage3_semantic.py --dataset mirflickr25k` passed.
- Stage 3B / NUS-WIDE: not started.
- Stage 3C / MSCOCO: not started.
- Stage 4, model construction, and training: not started.

## Stage 3A MIRFlickr-25K Status

- Stage 3A / MIRFlickr-25K status: RA-like label-positive Stage 1 revision, Stage 2 rerun, and Stage 3 formal rerun completed locally with `lambda_ar_fusion=0.70`, `tau_confidence=0.0075`; pending review before Stage 3B.
- Stage 3A output dir: `data/processed/mirflickr25k/semantic_cache/se_c_s_formal_v1/`.
- Stage 3A formal matrices: `A.npy`, `R.npy`, `Se.npy`, `C.npy`, `S.npy`; all are ignored runtime artifacts and must not be staged.
- Stage 3A validator: `validate_stage3_semantic.py --dataset mirflickr25k` passed.
- MIR profile used: `lambda_ar_fusion=0.70`, `tau_confidence=0.0075`, `topk_for_diagnostics=50`.
- Stage 3A stop/go: passed for MIR only. This does not mark NUS-WIDE, MSCOCO, Stage 4, or training as completed.
- Stage 3A tau sensitivity audit first selected `tau_confidence=0.01`; the
  later MIR candidate decision audit selected `lambda_ar_fusion=0.70`,
  `tau_confidence=0.0075` because it passed the hard support, topology, and
  label-aware disaster-exclusion gates.
- Stage 3A final MIR diagnostics after RA-like revision:
  `diag_mean_s=0.05691593816830318`,
  `offdiag_mean_s=0.00011715354766653248`,
  `diag_over_offdiag_ratio=485.8234283293709`,
  `diag_in_row_topk_rate=0.8572`, `diag_in_col_topk_rate=0.8502`,
  `row_topk_coverage=0.9998`, `col_topk_coverage=1.0`.
- Do not apply second normalization, min-max scaling, top-k masking, identity
  boosting, or artificial amplification to `S`.

## Global RA comparison protocol lock

Final comparison must follow the RA/RANEH Table 1 protocol.

Required hash code lengths:
- 16 bits
- 32 bits
- 64 bits
- 128 bits

Required retrieval directions:
- Image-to-Text, I->T
- Text-to-Image, T->I

Required final metrics:
- mAP@50
- Top-R precision

Formal final retrieval must use binary hash codes `B_I` and `B_T` with Hamming
distance as the primary ranking criterion. Continuous `H_I/H_T` cosine ranking
is allowed only for diagnostics and must not be reported as the formal final
result.

Later Stage 4/6/7 configs must explicitly support all four hash code lengths
`[16, 32, 64, 128]`. Do not implement only one bit length, do not report only
128-bit, and do not use continuous CLIP or continuous hash features as a
substitute for final binary hash retrieval.

## Stage 1 Status

- Stage 1 status: completed; MIRFlickr-25K uses the RA-like label-positive revision checkpoint.
- Stage 1 datasets:
  - mirflickr25k: raw=25000, filtered=20015, query=2000, retrieval=18015, train=5000; filtering_policy=`mir_pragmatic_high_signal_label_positive_v2`.
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
- MIR RA-like label-positive revision: candidate_5 selected; zero_label_filtered_count=0, zero_label_query_count=0, zero_label_train_count=0, zero_label_retrieval_count=0, query_with_no_relevant_retrieval_count=0.
- NUS empty_tag_row_count=2005.
- NUS concept_subset=sky, clouds, person, water, animal, grass, buildings, window, plants, lake.
- COCO zero_label_image_count=1069.

## Stage 2 Boundary

- Stage 2 may consume Stage 1 manifest/split artifacts.
- Stage 2 must not rewrite sample_id, text_source, label_vector, or split.
- Stage 2 must not bypass Stage 1 by reading raw data directly.

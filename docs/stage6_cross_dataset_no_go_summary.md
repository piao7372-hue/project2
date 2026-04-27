# Stage6 Cross-Dataset No-Go Summary

Status: research record only. This is not a formal Stage7 evaluation result.

## 1. MIR

- best exploratory I2T = 0.8519076634
- best exploratory T2I = 0.8348295490
- mean = 0.8433686062
- RA/RANEH MIR 128-bit target = 0.961 / 0.926
- MIR gap large
- MIR No-Go recorded in `docs/mir_experimental_no_go_log.md`

## 2. COCO

- Stage2 CLIP I2T = 0.9130829416
- Stage2 CLIP T2I = 0.9165909334
- best Stage6 smoke = backup
- backup I2T = 0.7361261851
- backup T2I = 0.7735027700
- backup mean = 0.7548144775
- Stage6 far below Stage2 baseline
- COCO No-Go recorded in `docs/coco_stage6_smoke_no_go_log.md`

## 3. NUS

- Stage2 CLIP I2T = 0.6903989647
- Stage2 CLIP T2I = 0.6751644357
- best Stage6 smoke = backup
- backup I2T = 0.6805707170
- backup T2I = 0.6410526693
- backup mean = 0.6608116931
- Stage6 below Stage2 baseline
- NUS No-Go documented in `docs/nus_stage6_smoke_no_go_log.md`

## 4. Cross-dataset pattern

- Current Stage6 mainline fails across MIR / COCO / NUS.
- Repeated pattern:
  - `L_pair` decreases strongly
  - `L_sem` increases or does not improve
  - bit health stays stable
  - Hamming mAP is weak
- This suggests global hash learning / objective contract issue, not a dataset-specific bit-collapse issue.
- Do not continue current Stage6 mainline.
- Do not run Stage7.
- Do not expand 16/32/64.
- Next work should be:
  - formal hash learning objective redesign, or
  - RANEH-faithful branch planning / reproduction-style baseline, or
  - paper-level theory/spec correction.

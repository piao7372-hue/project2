# mAP-First Execution Plan

This document is an execution roadmap for later Codex turns. It does not
replace `docs/specs/engineering_spec.md`; the engineering spec remains the
primary implementation contract.

## Final Goal

Obtain strong final binary-hash retrieval results comparable to RA/RANEH
Table 1 under the same experimental protocol.

Required datasets:
- MIRFlickr-25K
- NUS-WIDE
- MSCOCO

Required hash code lengths:
- 16 bits
- 32 bits
- 64 bits
- 128 bits

Required retrieval directions:
- Image-to-Text, I→T
- Text-to-Image, T→I

Required formal metrics:
- mAP@50
- Top-R precision

Formal retrieval:
- Use binary hash codes `B_I` and `B_T`.
- Use Hamming distance as the primary ranking criterion.
- Continuous `H_I/H_T` cosine ranking is diagnostic only.
- CLIP cosine baseline is diagnostic only.

## Stage 2 Gate

- CLIP feature cache must be frozen.
- `paired_cosine_mean > random_cosine_mean`.
- `paired_cosine_median > random_cosine_median`.
- `baseline_summary` completed.
- No rewriting manifest, split, text, or label fields.

## Stage 3 Gate

- `S` is built only on train split, `N=5000`.
- `S = C * Se` is the only formal cross-modal supervision target.
- No full filtered `S`.
- No query/retrieval contamination.
- No `S` rescaling, min-max scaling, top-k masking, or identity injection.

Each dataset must pass:
- semantic validator
- tau/lambda sensitivity or selection audit
- semantic compatibility audit
- `S_II_star / S_TT_star` compatibility check

Stage 1/2 artifacts are frozen unless a readiness audit shows a hard failure.
Stage 3 candidate selection must use unsupervised `S`-quality criteria first.
Label-aware metrics are diagnostics only; they must not become training
supervision or the primary tuning target.

Current Stage 3 acceptance summary:
- Formal cleanup/review summary: `docs/stage3_acceptance_summary.md`.
- MIR Stage 3A, NUS Stage 3B, and COCO Stage 3C are locked locally.
- Stage 4 is not started and still requires explicit user authorization.

NUS-WIDE Stage 3B override note:
- The default NUS profile caused a Stage 3 No-Go because `S = C * Se` remained
  too uniform and low-rank in the initial candidate range.
- A No-Go recovery root-cause and extended lambda/tau audit found a low-risk
  diagnostic candidate: `lambda_ar_fusion=0.55`, `tau_confidence=0.005`.
- The user-approved NUS formal profile is therefore the recovery override above.
- The Stage 3 formula remains unchanged; no label supervision is used to
  construct `S`.
- `Omega_topk_diag.npz` remains diagnostic only.
- This override applies only to NUS Stage 3B and does not authorize Stage 3C,
  Stage 4, training, or evaluation.
- The NUS formal validator passed under the override, but the first extra
  safety audit was No-Go because 31 empty-text samples entered train
  supervision.
- The active NUS recovery policy is `nus_train_nonempty_text_v2`: keep the
  filtered/query/retrieval counts and IDs unchanged, do not alter
  `text_source`, do not pad empty text, and select the first 5000 retrieval
  samples with non-empty `text_source` as train.
- After this Stage 1 split revision, NUS Stage 2 and Stage 3B were rerun with
  the unchanged Stage 3 formula and the selected profile
  `lambda_ar_fusion=0.55`, `tau_confidence=0.005`.
- NUS Stage 3B validator passed; compatibility risk is low; empty-text risk is
  resolved; hubness and induced topology risks are low.
- NUS concept fairness was recalibrated using prevalence-aware thresholds and
  integrated into the NUS safety audit.
  The previous medium risk was mainly caused by applying a uniform
  `lift>=2.0` gate to high-frequency concepts `sky` and `clouds`.
- Low-frequency concepts `grass`, `buildings`, `window`, `plants`, and `lake`
  show strong lift and diagonal top-k behavior after calibration.
- NUS Stage 3B formal `S` is locked under prevalence-aware fairness
  calibration: validator passed, compatibility risk is low, safety risk is
  low, and `nus_stage3_s_good_for_stage4_and_stage5=true`.
- NUS Stage 3B remains acceptable for later Stage 4/5 preparation; this note
  does not authorize Stage 4, training, or evaluation.

For each dataset, Stage 3 must pass all of:

1. semantic validator
2. tau/lambda selection audit
3. semantic compatibility audit
4. `S_II_star / S_TT_star` compatibility audit

Stage 4 may start only after all three datasets have formal Stage 3 `S`
locked.

## Stage 4 Gate

- Implement ChebyKAN -> recursive semantic tree -> graph refinement -> hash head.
- Must support `hash_bits = [16, 32, 64, 128]`.
- No CLIP backbone training.
- No raw data reading.
- No `S` regeneration.

## Stage 5 Gate

Loss must follow `docs/specs/theory_loss_function.md`:
- `L_sem`
- `L_pair`
- `L_q`
- `L_bal`

`S_II_star` and `S_TT_star` must be induced from `S`, not directly replaced by
`S`. Perform loss-scale audit before long training. Track
`beta_relation_weight` effect because `S` values may be small.

Stage 5D adds a loss-weight sensitivity audit for Stage 6 dev-only candidate
profiles. No training is performed, no final Stage 6 profile is selected, and
the default Stage 5 audit profiles remain unchanged.

Stage 5 is completed locally. Acceptance summary:
`docs/stage5_acceptance_summary.md`. Stage 6 is not started. Stage 7 is not
started. Stage 6 dev training must use the Stage 5D recommendations as
candidate profiles only, not as final training profiles.

MIR v2 multi-target supervision contract draft is under review:
`docs/mir_multitarget_supervision_v2_contract_draft.md`. No formal
`docs/specs/*` specification was modified for this draft. No Stage 6 training
has started from this draft. No training was performed for this draft.

## Stage 6 Gate

Use dev mode first. Start with 128-bit sanity training per dataset.

Track:
- total loss
- `L_sem / L_pair / L_q / L_bal`
- bit balance
- constant bit ratio
- unique code ratio
- train retrieval proxy mAP
- validation/query retrieval mAP if available

Do not proceed to `formal_report` until dev mode is stable.

## Stage 7 Gate

Final report must include all:

`MIR/NUS/COCO × 16/32/64/128 × I→T/T→I`.

Use Hamming distance on `B_I/B_T`. Report mAP@50 and Top-R precision. Compare
against RA/RANEH table under the same bit lengths.

## Execution Order

1. Finish MIR Stage 3A semantic compatibility and candidate selection.
2. If a Pre-Stage3 readiness audit finds a hard Stage 1 relevance failure,
   revise only the affected dataset before later stages. MIRFlickr-25K uses
   `mir_pragmatic_high_signal_label_positive_v2` locally after the zero-label
   query audit.
3. If needed, revise MIR formal Stage 3A.
4. Review the NUS Stage 3B recovery outcome:
   `nus_train_nonempty_text_v2`, NUS-only Stage 1/2 rerun, Stage 3 lambda/tau
   selection, formal Stage 3B, compatibility audit, extra safety audit, and
   prevalence-aware concept fairness calibration.
5. Keep NUS Stage 3B locked.
6. COCO Stage 3C is locked locally with `lambda_ar_fusion=0.80` and
   `tau_confidence=0.01`.
7. Stage 3 full cleanup and validation.
8. Commit or push only with explicit user authorization.
9. Stage 4 model implementation.
10. Stage 5 loss implementation and loss-scale audit.
11. Stage 6 dev training, starting from 128-bit.
12. Expand to 16/32/64/128 bits.
13. Stage 7 formal Hamming evaluation.
14. Freeze formal_report profile and rerun final experiments.

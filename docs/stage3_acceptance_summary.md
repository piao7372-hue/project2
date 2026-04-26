# Stage 3 Acceptance Summary

Stage 3 overall status: completed locally, ready for cleanup/review before commit.

## MIR

- filtering_policy = mir_pragmatic_high_signal_label_positive_v2
- Stage 2 rerun baseline: I2T 0.7806291322, T2I 0.7872354135
- Stage 3 profile: lambda_ar_fusion = 0.70, tau_confidence = 0.0075
- Stage 3 validator: passed
- compatibility risk_level: low

## NUS

- train_selection_policy = nus_train_nonempty_text_v2
- empty_text_train_count = 0
- Stage 3 profile: lambda_ar_fusion = 0.55, tau_confidence = 0.005
- Stage 3 validator: passed
- compatibility risk_level: low
- safety risk_level: low
- prevalence-aware concept fairness: low_or_acceptable

## COCO

- Stage 3 profile: lambda_ar_fusion = 0.80, tau_confidence = 0.01
- Stage 3 validator: passed
- compatibility risk_level: low
- safety/evaluator risk_level: medium
- reason: zero_label_query_count = 11, query_with_no_relevant_retrieval_count = 11
- estimated mAP loss from no-relevant queries ≈ 0.0055
- note: evaluator risk belongs to Stage 7 all-query / valid-query diagnostic handling, not Stage 3 failure.

## Global

- Stage 3 formal supervision target remains S = C ⊙ Se.
- Omega_topk_diag is diagnostic only.
- No label supervision was used to construct S.
- Stage 4 not started.

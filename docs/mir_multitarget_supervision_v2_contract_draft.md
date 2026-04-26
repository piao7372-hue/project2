# MIR v2 Multi-Target Supervision Contract Draft

Status: draft only, under human review.

This document is not a formal Stage 3, Stage 5, or Stage 6 specification. It
does not authorize training, mAP evaluation, NUS/COCO v2 execution, cache
generation, or formal spec changes. The existing Stage 4 v1 model chain remains
unchanged for the first MIR slice.

## 1. Scope And Fixed Decisions

Primary contract: Contract A, multi-target Stage 3 supervision.

Execution policy: Contract C-style curriculum.

First model slice: keep Stage 4 v1 unchanged.

First dataset scope: MIRFlickr-25K only.

First bit scope after approval: 128-bit dev probe only.

Forbidden until MIR passes R0-R3: NUS-WIDE v2 and MSCOCO v2.

No-label boundary:

- No `label_vector` may be used to construct any training target.
- Query/retrieval samples and labels are forbidden in target construction.
- Raw image/text files are forbidden in target construction.
- CLIP must not be loaded while building v2 supervision; only frozen Stage 2
  train features may be read.
- Label-aware metrics remain diagnostic-only and cannot select targets.

Current evidence motivating this draft:

- Current best exploratory MIR 128-bit: I2T = 0.8519076634,
  T2I = 0.8348295490, mean = 0.8433686062.
- RA/RANEH MIR 128-bit reference: I2T = 0.961, T2I = 0.926.
- Stage 5/6 diagnostics indicate pair/semantic conflict.
- Image-teacher targets are strong, but direct teacher training probes reduced
  mAP, so teacher information must be constrained by topology, confidence, and
  curriculum rather than used as a dense loss from epoch 0.

## 2. V2 Supervision Objects

All objects below are defined on the MIR train split only, with `N = 5000`.
They are proposal objects for a later v2 package; no files are created by this
draft.

### 2.1 Pair Identity

```text
S_pair_identity = I_N
```

Purpose:

- Used only as a paired diagonal identity anchor.
- Handled by `L_pair` or an explicit identity branch.
- Not mixed with offdiagonal semantic regression.
- Does not use labels, raw data, query/retrieval data, or CLIP loading.

Contract meaning: the original paired image/text sample at index `i` should
remain aligned. This object carries identity alignment only, not semantic
similarity between different samples.

### 2.2 Offdiag Semantic Relation

```text
S_sem_offdiag = S * (1 - I_N)
```

Where current v1 semantic relation is retained:

```text
S = C * Se
```

Meaning:

- `S` remains the no-label cross-modal semantic relation produced from Stage 2
  train features and Stage 3 confidence calibration.
- The diagonal is excluded from the semantic branch.
- `S_sem_offdiag` no longer handles paired diagonal alignment.
- Offdiagonal semantic supervision should be ranking/top-k based, not dense
  raw-`S` diagonal regression.

### 2.3 Confidence Weights

```text
W_sem = C * M_sem
```

Where:

```text
M_sem = high-confidence no-label offdiag mask
```

Meaning:

- `C` is a reliability/confidence weight.
- `C` is not the sole mechanism for reducing target amplitude toward near-zero.
- `M_sem` must be constructed without labels, using only train-split no-label
  semantic confidence and topology criteria.
- `M_sem` must exclude diagonal pairs unless a future approved branch gives a
  separate reason.

Proposed policy:

- Use `W_sem` to focus rank edges on reliable offdiagonal relations.
- Keep weak or ambiguous offdiagonal relations available for diagnostics, but
  do not force dense regression on all weak entries.

### 2.4 Image Teacher

```text
T_img = (1 + X_I X_I^T) / 2
G_img_teacher = mutual_topk_or_rank_topology(T_img)
```

Requirements:

- `X_I` comes only from frozen Stage 2 train image features.
- No `label_vector`.
- No raw images.
- No raw text.
- No CLIP model loading.
- No query/retrieval features.
- No replacement of Stage 3 `S`.

Meaning:

- `T_img` is an image-side topology prior derived from frozen image features.
- `G_img_teacher` is a sparse or rank-based teacher topology, not a dense target
  that must be fully regressed from epoch 0.
- `G_img_teacher` may supervise `P_II` and weakly transfer to `P_TT` by paired
  train index, but it must not become a cross-modal semantic truth matrix.

### 2.5 Ranking Edges

```text
E_rank = rank_edges(S_sem_offdiag, W_sem)
```

Requirements:

- Edges are no-label.
- `label_vector` is diagnostic only.
- Query/retrieval labels are forbidden.
- The diagonal is excluded.
- Edges must record source object IDs and threshold/top-k policy in metadata.

Meaning:

- `E_rank` converts reliable offdiagonal semantic evidence into ordering
  constraints.
- This avoids treating every raw `S` entry as an equally meaningful dense
  regression target.

## 3. Stage 5 v2 Loss Draft

Proposed total loss:

```text
L_total =
  lambda_pair L_pair
+ lambda_rank L_cross_rank
+ lambda_img L_img_topology
+ rho_txt lambda_img L_txt_topology
+ lambda_q L_q
+ lambda_bal L_bal
```

All terms use continuous `H_I` and `H_T`. No training loss may use `B_I`,
`B_T`, `sign`, or `label_vector`.

### 3.1 L_pair

Role: paired identity anchor.

Definition:

- Aligns only `H_I[i]` with `H_T[i]`.
- Represents `S_pair_identity = I_N`.
- Does not consume offdiagonal semantic targets.
- Does not use `label_vector`.
- Does not use `B` or `sign`.

Reason: paired identity is a separate anchor. It should not be hidden inside
dense `S` regression, where diagonal and offdiagonal objectives can conflict.

### 3.2 L_cross_rank

Role: offdiagonal cross-modal semantic ranking.

Inputs:

- `E_rank`
- `S_sem_offdiag`
- `W_sem`

Definition:

- The diagonal is excluded.
- The loss may be pairwise rank, listwise rank, or top-k contrastive ranking,
  but it must be specified before implementation.
- It is not dense raw-`S` diagonal regression.
- Ranking positives and negatives must be constructed from no-label train
  objects only.

Reason: Stage 3 `S` is weak in absolute amplitude, but still carries
offdiagonal ordering information. Ranking uses that ordering without forcing
all low-confidence amplitudes to dominate optimization.

### 3.3 L_img_topology

Role: image-side topology distillation.

Input:

- `G_img_teacher`

Target:

- `P_II`, the image-image relation predicted from continuous image hashes.

Constraints:

- No labels.
- No raw data.
- No dense teacher loss from epoch 0.
- Teacher topology should be sparse/rank-based and confidence controlled.

Reason: image topology is the strongest observed teacher signal, but direct
dense training failed. This term must preserve topology without replacing the
cross-modal relation contract.

### 3.4 L_txt_topology

Role: small-weight text-side topology transfer.

Input:

- `G_img_teacher`, transferred to text by paired train index.

Target:

- `P_TT`, the text-text relation predicted from continuous text hashes.

Weight:

```text
rho_txt * lambda_img
```

Rules:

- `rho_txt` must be small.
- `rho_txt` must be dataset-specific.
- MIR v2 must start conservatively because text-text proxy is weak.
- The text branch must not be forced to fully imitate image topology
  immediately.

### 3.5 L_q And L_bal

Role: unchanged Stage 5 v1 quantization and bit-balance terms.

Rules:

- Use continuous `H`.
- Do not use `B` or `sign` in training loss.
- `L_q` and `L_bal` may be weak in early phases and raised during binary polish.

## 4. Curriculum Policy

The v2 contract must use curriculum scheduling. Direct teacher or semantic
dense loss from epoch 0 is forbidden.

### Phase 1: Pair Bootstrap

Active:

- `L_pair`
- weak `L_q`
- weak `L_bal`

Purpose:

- Establish stable paired identity alignment.
- Avoid early teacher overconstraint.
- Avoid early bit collapse.

### Phase 2: Image Topology Ramp

Active:

- Phase 1 terms
- `ramp(lambda_img) L_img_topology`

Purpose:

- Introduce image topology after identity alignment is stable.
- Avoid dense teacher domination from epoch 0.

### Phase 3: Cross Semantic Ranking Ramp

Active:

- Phase 1 and Phase 2 terms
- `ramp(lambda_rank) L_cross_rank`
- small `rho_txt L_txt_topology`

Purpose:

- Bring in offdiagonal no-label semantic ranking.
- Transfer only limited image topology to the weak text side.

### Phase 4: Binary Polish

Active:

- All target terms at approved dev weights.
- `L_q` and `L_bal` raised to target values.

Purpose:

- Reduce quantization gap after semantic and topology structure have stabilized.
- Preserve bit balance and avoid collapse before diagnostic Hamming checks.

Logging requirement:

- All effective lambdas must be logged per epoch.
- Phase boundaries, ramp values, and active loss terms must be recorded.
- If a loss term is inactive, its effective lambda must be logged as zero.

Why direct teacher loss failed before:

- The teacher target was strong as a topology signal, but direct probes forced
  the model to match the teacher distribution too early.
- The weaker text branch was asked to imitate image topology before paired and
  semantic alignment were stable.
- Dense teacher matching can dominate offdiagonal semantic ranking and paired
  identity alignment.

## 5. Proposed Cache And Artifact Schema

Proposed path:

```text
data/processed/mirflickr25k/semantic_cache/multitarget_v2_mir/
```

This path is proposed only. No files are created in this turn.

Proposed artifacts:

```text
S_pair_identity.npy or implicit_identity_flag in contract_meta.json
S_sem_offdiag.npy
W_sem_confidence.npy
G_img_teacher.npy or sparse G_img_teacher.npz
E_rank.npz
contract_meta.json
validator_summary.json
```

Required `contract_meta.json` fields:

```text
dataset = mirflickr25k
train_count = 5000
source_feature_set_id = clip_vit_b32_formal_v1
source_semantic_set_id = se_c_s_formal_v1
source_stage4_model_id = chebykan_tree_graph_hash_v1
stage4_first_slice_unchanged = true
no_label_contract = true
label_vector_used_for_target = false
query_retrieval_used_for_target = false
raw_used = false
clip_loaded = false
created_from_train_split_only = true
allowed_datasets_initial = ["mirflickr25k"]
nuswide_v2_forbidden_until_mir_passes = true
mscoco_v2_forbidden_until_mir_passes = true
formal_spec_modified = false
training_performed = false
map_evaluated = false
```

No-label guarantees:

- Target construction may read only frozen train-split Stage 2 features, Stage 3
  `S/C/Se`-related train artifacts, order/hash metadata, and approved configs.
- Any appearance of `label_vector` in target construction is a hard failure.
- Any query/retrieval target construction input is a hard failure.

## 6. Validator Gates

### 6.1 No-Label Scanner

Checks:

- `label_vector` is not used in target construction.
- Query/retrieval samples are not used.
- Raw image/text files are not read.
- CLIP is not loaded.
- Stage 1, Stage 2, and Stage 3 order hashes match.
- `train_count = 5000`.
- Dataset is exactly `mirflickr25k`.

Failure: any violation is a hard No-Go.

### 6.2 Object Health

Checks:

- Dense object shape is `[5000, 5000]`; sparse objects must have equivalent
  declared shape.
- All values finite.
- Dense weights and relations in `[0, 1]`.
- `S_pair_identity` diagonal is one and offdiagonal zero, or identity is
  declared implicit in metadata.
- `S_sem_offdiag` diagonal is zero.
- `G_img_teacher` diagonal policy is explicit.
- Top-k/rank edges are nonempty.
- Coverage is above the approved MIR threshold.
- Hubness is not high.
- `W_sem` has nonzero support and does not collapse to all near-zero weights.

Failure: malformed shape, NaN/Inf, range violation, diagonal-policy violation,
empty top-k, high hubness, or collapsed confidence support is a hard No-Go.

### 6.3 Loss Synthetic Gate

Checks:

- Blockwise equals dense on synthetic data.
- Gradients are finite.
- No `B` or `sign` in training loss.
- No label input.
- No loss component dominates beyond an approved cap.
- Effective lambdas respect phase schedule.
- Deactivating a phase sets its effective lambda to zero.

Failure: non-equivalence, non-finite gradients, forbidden input, forbidden
discrete operation, or dominating loss component is a hard No-Go.

### 6.4 MIR Dev Gate

Diagnostic Hamming mode:

```text
diagnostic_only = true
not_formal_stage7_result = true
ranking = hamming
uses_model_binary_codes = true
```

MIR 128-bit dev gate:

```text
hard survival:
  no direction worse than current best by > 0.005

go:
  mean >= 0.858
  both directions improve

strong-go:
  mean >= 0.870
  T2I not lagging
```

Current best baseline for this gate:

```text
I2T = 0.8519076634
T2I = 0.8348295490
mean = 0.8433686062
```

Formal RA survival gate remains:

```text
eventually I2T >= 0.90 and T2I >= 0.88 before any final-claim language
```

## 7. MIR-Only Allowlist

Initial v2 allowed datasets:

```text
["mirflickr25k"]
```

Forbidden until MIR passes R0-R3:

```text
nuswide
mscoco
```

Reason:

- Avoid breaking existing NUS/COCO v1.
- MIR is the current bottleneck.
- v2 contract is not yet proven.
- NUS and COCO have different topology and evaluator risks that must not be
  silently inherited from MIR.

## 8. Required Future Spec Deltas

These are proposed future edits only. This draft does not modify the formal
`docs/specs/*` files.

| File | Section to update | New concepts | Old statement to supersede | Breaking change | Migration note |
| --- | --- | --- | --- | --- | --- |
| `docs/specs/theory_semantic_relation.md` | Final supervision object / relation output | `S_pair_identity`, `S_sem_offdiag`, `W_sem`, `E_rank` as auxiliary v2 package objects while retaining `S = C * Se` | "Stage 3 outputs one final supervision matrix S" as the complete downstream object | Yes for v2 only | Keep v1 `se_c_s_formal_v1`; add v2 cache ID, no compatibility alias |
| `docs/specs/theory_post_semantic_method.md` | Post-semantic method input boundary | Optional v2 supervision package, Stage 4 v1 unchanged for first slice, teacher topology not part of model architecture initially | "Given fixed final S, learn H_I/H_T" as the sole downstream input statement | Partial | First MIR slice consumes v2 loss objects without modifying Stage 4 v1 |
| `docs/specs/theory_loss_function.md` | Loss objective and same-modal topology | `L_cross_rank`, `L_img_topology`, `L_txt_topology`, curriculum effective lambdas | `L_sem = L_IT + alpha/2(L_II + L_TT)` as the only relation loss | Yes for v2 only | Keep Stage5 v1 loss path and add explicit v2 loss ID |
| `docs/specs/engineering_spec.md` | Stage 3, Stage 5, Stage 6 gates | MIR-only v2 supervision package schema, no-label scanner, synthetic gate, 128-bit dev gate, NUS/COCO forbid rule | Stage 3 freezes one `S`; Stage 5 derives all topology from `S`; Stage 6 starts from Stage5D candidates | Yes for MIR v2 track | Add staged migration R0-R5; do not replace v1 until R1-R3 pass |
| `docs/map_first_execution_plan.md` | Execution order and gates | R0-R5 MIR v2 review path, NUS/COCO compatibility audit after MIR pass | Direct Stage 6 dev training from Stage5D candidates | Yes for immediate roadmap | Stage5D candidates remain v1 dev-only; v2 requires separate approval |

## 9. Roadmap After Approval

### R0: Spec-Only Review

Allowed files:

- `docs/mir_multitarget_supervision_v2_contract_draft.md`
- optional status notes in `docs/project_status.md`
- optional status notes in `docs/map_first_execution_plan.md`

Outputs:

- Reviewed draft.
- Human decision: revise, approve R1, or reject.

Success gate:

- Human accepts object definitions, no-label boundary, validators, and roadmap.

No-go condition:

- Any unresolved ambiguity about label use, dataset scope, Stage 4 changes, or
  Stage 6 training permission.

### R1: MIR v2 Supervision Package Builder And Validator

Allowed files:

- New MIR-only builder under `scripts/` after approval.
- New validator under `scripts/` and/or `src/datasets/validators/` after
  approval.
- MIR-only config draft after approval.

Outputs:

- Proposed v2 package under
  `data/processed/mirflickr25k/semantic_cache/multitarget_v2_mir/`.
- `contract_meta.json`.
- `validator_summary.json`.

Success gate:

- No-label scanner passes.
- Object health passes.
- Stage 1/2/3 hashes match.
- No NUS/COCO execution.

No-go condition:

- Any target uses labels, query/retrieval, raw, CLIP loading, or non-MIR data.
- v2 package does not beat v1 `S` proxy by a meaningful pre-approved margin.

### R2: Stage 5 v2 Synthetic Loss Audit

Allowed files:

- New v2 loss module and synthetic smoke only after approval.
- New validator/smoke script only after approval.

Outputs:

- Synthetic-only audit report.
- Dense/blockwise equivalence report.
- Gradient finite report.
- Loss-ratio and component dominance report.

Success gate:

- Blockwise equals dense.
- Gradients finite.
- No `B/sign`.
- No label.
- No loss component dominates beyond cap.

No-go condition:

- Any forbidden input or operation appears.
- Loss components are unstable before real data.

### R3: MIR 128-Bit Dev Probe

Allowed files:

- Stage6 dev runner changes only after approval.
- MIR-only run config only after approval.

Outputs:

- MIR 128-bit dev diagnostics.
- Hamming diagnostic report marked not formal Stage7.
- Epoch effective-lambda logs.

Success gate:

- Hard survival: no direction worse than current best by more than `0.005`.
- Go: mean `>= 0.858` and both directions improve.
- Strong-go: mean `>= 0.870` and T2I not lagging.

No-go condition:

- MIR 128-bit dev does not beat current best by at least `+0.015` mean.
- T2I drops while I2T improves.
- Teacher topology improves proxy but lowers Hamming mAP.
- Gradient conflict remains strongly negative.

### R4: Formal Spec Update If R1-R3 Pass

Allowed files:

- `docs/specs/theory_semantic_relation.md`
- `docs/specs/theory_post_semantic_method.md`
- `docs/specs/theory_loss_function.md`
- `docs/specs/engineering_spec.md`
- relevant configs
- roadmap/status docs

Outputs:

- Formal v2 specs.
- Versioned config IDs.
- Migration notes.

Success gate:

- Specs and configs align with proven MIR R1-R3 behavior.
- v1 remains auditable.

No-go condition:

- R1-R3 evidence is incomplete or mixed.
- Formal spec edits would silently change NUS/COCO behavior.

### R5: NUS/COCO Compatibility Audit

Allowed files:

- Audit scripts/configs only after R4 approval.
- No NUS/COCO v2 package generation until audit approval.

Outputs:

- Read-only compatibility report for NUS and COCO.
- Hubness, coverage, text-risk, evaluator-risk assessment.

Success gate:

- No v1 regression.
- Dataset-specific risks are documented before any v2 execution.

No-go condition:

- NUS/COCO v1 regression breaks during MIR-only work.
- MIR assumptions do not transfer safely.

## 10. Risk Register

### Teacher Overconstraint Risk

Risk: `G_img_teacher` may dominate pair and semantic objectives.

Mitigation:

- Sparse/rank teacher only.
- No full dense teacher loss from epoch 0.
- Ramp `lambda_img`.

Validator/audit signal:

- Loss component ratio.
- Gradient conflict.
- Hamming direction split.

Abort condition:

- Teacher proxy improves while Hamming mAP drops.

### Text-Side Topology Risk

Risk: weak text branch may be forced to imitate image topology.

Mitigation:

- Small dataset-specific `rho_txt`.
- Introduce text topology only after pair and image topology phases.

Validator/audit signal:

- T2I mAP.
- `P_TT` topology health.
- Gradient norm from `L_txt_topology`.

Abort condition:

- T2I drops while I2T improves.

### Hubness Risk

Risk: top-k teacher or semantic edges may create high-degree hubs.

Mitigation:

- Mutual top-k or rank-balanced construction.
- Hubness diagnostics in validator.

Validator/audit signal:

- degree p95/p99.
- degree max-over-mean.
- degree gini.

Abort condition:

- Hubness risk becomes high or coverage collapses.

### Schedule/Ramp Risk

Risk: phase boundaries or ramps hide instability.

Mitigation:

- Log effective lambdas per epoch.
- Require ablation checkpoints.

Validator/audit signal:

- Loss spikes at phase transition.
- Non-finite gradients.
- bit collapse after ramp.

Abort condition:

- Any phase transition causes unrecovered collapse.

### MIR-Only Overfit Risk

Risk: v2 improves MIR but encodes MIR-specific assumptions.

Mitigation:

- Keep NUS/COCO forbidden until R0-R3 pass.
- Require R5 read-only compatibility audit.

Validator/audit signal:

- MIR-specific thresholds or hard-coded policies.
- NUS/COCO v1 regression.

Abort condition:

- MIR v2 requires assumptions incompatible with NUS/COCO v1 contracts.

### NUS/COCO Blast-Radius Risk

Risk: v2 changes shared code/config and breaks existing v1 datasets.

Mitigation:

- Versioned IDs.
- MIR-only allowlist.
- No compatibility aliases.

Validator/audit signal:

- `git ls-files` boundaries.
- unchanged v1 validators.
- explicit dataset allowlist checks.

Abort condition:

- NUS/COCO v1 regression breaks during MIR-only work.

### RA Gap Not Guaranteed

Risk: even a clean v2 may not close the MIR RA gap.

Mitigation:

- Hard numeric dev gates.
- Stop before formal claims.
- Treat RA survival gate separately from dev improvement gate.

Validator/audit signal:

- 128-bit Hamming diagnostic mAP.
- direction balance.
- quantization and bit-health diagnostics.

Abort condition:

- v2 fails to beat current best by at least `+0.015` mean on MIR 128-bit dev.

## 11. No-Go Rules

- No-go if any target uses `label_vector`.
- No-go if query/retrieval enters target construction.
- No-go if raw data enters target construction.
- No-go if CLIP is loaded to build v2 supervision.
- No-go if v2 package does not beat v1 `S` proxy by a meaningful pre-approved
  margin.
- No-go if MIR 128-bit dev does not beat current best by at least `+0.015`
  mean.
- No-go if T2I drops while I2T improves.
- No-go if teacher topology improves proxy but lowers Hamming mAP.
- No-go if gradient conflict remains strongly negative.
- No-go if NUS/COCO v1 regression breaks during MIR-only work.

## 12. Explicit Prohibitions For This Draft

- No training code.
- No data/processed v2 cache.
- No training.
- No mAP evaluation.
- No formal `docs/specs/*` modification.
- No Stage 1/2/3/4/5 formal output modification.
- No NUS/COCO execution.
- No commit.
- No push.

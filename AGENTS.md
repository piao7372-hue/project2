# AGENTS

This repository is a cross-modal hashing project. Keep this file short: it
defines project-level rules and the reading order only.

## Reading Order

1. The latest explicit user request in the current conversation has priority.
2. `AGENTS.md` project rules.
3. `docs/specs/engineering_spec.md` is the primary code implementation
   specification.
4. `docs/specs/theory_semantic_relation.md`,
   `docs/specs/theory_post_semantic_method.md`, and
   `docs/specs/theory_loss_function.md` are the primary mathematical
   definition and loss-function specifications.
5. `C:/Users/ASVS/Desktop/1/科研论文ra.pdf` is reference-only for dataset
   protocol, reference results, and experiment targets.

## Current Scope

- Current stage: Stage 0A.
- Allowed work now: minimal project skeleton, environment lock, Stage 0
  configuration schema, and project specification documents.
- Do not enter Stage 1 or later until the current stage validator passes.
- Do not download raw data or model weights during Stage 0A document work.

## Hard Rules

- The engineering spec is the main implementation contract.
- The theory specs are the main source for math definitions and losses.
- RA paper content must not override Stage 3+ model design.
- Current stage validator failure blocks the next stage.
- No silent fallback behavior.
- No automatic bad-sample skipping.
- No automatic zero-vector padding.
- No automatic model switching.
- No legacy semantic aliases or compatibility layers.
- No `git commit` or `git push` without explicit user permission.
- Use this absolute Python interpreter path for project commands:
  `C:\Users\ASVS\anaconda3\envs\deeplearning\python.exe`

## Source And Product Boundary

- Source files belong under `configs/`, `docs/`, `scripts/`, and `src/`.
- Runtime products belong under `outputs/` or `data/` and are ignored by git.
- Do not stage runtime products, datasets, caches, logs, reports, or
  checkpoints as source.

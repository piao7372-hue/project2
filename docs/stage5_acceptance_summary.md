# Stage 5 Acceptance Summary

## Status

Stage 5 status: completed locally, ready for commit.

## Scope

- Implemented theory-consistent hash loss.
- Implemented S-derived `S_II_star` / `S_TT_star`.
- Implemented dense-vs-blockwise exact synthetic validation.
- Implemented real-data loss audit for MIR/NUS/COCO.
- Implemented loss-weight sensitivity audit.

## Theory Constraints

- Loss uses continuous `H_I` / `H_T`.
- Loss does not use `B_I` / `B_T`.
- Loss does not use `sign`.
- Loss does not use `label_vector`.
- `S_II_star` and `S_TT_star` are induced from `S`.
- `S` is the only cross-modal supervision target.
- No optimizer step or training was performed.

## Stage 5D Results

MIR:

- current profile risk = medium
- primary Stage 6 dev candidate = `mir_sem1p5_pair0p45`
- backup = `mir_sem2p0_pair0p35`

NUS:

- current profile risk = low
- primary Stage 6 dev candidate = `nus_current`
- backup = `nus_sem1p5_pair0p35`

COCO:

- current profile risk = high
- primary Stage 6 dev candidate = `coco_sem1p5_pair0p60`
- backup = `coco_sem2p0_pair0p50`

## Important Warning

- Stage 6 must not treat these candidates as final profiles.
- Stage 6 dev training must validate them with 128-bit mAP, bit balance, loss stability, and Hamming retrieval diagnostics.
- COCO current profile should not be used as the only Stage 6 run because its loss-weight profile is high risk.

## Stage 6 Boundary

- Stage 6 not started.
- Stage 7 not started.
- No mAP has been evaluated in Stage 5.

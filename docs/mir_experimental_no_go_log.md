# MIR Experimental No-Go Log

Status: research record only. This is not a formal specification, not a Stage 6
entry, and not a Stage 7 result.

## 1. Current Best

Current best exploratory MIR 128-bit result:

- I2T = 0.8519076634
- T2I = 0.8348295490
- mean = 0.8433686062

RA/RANEH MIR 128-bit target:

- I2T = 0.961
- T2I = 0.926

Approximate gap to RA/RANEH:

- I2T ~= -0.109
- T2I ~= -0.091

## 2. No-Go Routes Already Tried

Stage 2 prompt/text audit:

- Prompt ensemble mean ~= 0.804.
- Improvement ~= +0.020.
- This is not enough to explain or close the RA gap.

S adequacy:

- Current S mean ~= 0.7698.
- S_TT is weak.
- Conclusion: `S_not_adequate`.

S revision candidates:

- Best postprocess mean ~= 0.7735.
- Conclusion: `postprocessing_not_sufficient`.

No-label target design:

- Best candidate did not exceed S by enough to change the decision.
- Conclusion: `no_label_target_not_sufficient`.

Stage 4 V0-V3 architecture ablation:

- V0 mean ~= 0.8359.
- V1 ~= 0.8006.
- V2 ~= 0.8377.
- V3 ~= 0.8019.
- Graph is useful, tree is neutral, and this family is not gap-closing.

Image-guided teacher:

- Teacher target is strong:
  - mean ~= 0.887
  - P@50 ~= 0.924
  - SII/STT ~= 0.887 / 0.887
- Direct teacher probes and topology-loss probes did not improve model mAP.

Multitarget v2:

- R1 package was valid.
- R2 synthetic/calibration was partially valid.
- R2b found low-risk calibration candidates.
- R3 fixed-weight probe was No-Go.
- R4 curriculum probe was No-Go.

R4 curriculum results:

- C1 pair-protected T2:
  - I2T = 0.8391979928
  - T2I = 0.8380351735
  - mean = 0.8386165831
- C2 pair-protected T3:
  - I2T = 0.8308761966
  - T2I = 0.8207173751
  - mean = 0.8257967858
- Both remain below the current best exploratory mean = 0.8433686062.

## 3. Main Causes Ruled Out

- Quantization gap is not primary.
- Inference graph mode is not primary.
- Train/query generalization is not primary.
- Stage 1 data cleaning is not primary.
- Simple Stage 2 prompt revision is not enough.
- Simple Stage 4 V0-V3 changes are not enough.

## 4. More Likely Bottlenecks

- No-label semantic supervision is not strong enough.
- Text-side structure is weak.
- The current objective interface cannot absorb the strong image-teacher target
  effectively.
- The current self-designed route may require a larger theory-level revision or
  an RA-faithful branch.

## 5. Recommendations

- Stop current MIR v2 local probes.
- Clean experimental files and outputs.
- Keep this No-Go log.
- Then proceed to NUS/COCO baseline diagnostic or RA-faithful planning.
- Do not repeat these local probes unless theory/spec changes.

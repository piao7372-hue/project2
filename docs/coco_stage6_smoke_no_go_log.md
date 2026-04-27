# COCO Stage6 Smoke No-Go Log

## 1. Stage2 Baseline

- I2T = 0.9130829416
- T2I = 0.9165909334
- mean approx 0.9148369375

## 2. Primary Smoke Result

- candidate = coco_sem1p5_pair0p60
- I2T = 0.7308668822
- T2I = 0.7645312136
- mean = 0.7476990479
- L_pair 2.0078551769 -> 0.3502441049
- L_sem 0.3871515989 -> 0.4486243129
- bit health stable
- pass/no-go = no-go

## 3. Backup Smoke Result

- candidate = coco_sem2p0_pair0p50
- I2T = 0.7361261851
- T2I = 0.7735027700
- mean = 0.7548144775
- L_pair 2.0078551769 -> 0.3816738427
- L_sem 0.3871515989 -> 0.4353919625
- bit health stable
- pass/no-go = no-go

## 4. Interpretation

- COCO is not a MIR-specific failure.
- Stage2 CLIP is strong, but Stage6 hash training collapses diagnostic Hamming performance.
- `L_pair` decreases strongly while `L_sem` increases.
- Bit health is stable, so the failure is not simple bit collapse.
- Primary and backup both fail survival gates.
- The current Stage6 mainline should stop.
- Do not run NUS under this current mainline until the hash learning / objective contract is redesigned.

## 5. Next Recommendation

- Stop current Stage6 mainline.
- Clean experimental COCO smoke files and outputs.
- Preserve this No-Go log.
- Next work should be formal theory / hash learning objective redesign or RANEH-faithful branch planning, not more profile tuning.

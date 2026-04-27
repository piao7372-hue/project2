# NUS Stage6 Smoke No-Go Log

Status: research record only. This is not a formal Stage7 result and not a
formal RA/RANEH conclusion.

## 1. Stage2 baseline

- I2T = 0.6903989647
- T2I = 0.6751644357
- mean ~= 0.6827817002

## 2. Primary smoke result

- candidate = nus_current
- I2T = 0.6765759835
- T2I = 0.6390713674
- mean = 0.6578236754
- L_pair 1.9993755817 -> 0.8833292723
- L_sem 0.4253017306 -> 0.4823571742
- bit health stable
- no_relevant_query_count = 0
- pass/no-go = no-go

## 3. Backup smoke result

- candidate = nus_sem1p5_pair0p35
- I2T = 0.6805707170
- T2I = 0.6410526693
- mean = 0.6608116931
- L_pair 1.9993755817 -> 0.9299941063
- L_sem 0.4253017306 -> 0.4693430364
- bit health stable
- no_relevant_query_count = 0
- pass/no-go = no-go

## 4. Interpretation

- NUS is also No-Go under current Stage6 mainline.
- Backup improves only +0.00299 mean over primary.
- Both primary and backup are below Stage2 CLIP baseline.
- `L_pair` decreases strongly while `L_sem` increases.
- Bit health is stable, so failure is not bit collapse.
- `no_relevant_query_count = 0`, so this is not evaluator no-relevant artifact.
- Current Stage6 mainline should stop.
- Do not run more NUS profiles under the same mainline.

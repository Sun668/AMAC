# AMAC Competitive Risk Baselines

This post-hoc, method-locked experiment tests whether the validated AMAC result is
explained by ordinary confidence calibration rather than multimodal risk features.
It reuses the frozen CH-SIMS v2 test predictions and never retrains or changes the
upstream sentiment model.

## Question

At comparable coverage, does a learned feature-based risk estimator outperform:

- `B4`: the archived fixed confidence threshold baseline;
- `PLATT`: Platt scaling of the raw confidence only;
- `ISO`: isotonic calibration of the raw confidence only;
- `LR`: logistic regression on the same contract features as `H0`.

`H0`, `B2`, and `B4` are copied byte-for-byte at row level from the validated
CH-SIMS v2 test artifact. New estimators are fitted only on the archived training
predictions, tuned only on the deterministic training calibration groups, and
evaluated once on the archived test predictions.

## Interpretation

- If `LR` is non-inferior to `H0`, the defensible contribution is the learned
  risk-aware commitment contract, not a particular neural architecture.
- If `LR` or `H0` beats both scalar calibrators, the gain is not adequately
  explained by calibrating a single confidence score.
- If scalar calibration matches the feature-based methods, the present method
  claim is too weak for a strong paper and must be narrowed.

## Formal command

```bash
python experiments/exp_2026_09_03_amac_competitive_baselines/scripts/run_baselines.py \
  --run-id competitive_baselines_v1
python experiments/exp_2026_09_03_amac_competitive_baselines/scripts/validate_baselines.py \
  --result-dir experiments/exp_2026_09_03_amac_competitive_baselines/results/competitive_baselines_v1
```


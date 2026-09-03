# EmotionTalk held-out group audit

## Purpose

Audit the external contract result at the highest independent grouping level
available in the released index. The test contains only G00013, G00014, and
G00015, so the analysis reports per-group effects and does not use a
three-cluster confidence interval for inferential claims.

## Estimands

- Preterminal committed-error reduction: B4 minus H0.
- Revision-per-path reduction: B4 minus H0.
- Absolute stage-two coverage gap.

Positive reduction values favor H0. The result is descriptive evidence across
three held-out groups and cannot establish population-level transfer.

## Frozen inputs

- Validated `emotiontalk_external_v1_recovery/per_path.csv`.
- Its frozen parameter snapshot and validator report.
- No refitting, retuning, API call, or prediction regeneration.

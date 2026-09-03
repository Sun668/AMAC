# PLAN

1. Freeze hashes for code, parameters, and every upstream artifact.
2. Verify the upstream CH-SIMS v2 result passed its independent validator.
3. Reconstruct the frozen train and test sequential events.
4. Fit `LR`, `PLATT`, and `ISO` without using test labels.
5. Tune commitment thresholds on deterministic train calibration groups only.
6. Evaluate five fixed seeds on the untouched archived test predictions.
7. Compute paired clip-level bootstrap intervals and independent validation.
8. Promote only claims supported by the frozen artifacts and gates.


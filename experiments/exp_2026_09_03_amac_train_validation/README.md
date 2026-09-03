# AMAC train/validation development experiment

This experiment tests whether a learned anytime affect contract can reduce incorrect pre-final commitments and committed-state revisions under asynchronous text/audio/vision arrival.

This is a development-only run. It uses CH-SIMS v2 `train` for cross-fitted learning and internal calibration, and `valid` for a held-out development evaluation. The official `test` split is not indexed or used. Results cannot be reported as final thesis evidence.

Primary comparison: learned AMAC (`M0`) versus fixed confidence hysteresis (`B4`) at stage-two coverage of at least 0.90.

Run order:

1. Run the project-governor preflight.
2. Run `scripts/run_development.py` with the frozen parameter snapshot and an absent condition directory.
3. Run `scripts/validate_results.py` against the completed directory.
4. Use `decision.json` only to decide whether a later frozen test experiment should be prepared.

No external API or model is used. The USD budget is zero.

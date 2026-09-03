# PLAN

## Hypothesis

At stage-two coverage of at least 0.90, a learned current-state risk estimate
reduces prefinal committed error and committed revisions relative to fixed
confidence routing. Prefix history is retained only if validation supports it.

## Frozen selection logic

1. Train `M0` and `H0` for five fixed seeds using grouped train-only fitting
   and calibration.
2. Require each candidate to pass the same B2/B3/B4, coverage, interval, and
   final-identity gates.
3. Select `H0` when it passes all core gates and is non-inferior to `M0` on
   every seed under the frozen tolerances.
4. Otherwise select `M0` if it passes all core gates.
5. If neither condition applies, keep CH-SIMS v2 test sealed.
6. Report history materiality separately; it is not allowed to retroactively
   redefine the core learned-risk claim.

## Ordered execution

1. Freeze parameters and code hashes.
2. Run generic read-only preflight.
3. Execute the five-seed validation matrix into a new output directory.
4. Run the independent validator.
5. Record the selected model and either authorize or deny one-shot test setup.

## Leakage and invalidation

The test split must not be indexed, loaded, summarized, or used for selection.
Any change to data identity, features, targets, seeds, thresholds, gates,
runner, or validator requires a new run ID and frozen snapshot.

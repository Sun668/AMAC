# AMAC Risk-Contract Validation

This experiment narrows AMAC to one testable mechanism: a learned estimate of
the risk that the currently visible multimodal prefix would produce an
incorrect affect state. The agent uses that estimate to WAIT, COMMIT, or
REVISE while preserving the final TAV prediction.

The run is validation-only. It uses CH-SIMS v2 `train` for model fitting and
internal calibration and `valid` for selection. The `test` split is forbidden.
A frozen selection rule chooses either the full-history model `M0` or the
simpler no-history model `H0`; it does not require history to appear useful.

## Conditions

| ID | Meaning |
|---|---|
| B0 | Commit eagerly at every arrival |
| B1 | Wait for all modalities |
| B2 | Fixed confidence threshold |
| B3 | Commit after two consecutive states agree |
| B4 | Fixed confidence threshold plus revision margin |
| M0 | Learned risk contract with prefix-history features |
| H0 | Learned risk contract without history/change features |
| O1 | Label-aware oracle, diagnostic upper bound only |

`M0` and `H0` are trained only on whether the current prefix state matches the
gold state. Unsupported future-stability, modality-quality, and explicit
revision-loss claims from the previous matrix are deliberately removed.

## Evidence boundary

Passing this matrix only authorizes creation of a one-shot test snapshot. It
is not a CH-SIMS v2 official leaderboard result and is not itself the thesis
test result.

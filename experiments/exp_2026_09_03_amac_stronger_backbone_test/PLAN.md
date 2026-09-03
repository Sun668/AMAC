# PLAN

## Formal robustness question

After development on `train -> valid`, does H0 reduce wrong pre-final affect
commitments under a nonlinear subset-aware multimodal backbone on the frozen
CH-SIMS v2 test split?

## Frozen primary hypothesis

At H0 stage-two coverage of at least 0.90 and an absolute H0-B4 coverage gap
of at most 0.02, H0 reduces pre-final committed error by at least 0.02 absolute
for every risk seed, with every paired clip-bootstrap lower bound above zero.

Revision reduction is exploratory because the development run found that the
stronger backbone itself removed most revisions. Terminal `TAV` quality and a
mechanically defined source-disjoint subset are robustness checks. This change
was made after validation development and before this runner indexed `test`.

## Fixed design

- Data: official CH-SIMS v2 processed unaligned features.
- Training: `train` only; evaluation: `test` only; `valid` is not indexed.
- Backbone: the exact development-selected masked modality-token Transformer.
- Inputs: all seven non-empty `T/A/V` subsets; all six arrival orders.
- Cross-fitting: three source-group folds for H0 training predictions.
- Risk seeds: 1, 12, 123, 1234, 12345.
- Policies: B0, B1, B2, B3, B4, H0, O1.
- Policy selection: deterministic calibration source groups from training only.
- Statistics: 5,000 paired bootstrap repetitions by original clip.
- Source-disjoint audit: test source prefix absent from training source prefixes.
- Final state: forced equal to the full `TAV` backbone prediction for all policies.

## Interpretation

Passing supports backbone robustness of the AMAC error-exposure claim. It does
not establish raw-audio/video end-to-end perception, unseen-dataset
generalization, or a universal revision benefit.


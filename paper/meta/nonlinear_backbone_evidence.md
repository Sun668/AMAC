# Nonlinear-backbone evidence note

## Identity

- Run: `masked_fusion_test_v1`
- Snapshot SHA-256: `2de4dc06420c6ddb1ab5336c7411be1f7f810b732e1310e0948ac76759d43c61`
- Validator: passed
- Dataset: CH-SIMS v2 official processed unaligned container
- Splits: train for fitting and test for evaluation; valid not indexed
- External API cost: USD 0

## Supported claims

- The subset-aware nonlinear backbone improves terminal test MAE from 0.4293
  to 0.3033 and Acc-3 from 64.89% to 71.08% versus the frozen Ridge reference.
- At a per-seed H0-B4 stage-two point-estimate coverage gap below 0.02, H0 reduces committed
  pre-final error by 5.56-6.06 percentage points for every seed.
- Every 10,000-repetition source-ID-prefix cluster-bootstrap error interval
  has a positive lower endpoint on the 142-group official test.
- Revision reduction is positive and statistically supported on this test,
  but remains exploratory because development did not pass its preregistered
  revision gate.

## Limits

- The 190-clip source-disjoint subset contains only 15 source groups and is
  descriptive because its policy coverage gap exceeds the primary 0.02 tolerance.
- The official split shares 127 source-ID-prefix groups between train and test;
  the result does not establish cross-source generalization.
- The archived official-split run met its prespecified numerical performance
  gates. It is not described as passing every validity gate.
- Inputs remain released feature tensors, not raw-media end-to-end inference.
- This experiment supports backbone robustness, not cross-dataset
  generalization or downstream response quality.

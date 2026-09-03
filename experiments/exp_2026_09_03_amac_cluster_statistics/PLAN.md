# PLAN

## Purpose

Correct the uncertainty analysis after discovering that CH-SIMS v2 clips are
clustered within source videos. The previous clip bootstrap remains an
empirical clip-level sensitivity analysis, but it is not used as evidence for
cross-source generalization.

## Analysis class

This is a post-hoc statistical correction, not a new confirmatory experiment.
It does not retrain, retune, or regenerate predictions. It reads only frozen,
validated path records from the Ridge and masked-fusion test runs.

## Estimands

- Error reduction: `B4 committed error - H0 committed error`.
- Revision/path reduction: `B4 revisions/path - H0 revisions/path`.
- Revision/opportunity reduction: `B4 revisions/committed pre-final state - H0 revisions/committed pre-final state`.
- Coverage gap: `abs(B4 stage-two coverage - H0 stage-two coverage)`.

Positive reductions favor H0. Ratio estimands are computed from pooled
numerators and denominators, not by averaging clip-level ratios.

## Resampling

1. Derive a source-video group from the clip id prefix before `$_$`.
2. Aggregate all clips and all six arrival paths within each source group.
3. Sample source groups with replacement and retain each sampled group's full
   aggregate contribution.
4. Use 10,000 deterministic percentile-bootstrap repetitions per risk seed.
5. Analyze both the full official test set and the 15-group source-disjoint subset.
6. For the 15-group subset, additionally report every group effect, the
   positive-group count, an exact two-sided sign test, median/IQR, and a
   small-sample t interval for the equal-group mean.

## Validity requirements

- Source artifacts and validators must match their frozen SHA-256 identities.
- Every test clip must map to exactly one non-empty source group.
- Ridge full test must contain 142 groups; its source-disjoint subset must
  contain 15 groups and 190 clips.
- All six paths must remain grouped under each clip and source group.
- An independent validator must reproduce every point estimate and interval.

## Interpretation boundary

Cluster bootstrap addresses within-source dependence in uncertainty estimates.
It does not remove the training-test source overlap. Full-test results remain
official-split performance; the 15-group analysis is limited robustness
evidence, not a definitive cross-source generalization test.


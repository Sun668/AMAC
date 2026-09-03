# AMAC P0 robustness analysis

## Purpose

Repair the manuscript's P0 evidence gaps without changing the sealed primary
CH-SIMS v2 experiment. This analysis is supplementary and post-hoc.

## Frozen questions

1. Do the official train and test partitions share clips or source-video groups?
2. What happens on test clips whose source video is absent from training?
3. What are the stage-specific denominators and revision opportunities?
4. How do risk estimators compare over a common committed-state coverage range?
5. What are the paired-bootstrap interval endpoints for H0 versus B4?

## Evidence boundary

- The 1,034-clip official-split result remains primary.
- The source-disjoint subset is fixed as test clips whose ID prefix before
  `$_$` never occurs in train.
- No model is fitted or selected on test labels.
- Threshold sweeps are descriptive diagnostics, not deployment selection.
- Source-disjoint results are supplementary because the subset is smaller.


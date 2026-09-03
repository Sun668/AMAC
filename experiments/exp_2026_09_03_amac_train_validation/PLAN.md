# PLAN

## Hypothesis

For identical CH-SIMS v2 clips and all six text/audio/vision arrival orders, M0 can learn when an affect state is both likely correct and likely stable. At matched stage-two coverage, it should expose fewer wrong pre-final states and revise committed states less often than B4.

## Design

1. Pool each official unaligned modality feature sequence using mean and standard deviation.
2. Produce leakage-controlled train predictions for all seven non-empty modality subsets using five-fold GroupKFold by source-video id and Ridge regression.
3. Fit each subset model on all train clips and predict valid clips.
4. Expand every clip into six arrival paths and pre-final stages.
5. Train a dual-head MLP on OOF train events: current-state correctness and future path stability.
6. Tune M0 and baseline thresholds only on a deterministic source-group calibration partition of train.
7. Evaluate B0-B4, M0, and O1 on valid, with paired bootstrap by original clip.
8. Independently validate hashes, row completeness, policy metrics, final-state identity, and the preregistered decision.

## Continue gate

A later formal-test run may be prepared only if all validity checks pass and M0 versus B4 achieves: stage-two coverage >= 0.90; pre-final committed error reduction >= 0.02 absolute with positive paired-bootstrap lower bound; committed revision reduction >= 20% relative with positive paired-bootstrap lower bound; final-state identity = 1.0. The underlying problem must also remain observable: eager revision >= 0.10 and eager premature exposure >= 0.05.

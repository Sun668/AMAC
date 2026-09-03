# PLAN

1. Join modality-specific metadata by clip identity without copying content or
   raw media into the experiment artifact.
2. Freeze source-file identities, joined-index hash, group split, classes,
   fusion rule, five seeds, policy parameters, metrics, and gates.
3. Split by top-level source group so clips from one source do not cross train,
   calibration, and test partitions.
4. Simulate all six T/A/V arrival orders. At each prefix, fuse only visible
   modality consensus outputs using annotation agreement and confidence.
5. Train H0 to predict whether the current fused state matches the independent
   multimodal consensus. Compare with B0-B4 and O1.
6. Independently validate hashes, rows, metrics, split isolation, and effects.

The required external claim is direction consistency with CH-SIMS v2 at
coverage >= 0.90, not identical effect size.

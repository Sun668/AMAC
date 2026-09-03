# AMAC stronger-backbone development gate

This experiment replaces the linear Ridge affect predictor with a shared,
nonlinear, subset-aware modality-token fusion network. It uses only the
official CH-SIMS v2 training and validation splits and cannot be promoted as
final test evidence.

The frozen snapshot binds the dataset, Ridge reference, code, architecture,
optimization, policy search, statistics, resources, and gates. Results are
written once to `results/<run-id>` and checked by an independent validator.


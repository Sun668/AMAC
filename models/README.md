# AMAC model release

This directory contains reusable implementation rather than experiment
orchestration.

- architectures.py: correctness MLP and masked-fusion Transformer.
- features.py: CH-SIMS v2 and EmotionTalk prefix feature construction.
- training.py: deterministic Ridge fitting and neural training functions.
- checkpoints.py: loaders and inference helpers for the weights directory.
- infer.py: minimal correctness-checkpoint CLI.
- go/: stateful WAIT, COMMIT, HOLD, and REVISE Agent contract and JSON tool.

CH-SIMS correctness models consume the frozen 25-dimensional event feature.
EmotionTalk models consume the 15-dimensional annotation-derived feature.
Exact ablations and policy thresholds are in
paper/evidence/<run-id>/parameters.json.

Restricted datasets and official feature containers are not redistributed.

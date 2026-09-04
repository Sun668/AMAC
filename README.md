# AMAC

Official paper, reusable model implementation, trained checkpoints, and compact
claim evidence for AMAC: Risk-Aware Commitment under Asynchronous Modality
Arrival for Multimodal Affective Agents.

## Repository layout

- models/: model definitions, feature construction, training and inference
  helpers, and the executable Go Agent contract.
- weights/: formal neural checkpoints used in the manuscript, with SHA-256
  hashes and provenance.
- paper/: LaTeX source, PDF, figures, submission notes, and compact evidence.
- paper/evidence/: final metrics, frozen parameters, manifests, and independent
  validator reports only.

The repository excludes restricted datasets, raw media, official feature
containers, development notebooks, experiment orchestration, PLAN and TODO
files, caches, and per-path bulk outputs.

## Model setup

Create a Python environment and install requirements.txt. Loaders in
models.checkpoints consume checkpoints in weights/. Real prefix features must be
constructed with models.features and the frozen settings in the corresponding
paper/evidence/<run-id>/parameters.json.

## Evidence boundary

The paper uses the official CH-SIMS v2 split under all six simulated arrival
orders. Main numbers are official-split operating-point results, not a
cross-source generalization claim. EmotionTalk is an annotation-derived,
three-held-out-group descriptive check, not raw-media end-to-end transfer.

Compact evidence directories preserve published metrics and validation status.
They are not full experiment histories and do not replace access to licensed
datasets.

## Paper

The manuscript source is paper/main.tex. See paper/ARXIV_SUBMISSION.md for the
arXiv checklist.

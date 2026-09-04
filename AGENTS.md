# AMAC repository instructions

## Scope

This publication release has three substantive areas:

- models/: reusable Python model code and the Go Agent contract.
- weights/: formal trained neural checkpoints and hashes.
- paper/: manuscript sources and compact final claim evidence.

Do not add application UI, unrelated research, raw datasets, private media,
caches, notebooks, experiment orchestration, PLAN or TODO files, or ad-hoc
diagnostic outputs.

## Evidence rules

- Treat paper/evidence/ as immutable publication evidence.
- Do not overwrite an existing evidence directory.
- Do not claim that compact evidence is a complete experiment history.
- Do not claim cross-source generalization from the official CH-SIMS v2 split.
- Treat EmotionTalk as an annotation-derived descriptive check over three
  held-out groups.
- Do not present the MLP architecture itself as the paper contribution.

## Weight rules

- Only checkpoints used by a reported manuscript result belong in weights/.
- Every checkpoint must have a source run and SHA-256 entry in
  weights/manifest.json and weights/SHA256SUMS.
- Restricted datasets and feature containers must not be committed.

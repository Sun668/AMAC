# AMAC

Official research artifacts for **AMAC: Risk-Aware Commitment under Asynchronous Modality Arrival for Multimodal Affective Agents**.

AMAC separates terminal multimodal recognition from the decision to expose an intermediate affect state. Given modalities arriving in an arbitrary order, the stateful contract emits `WAIT`, `COMMIT`, or `REVISE`, while forcing the terminal state to equal the complete text-audio-vision prediction.

## Repository contents

- `paper/`: IEEEtran LaTeX source, generated tables, evidence notes, references, and the current PDF.
- `internal/affectcontract/`: standalone Go state-transition contract.
- `internal/tools/`: JSON-facing stateful Agent tool with `start`, `observe`, and `reset` operations.
- `experiments/`: only the development, official-split, robustness, external descriptive, statistical, and replay code used by the manuscript.
- `experiments/**/snapshots/`: frozen parameters for the authoritative successful runs.
- `experiments/**/results/`: compact metrics, manifests, decisions, intervals, and validator reports used by the paper.

## Excluded artifacts

Raw CH-SIMS v2 and EmotionTalk data are not redistributed because their licenses and access terms apply. Caches, extracted tensors, checkpoints, failed exploratory runs, and large row-level outputs are also excluded. The compact manifests retain artifact identities and hashes; larger reproducibility bundles can be attached to a tagged release where licensing permits.

## Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Each experiment directory contains its own `README.md`, `PLAN.md`, frozen parameters, runner, validator, and authoritative compact results. Dataset paths must be supplied according to the corresponding experiment documentation.

## Go affect contract

```bash
go test ./...
```

The replay runner is located at:

```text
experiments/exp_2026_09_03_affect_contract_agent_replay/scripts/replay.go
```

Its input is generated from the formal path-level output by `prepare_replay.py`; the large generated JSONL is not stored in Git.

## Build the paper

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=output main.tex
```

The repository is prepared for an arXiv source package and later IEEE submission. Author identity, affiliation, funding, repository revision, and venue-specific declarations must be completed before submission.

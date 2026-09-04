# AMAC paper

This directory contains the complete manuscript package: LaTeX source, figures,
references, submission notes, metadata, and compact claim evidence.

## Main files

- main.tex: English manuscript in IEEEtran journal format, suitable for arXiv.
- AMAC_arXiv_draft.pdf: compiled manuscript.
- references.bib: cited primary literature.
- figures/amac_overview.tex: method overview included by main.tex.
- generated/: validated tables used by the manuscript.
- evidence/: final metrics, parameters, manifests, and validator reports.
- ARXIV_SUBMISSION.md: arXiv packaging and author checklist.
- meta/: paper context, evidence ledger, figure inventory, and format rules.

## Build

Run latexmk -pdf main.tex.

Evidence directories are deliberately compact. They preserve reported numbers
and frozen settings but omit experiment runners, development plans, restricted
datasets, caches, and row-level bulk outputs.

The primary reporting audit is evidence/p0_robustness_v2.

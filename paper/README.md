# AMAC arXiv paper

This directory contains the complete manuscript package for the affective-agent
study. All paper-specific prose, figures, references, evidence maps, and
submission notes live here.

## Main files

- `main.tex`: English manuscript in IEEEtran journal format, also suitable for arXiv.
- `references.bib`: cited primary literature.
- `figures/amac_overview.tex`: method overview included by `main.tex`.
- `generated/`: validated P0 tables generated from `p0_robustness_v2`.
- `ARXIV_SUBMISSION.md`: author actions and arXiv packaging instructions.
- `meta/`: paper context, evidence ledger, figure inventory, and format rules.

## Author action required

Replace the placeholder author, affiliation, email, funding, repository URL,
and contribution statements before submission. The manuscript deliberately
does not invent these details.

## Suggested build

```bash
latexmk -pdf main.tex
```

The experiment artifacts remain in `experiments/`; the manuscript cites their
stable run identifiers in the reproducibility appendix.

The P0 reporting audit is authoritative at
`experiments/exp_2026_09_03_amac_p0_robustness/results/p0_robustness_v2`.
Its validator independently recomputes stage denominators, source isolation,
bootstrap intervals, and risk--coverage summaries from frozen source artifacts.

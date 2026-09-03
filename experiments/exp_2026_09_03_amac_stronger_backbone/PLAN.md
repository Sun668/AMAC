# PLAN

## Question

Does the AMAC risk contract remain useful when the affect predictor changes
from seven independent Ridge regressors to one nonlinear, subset-aware
multimodal fusion backbone?

## Development hypothesis

On the official CH-SIMS v2 `train -> valid` split, the masked fusion backbone
should preserve terminal `TAV` quality within 0.02 MAE of the frozen Ridge
reference. At stage-two coverage of at least 0.90, H0 should reduce committed
pre-final error by at least 0.02 absolute and revisions by at least 20% relative
to B4.

This is a development gate, not paper evidence. It never indexes `test`. A
successful result permits a separately frozen formal test; a failed result is
retained without changing this protocol.

## Treatment and controls

| Comparison | Only intentional difference | Question |
|---|---|---|
| MaskedFusion vs Ridge | Affect prediction backbone | Does nonlinear fusion preserve terminal quality? |
| MaskedFusion+H0 vs MaskedFusion+B4 | Commitment score and hysteresis rule | Does learned risk retain its benefit? |
| B0/B1 | Eager and final-only endpoints | Are the service extremes visible? |

## Model and isolation

1. Pool each official unaligned feature sequence with mean and standard deviation.
2. Standardize each modality using training-fold statistics only.
3. Project text, audio, and vision into 64-dimensional modality tokens.
4. Apply explicit availability masks and a two-layer, four-head Transformer encoder.
5. Mask-pool visible tokens and regress the official multimodal sentiment score.
6. Train every clip under all seven non-empty modality subsets with equal loss weight.
7. Produce source-group cross-fitted train predictions and untouched valid predictions.
8. Train H0 from cross-fitted train events and tune policies only on deterministic train calibration groups.

## Statistics and gates

- Sample unit: original clip with all six arrival orders grouped.
- Uncertainty: 1,000 paired clip bootstrap repetitions.
- Validity: frozen hashes, no train-valid clip overlap, grouped cross-fitting,
  finite seven-subset predictions, complete traces, forced final identity, and
  independent validation.
- Performance: terminal MAE degradation at most 0.02; H0 stage-two coverage at
  least 0.90; H0-B4 error reduction at least 0.02; revision reduction at least
  20%; both paired interval lower bounds positive.

## Cost

- External API: USD 0.
- Device: local CPU, at most four threads.
- Expected wall time: 5-25 minutes.
- Expected incremental storage: below 100 MB.


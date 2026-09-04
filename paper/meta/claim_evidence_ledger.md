# Claim-evidence ledger

| ID | Manuscript claim | Authoritative artifact | Status |
|---|---|---|---|
| P1 | H0 lowers prefinal committed error versus matched-coverage B4 by 12.14 percentage points on CH-SIMS v2. | `paper/evidence/chsimsv2_test_v1_recovery/metrics.json` | supported |
| P2 | H0 lowers committed revisions versus B4 by 11.67 percentage points on CH-SIMS v2. | Same artifact and paired bootstrap output. | supported for CH-SIMS v2 |
| P3 | LR is numerically close to H0 at the selected operating point; this is descriptive rather than formal non-inferiority. | `paper/evidence/competitive_baselines_v1/result.json` | supported as post-hoc description |
| P3a | The tested nine-feature estimators differ from Platt and isotonic mappings over a shared committed-state risk--coverage interval. | `paper/evidence/p0_robustness_v2/metrics.json`; normalized AURC is 0.354 for H0, 0.354 for LR, 0.462 for Platt, and 0.460 for isotonic. | supported as post-hoc descriptive analysis |
| P4 | Across three held-out EmotionTalk groups, the annotation-derived error difference favors H0 by 5.35 percentage points. | `paper/evidence/emotiontalk_external_v1_recovery/metrics.json` | supported as a descriptive contract-level pattern |
| P5 | Revision reduction transfers to EmotionTalk. | Same external artifact; revisions increase by 2.98 percentage points. | rejected |
| P6 | The Go Agent tool exactly reproduces the frozen LR trajectories. | `paper/evidence/affect_contract_replay_v1/result.json` | supported for 18,612 calls |
| P7 | AMAC improves terminal TAV accuracy. | Terminal identity is fixed by design. | prohibited |
| P8 | MLP architecture is a contribution. | LR is numerically close to H0, so the MLP architecture is not isolated as a contribution. | rejected |
| P9 | Official CH-SIMS v2 train and test have no exact clip overlap but share 127 source-ID-prefix groups. | `paper/evidence/p0_robustness_v2/source_split_audit.json` | supported |
| P10 | On the 190-clip source-ID-prefix-disjoint subset, H0 has lower committed error and revision-per-opportunity than B4 for all five seeds. | `paper/evidence/p0_robustness_v2/metrics.json`; every paired 95% interval is positive; validator accepted. | supported as post-hoc supplementary robustness evidence |

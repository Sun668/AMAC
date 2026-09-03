# Claim-evidence ledger

| ID | Manuscript claim | Authoritative artifact | Status |
|---|---|---|---|
| P1 | H0 lowers prefinal committed error versus matched-coverage B4 by 12.14 percentage points on CH-SIMS v2. | `experiments/exp_2026_09_03_amac_chsimsv2_test/results/chsimsv2_test_v1_recovery/metrics.json` | supported |
| P2 | H0 lowers committed revisions versus B4 by 11.67 percentage points on CH-SIMS v2. | Same artifact and paired bootstrap output. | supported for CH-SIMS v2 |
| P3 | LR is numerically close to H0 at the selected operating point; this is descriptive rather than formal non-inferiority. | `experiments/exp_2026_09_03_amac_competitive_baselines/results/competitive_baselines_v1/result.json` | supported as post-hoc description |
| P3a | The tested nine-feature estimators differ from Platt and isotonic mappings over a shared committed-state risk--coverage interval. | `experiments/exp_2026_09_03_amac_p0_robustness/results/p0_robustness_v2/metrics.json`; normalized AURC is 0.354 for H0, 0.354 for LR, 0.462 for Platt, and 0.460 for isotonic. | supported as post-hoc descriptive analysis |
| P4 | The error-risk direction transfers to EmotionTalk annotations by 5.35 percentage points. | `experiments/exp_2026_09_03_amac_emotiontalk_external/results/emotiontalk_external_v1_recovery/metrics.json` | supported at contract level |
| P5 | Revision reduction transfers to EmotionTalk. | Same external artifact; revisions increase by 2.98 percentage points. | rejected |
| P6 | The Go Agent tool exactly reproduces the frozen LR trajectories. | `experiments/exp_2026_09_03_affect_contract_agent_replay/results/affect_contract_replay_v1/result.json` | supported for 18,612 calls |
| P7 | AMAC improves terminal TAV accuracy. | Terminal identity is fixed by design. | prohibited |
| P8 | MLP architecture is a contribution. | LR is non-inferior to H0. | rejected |
| P9 | Official CH-SIMS v2 train and test have no exact clip overlap but share 127 source-video groups. | `experiments/exp_2026_09_03_amac_p0_robustness/results/p0_robustness_v2/source_split_audit.json` | supported |
| P10 | On the 190-clip test-only-source subset, H0 has lower committed error and revision-per-opportunity than B4 for all five seeds. | `experiments/exp_2026_09_03_amac_p0_robustness/results/p0_robustness_v2/metrics.json`; every paired 95% interval is positive; validator accepted. | supported as post-hoc supplementary robustness evidence |

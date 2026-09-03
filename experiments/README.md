# Authoritative AMAC experiment set

Only experiments that directly support the manuscript are included. Historical alternatives and failed runs are omitted.

| Evidence role | Experiment | Authoritative run |
|---|---|---|
| Risk-model development | `exp_2026_09_03_amac_train_validation` | `dev_chsimsv2_amac_mlp_v7_seed20260903` |
| Frozen contract matrix | `exp_2026_09_03_amac_risk_contract_validation` | `risk_contract_v1` |
| CH-SIMS v2 official split | `exp_2026_09_03_amac_chsimsv2_test` | `chsimsv2_test_v1_recovery` |
| Competitive calibration controls | `exp_2026_09_03_amac_competitive_baselines` | `competitive_baselines_v1` |
| Reporting and source audit | `exp_2026_09_03_amac_p0_robustness` | `p0_robustness_v2` |
| Source-group statistical correction | `exp_2026_09_03_amac_cluster_statistics` | `cluster_statistics_v3` |
| Nonlinear backbone development | `exp_2026_09_03_amac_stronger_backbone` | `masked_fusion_dev_v3` |
| Nonlinear backbone test | `exp_2026_09_03_amac_stronger_backbone_test` | `masked_fusion_test_v1` |
| EmotionTalk contract-level check | `exp_2026_09_03_amac_emotiontalk_external` | `emotiontalk_external_v1_recovery` |
| EmotionTalk held-out group audit | `exp_2026_09_03_amac_emotiontalk_group_audit` | `emotiontalk_group_audit_v1` |
| Stateful Go replay | `exp_2026_09_03_affect_contract_agent_replay` | `affect_contract_replay_v1` |

# CH-SIMS v2 One-Shot Test

This experiment evaluates the validation-selected `H0` learned risk contract
once on the sealed CH-SIMS v2 test split. All thresholds are imported from the
independently validated `risk_contract_v1` artifacts and frozen before access.
No test-dependent tuning, model selection, or rerun is permitted.

The official MMSA regression metrics are reported as an anchor. The
asynchronous arrival protocol and contract metrics are a new protocol built on
the public dataset and must not be described as an official leaderboard task.

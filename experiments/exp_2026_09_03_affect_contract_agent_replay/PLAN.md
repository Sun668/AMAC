# PLAN

1. Derive replay observations only from frozen competitive-baseline artifacts.
2. Freeze replay input and implementation hashes before execution.
3. Invoke the actual Go Agent tool for all 6,204 test paths.
4. Compare every online committed state with the archived offline trajectory.
5. Measure decision latency and exercise malformed or unsafe calls.
6. Independently validate counts, hashes, identity, and robustness gates.


# Go Agent contract

This module is the executable AMAC state machine used by the paper. It exposes
WAIT, COMMIT, HOLD, and REVISE decisions and preserves the terminal
full-modality prediction.

The package is independent from application UI and transport code. JSON tool
responses contain code and message, matching the original agent contract.

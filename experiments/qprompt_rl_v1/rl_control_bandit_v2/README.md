# RL_CONTROL_BANDIT_V2

Completed: 108/108 evaluated endpoints and 4/4 R3a audits. Primary fresh-seed RL−FIX_FINE mean macro Dice: −0.001546709 (−0.154671 percentage points), 0/8 positive cells. The registered investment target was not met.

See [final report](reports/FINAL_INTERPRETATION.md), [results](reports/RESULTS.csv), [all physical costs](reports/ALL_PHYSICAL_COSTS.csv), [transaction audit](reports/TRANSACTION_AUDIT.json), and [patch history](PATCH_LOG.md). Training provenance is recorded per endpoint. P1 invalidated evidence is excluded from metrics and included in physical cost.

The original proposal JSON remains unchanged; separate execution authorization records the user request. R2 and R3c training were not authorized or executed. Original KI remains unbound. Private datasets, raw logs, runtime paths and checkpoints are excluded.

Entry: `from rl_control_bandit_v2.runner import main; main()` with private runtime environment. `test_contract.py` provides zero-optimizer CPU regressions; `closeout.py` audits saved states and aggregates private ledgers without training.

# Study closed: FAIL_BASELINE_FEASIBILITY

Updated 2026-08-31T19:10:19.011203+00:00. The earlier training-running snapshot remains preserved in commit `ddf1bd081ca51870f567a98381c1e563a9917fe3` and its verified NAS publication copy.

All six original runs and three queues exited 0, completing 80,400 formal updates. The one test evaluation completed all 36 cells, 612 forward calls and 2,430 case predictions. The artifact-only audit passed with zero extra model forwards. NAS archive verification passed; all originals remain.

Four scientific criteria passed. Mean delta I was -0.010064513822073073, below the frozen -0.01 lower bound, so the outcome remains **FAIL_BASELINE_FEASIBILITY**. Do not round it to a pass or repeat/tune this study. See [FINAL_REPORT.md](FINAL_REPORT.md), [RESULT.json](RESULT.json), [TEST_CELLS.csv](TEST_CELLS.csv), [STATUS.json](STATUS.json) and [ARCHIVE_RECEIPT.json](ARCHIVE_RECEIPT.json).

No formal worker remains active. Keep the 30-minute heartbeat for a separate bounded external-method source/comparability audit and prospective registration; do not restart this study or the closed prototype-derived line.

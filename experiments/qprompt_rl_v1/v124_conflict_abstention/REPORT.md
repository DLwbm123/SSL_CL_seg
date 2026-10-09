# V124 — conflict abstention

Status: PREREGISTERED, NOT STARTED. No scientific result.

Five fixed branches: BASE, IGNORE_C, RANDOM_MATCHED, UNIFORM_MATCHED, OFF. Reuse all32 image-held-out V123 ENTRY100 states;16 conditions ×2 paired streams;32000 training updates plus24 engineering qualification updates. Evaluate704 calls /2816 image exposures only after all320 endpoints seal. No Q_dev/test/hidden U labels, new annotations, actor/RL or target replacement.

Primary claim requires IGNORE_C to pass the frozen practical/robustness/absolute gates against every comparator. Model holdout remains researcher-exposed development. Weight matching does not guarantee matched KL or gradients. V123 failure remains unchanged.

CPU source selfchecks PASS with zero image/model/optimizer calls. See [PREREG.md](PREREG.md), source and SOURCE_SELFCHECK.json.

Run ID: `v124_abstention_20261009T105359Z`. Private checkpoints/logs stay on canonical NAS. No implicit retries or enlarged scope.

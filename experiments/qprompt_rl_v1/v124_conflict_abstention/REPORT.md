# V124 — conflict abstention

Status: RUNNING. Started 2026-10-09 18:55:30 Asia/Shanghai after source publication. No scientific result.

Five fixed branches: BASE, IGNORE_C, RANDOM_MATCHED, UNIFORM_MATCHED, OFF. Reuse all32 image-held-out V123 ENTRY100 states;16 conditions ×2 paired streams;32000 training updates plus24 engineering qualification updates. Evaluate704 calls /2816 image exposures only after all320 endpoints seal. No Q_dev/test/hidden U labels, new annotations, actor/RL or target replacement.

Primary claim requires IGNORE_C to pass the frozen practical/robustness/absolute gates against every comparator. Model holdout remains researcher-exposed development. Weight matching does not guarantee matched KL or gradients. V123 failure remains unchanged.

CPU source selfchecks PASS with zero image/model/optimizer calls. See [PREREG.md](PREREG.md), source and SOURCE_SELFCHECK.json.

Run ID: `v124_abstention_20261009T105359Z`. Private checkpoints/logs stay on canonical NAS. No implicit retries or enlarged scope.

Startup: all24 charged engineering qualification updates passed, including exact original/new BASE state and all five checkpoint replays, frozen memory and no teacher gradients. Main coordinator and four GPU workers were verified alive with neutral ps/nvidia identity at18:56:38local; no immediate failure. Startup GPU admission24121MiB each, NFS probePASS with13990GiB free. Source/protocol execution commit `3267e68401a8974a63bd0ecf402ec2577c7c2ba3`; remote HEAD and anonymous PREREG HTTP200 verified before launch. Existing hourly automation updated to this fixed run; no new schedule.

Full experiment is expected to exceed30minutes and runs independently of this chat. No repeated waiting or result claim in startup delivery. On actual completion, verify costs and once-only audit and publish all anonymous results/report. Private log location is the canonical NAS protocol directory under this run ID, `worker.log`.

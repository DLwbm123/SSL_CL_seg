# PRES-DSR-SF V0.2 final report

## Outcome

The one authorized validation-only attempt is `BLOCKED_INCOMPLETE_EVIDENCE`. E1-E6 were not adjudicated, so this run supports neither `PASS_PRES_DSR_SF_FEASIBILITY` nor a scientific failure label.

The execution used code commit `09f4600348f8708ca9e865f7d5c925b6472cd013`. Its exact-source admission suite passed 108 tests, including all 55 registered PRES-DSR-SF categories, with zero failures, errors, or skips.

## Root cause

The preregistered search contains five lambda candidates and four temperature candidates. Each of the six seed/stage routers therefore emits nine CV-selection rows, for an exact total of 54. The frozen call graph and the runtime coverage assertion incorrectly required 78. The correct router-score and routing-confusion coverage values were 915 and 117, respectively. The fail-closed assertion stopped the process with `router output coverage changed`.

## Boundary reached

Before the blocker, the backend import barrier, 2,962-entry private-input audit, nine-checkpoint audit, 1,485-case raw descriptor seal, and nine train-only memory seals completed. The maximum memory size was 200 rows per seed/domain, below the registered cap of 512.

The blocker occurred before the router seal, expert-probability stage, validation segmentation evaluation, bootstrap evaluation, and E1-E6 compilation. Optimizer and autograd counts remained zero; no training or method registration occurred.

## Archive and stop

The durable parent recorded child exit code 1 and sealed 37 files totaling 3,673,752 bytes. Independent full-byte verification passed with phase content digest `26c0d59c013b81bade49b98b914efd2ce1d557b3f4cc64a80aa44504ed8f9213`.

No second attempt is authorized. The protocol stops for independent review. No test evaluation, C0 regeneration, validation refit, expert fine-tuning, LoRA, adapter, performance training, Prostate, MnMS, full sweep, or main merge was launched.

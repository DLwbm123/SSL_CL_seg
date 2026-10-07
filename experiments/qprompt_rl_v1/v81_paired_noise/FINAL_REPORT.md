# V8.1 final: paired noise reduced skips without practical gain

Both 64-group priors and all 20 development endpoints completed; training was sealed before query readout. The preregistered outcome is STOP_PAIRED_NOISE_NO_PRACTICAL_GAIN. C/D did not run. Development images are reused diagnostic data, not independent confirmation.

| Method | New Dice (%) | Old Dice (%) | Utility ×100 |
|---|---:|---:|---:|
| NATIVE | 73.4741 | 81.2819 | -7.3256 |
| FIXED_BEST | 73.7624 | 81.0925 | -7.2268 |
| UNIFORM_ACTION | 73.7753 | 81.2891 | -7.0331 |
| PRE_FROZEN_601 | 73.7553 | 81.0894 | -7.2353 |
| PRE_FROZEN_602 | 73.7051 | 81.2896 | -7.1436 |

## Mechanism and limits

The calibrated floor changed from 0.00202752 to 0.00073794. Skipped groups fell from 46/64 to 30/64 for 601 and 48/64 to 32/64 for 602. Actor optimizer updates rose from 72/64 to 136/128. The mechanism changed as intended, but utility stayed below uniform and was slightly worse than matching V8 priors. This rejects the floor-only intervention on the current diagnostic. Keep both negative results; do not tune the floor again.

Prior 601 versus uniform: new -0.0200 pp, old -0.1996 pp, utility -0.2021 (×100). Prior 602: new -0.0702 pp, old +0.0005 pp, utility -0.1104 (×100). These relative differences coexist with about five percentage points of absolute old-query degradation from frozen memory references. No target-domain, low-label or sealed-test benefit was evaluated.

All new and original native/fixed/uniform mean scores are exactly equal, consistent with unchanged matched student control paths. That check does not prove every hidden state identical; qualified full-state isolation checks supply the relevant mechanical evidence.

## Next hypothesis, not a proven cause

Policy training carries eight context states from step 100 to 900. Development ends at 300, samples actions at 100/200, and progress is step divided by registered horizon. The same absolute step therefore has different policy features and native scheduler positions. Most training decisions also follow longer accumulated adaptation than development. A separate preregistered episode-alignment diagnostic will change training to fresh registered 300-step episodes with decisions at 100/200; loss, actions, reward, controls and calibrated floor stay fixed. This is a training distribution hypothesis, not a claim that original V8 violated its protocol.

## Accounting and delivery

V8.1 spent 55,608 new student optimizer invocations and 268 new actor invocations. Cumulative V8 + V8.1: 134,437 student (37 qualification + 8,000 auxiliary + 1,600 entries + 14,400 audit + 102,400 priors + 8,000 development), 412 actor (12 qualification + 400 prior). Historical common source adds 8,000 student updates separately. No invocation failed. Complete anonymous trajectories, all 20 endpoint/channel scores and cumulative ledger are published; images, role maps and weights remain private on NAS.

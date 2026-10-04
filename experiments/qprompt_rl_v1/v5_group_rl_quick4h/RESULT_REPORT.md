# V5 quick-screen result report

## Result

All **8/8 full-horizon endpoints** completed with no failed, stopped or unstarted cells. The GRPO_FS screen did **not** show additional endpoint benefit over its offline initialization or matched PPO. Against ORIGINAL it traded lower new-domain Dice for better old-domain retention. This is a negative result for added GRPO value in this exact reduced scope.

- Original clock: 2026-10-04 18:12:57 CST.
- Computation and reporting finished: 2026-10-04 20:28:59 CST.
- Elapsed from the original start: 8161.9 seconds (2 h 16 min 2 s), including work before the scope amendment. Four-hour cutoff: 2026-10-04 22:12:57 CST.
- Seed168, controller401, two domains, G=2, one update group per learner/domain,16 actor optimizer calls per group. Full native horizons3200/2100; no shortened trajectory counted as a completed endpoint.
- Execution source `029d666885fbe80dd5b7bc0d0e7a0b00029c55e5`; retained calibration source `77cf65db6802ebbeaf2cd6b7c097fc11f0ebc2cc`.

## Endpoint scores

Dice values are percentages. New macro Dice is the equally weighted mean of rim and cup Dice; disc-union Dice is a separate diagnostic. Each cell is a single seed, not an estimate with replicated uncertainty.

| Domain | Method | New macro | Rim | Cup | Disc union | Old REFUGE | Old change from entry (pp) |
|---|---|---:|---:|---:|---:|---:|---:|
| RIM_ONE_r3 | ORIGINAL | 71.1214 | 74.1765 | 68.0663 | 89.2188 | 71.3164 | -11.8088 |
| RIM_ONE_r3 | OFFLINE_25 | 70.5787 | 73.6564 | 67.5009 | 88.9107 | 74.3830 | -8.7422 |
| RIM_ONE_r3 | PPO_MATCHED | 70.5787 | 73.6564 | 67.5009 | 88.9107 | 74.3830 | -8.7422 |
| RIM_ONE_r3 | GRPO_FS | 70.5460 | 73.6134 | 67.4787 | 88.8689 | 74.3823 | -8.7430 |
| Drishti_GS | ORIGINAL | 74.9987 | 73.4051 | 76.5924 | 93.1183 | 66.9554 | -16.1698 |
| Drishti_GS | OFFLINE_25 | 74.3410 | 72.0311 | 76.6510 | 93.7665 | 68.6058 | -14.5194 |
| Drishti_GS | PPO_MATCHED | 74.3283 | 72.0200 | 76.6366 | 93.7658 | 68.6130 | -14.5122 |
| Drishti_GS | GRPO_FS | 74.3410 | 72.0311 | 76.6510 | 93.7665 | 68.6058 | -14.5194 |

Old REFUGE entry macro Dice is83.1252%. GRPO still forgets8.7430pp after RIM and14.5194pp after Drishti; better retention relative to ORIGINAL is not elimination of forgetting. RIM rim, cup and disc-union all decline relative to ORIGINAL. Drishti rim declines, while cup and disc-union improve; its rim/cup macro still declines.

## Paired differences: GRPO_FS minus comparator

These are equal-weight averages of **two domain cells**, not two independent subject groups or repeated seeds. Units are percentage points.

| Comparator | New macro difference | Old REFUGE difference |
|---|---:|---:|
| ORIGINAL | -0.616561 | +2.358138 |
| OFFLINE_25 | -0.016320 | -0.000387 |
| PPO_MATCHED | -0.009963 | -0.003976 |

## What the controller diagnostics support

GRPO made16 actor updates per domain, with nonzero gradients and nonzero KL from initialization. On the fixed diagnostic panel, GRPO argmax agreement with initialization was98.4127% for Drishti and100% for RIM (KL0.0005895 and0.0001243 respectively). This diagnostic panel is distinct from the endpoint rollout.

For endpoint deployment, action-count histograms for [0,0.125,0.5] were Drishti GRPO=[2,4,78], OFFLINE=[2,4,78]; RIM GRPO=[98,0,30], OFFLINE=[99,0,29]. Drishti endpoint scores matched OFFLINE exactly. Matching histograms alone do not prove matching action sequences. RIM GRPO was0.032641pp below OFFLINE on new macro Dice.

Both learners sampled the same two development reward values in each domain at their common initialization. These rewards belong to rollout collection before the only policy update; they are not evidence that the updated policy improved. GRPO fixed-scale advantage standard deviations were0.10757 (Drishti) and0.64469 (RIM). No sampled update ratio exceeded the clipping interval, so these receipts do not support attributing the small change to active PPO ratio clipping. The observed change is small; this screen cannot establish why more training would or would not help.

## Cost and checks

- Performance student updates74200: calibration26500, reference5300, development21200, endpoints21200.
- Native qualification690 and stopped source169 calibration4202 remain charged: **79092 total new student updates**, all recorded attempts matched successes.
- Actor64, critic32, critic warmup128 optimizer calls; synthetic qualification65916 is separate and already includes the additional16-call G2 check. Historical source training24000 is reused and separate.
- Full horizons, unique eight-cell coverage, metric arithmetic, paired differences and attempt/success accounting checked. Eight separate val jobs launched after the complete endpoint lock at20:27:19 CST; computational completion was20:28:59 CST. Coordinator exited; no automatic continuation.
- Resource receipts include per-job wall time, peak allocation and evaluator costs. CUDA intervals include host gaps and are not exact GPU kernel busy time. Detailed operation/interval counters cover recorded scopes; retired4202 student updates are charged, but their unrecorded detailed operation/time breakdown is not reconstructed. Do not interpret incomplete interval sums as total GPU-hours.

## Scope limits and publication

One segmentation seed, one controller seed, only one G2 update group and historical development-val patients. No independent-patient confirmation, seed-level standard deviation, stable-advantage claim, Bandit comparison, reward/phase shuffle, or multistep-credit conclusion. ORIGINAL KI identity remains unverified. The original full V5 matrix remains superseded and incomplete. No automatic promotion, retuning or additional training is authorized by this result.

This folder publishes source, frozen scope, anonymous aggregate scores, reward/policy summaries, cost and completion receipts. Patient/case data, private features, raw trajectories/logs and checkpoints remain on NAS. `FINAL.json` preserves the computation-time private-completion snapshot; GitHub delivery is verified separately after the report commit.

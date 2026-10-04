# V5 engineering qualification — PASS

Verified at 2026-10-04 18:24:24 CST. Execution source `77cf65db6802ebbeaf2cd6b7c097fc11f0ebc2cc`. This is engineering qualification and startup evidence, not a segmentation performance result.

## Native qualification

- 690/690 physical student optimizer calls; attempt/success accounting passed with no duplicates or failed student transactions.
- Original V4 390-step parity/zero-U/two-case/25-step resume suite passed.
- Both domains passed two exact replays of the 75-step 0/.125/.5 mixed sequence, including complete student/teacher/Adam/scheduler/scaler/RNG/Provider state.
- Feature extraction preserved full state. Provider is stateless by segmentation seed and cursor.
- Current private fit/online/audit identities exactly equal the V4 role files; RIM counts 12/2/2 and Drishti 6/2/2. Test and hidden U labels remain sealed.
- Group reward translation, population std, equal-reward entropy handling, GAE terminal boundary, clip direction, reward multiset, optimizer continuation and U0 exclusion checks passed. Inputs/behavior probabilities/rewards/advantages are detached; controller code receives CPU features only. Runtime student-state immutability is also asserted after group updates.

## Fixed synthetic learning checks

Each task/method used 256 groups, with no real feedback or validation input. The reported probability is for the predeclared correct synthetic action.

| Task | Method | Before | After |
|---|---|---:|---:|
| immediate | PPO_MATCHED | 0.333333 | 0.986463 |
| immediate | GRPO_STD | 0.333333 | 0.997807 |
| immediate | GRPO_FS | 0.333333 | 0.996917 |
| delayed | PPO_MATCHED | 0.333333 | 0.949782 |
| delayed | GRPO_STD | 0.333333 | 0.993702 |
| delayed | GRPO_FS | 0.333333 | 0.994057 |
| offline_RIM_ONE_r3 | PPO_MATCHED | 0.099552 | 0.991469 |
| offline_RIM_ONE_r3 | GRPO_STD | 0.099552 | 0.996285 |
| offline_RIM_ONE_r3 | GRPO_FS | 0.099552 | 0.995871 |
| offline_Drishti_GS | PPO_MATCHED | 0.034819 | 0.980903 |
| offline_Drishti_GS | GRPO_STD | 0.034819 | 0.991205 |
| offline_Drishti_GS | GRPO_FS | 0.034819 | 0.992531 |

New synthetic tests consumed 49,200 actor and 16,416 critic optimizer calls; the reused V4 self-check consumed 268 calls, and the RNG-isolation check consumed 16. **Total synthetic calls: 65,900**, separate from 690 student updates.

## Explicit setup correction

The first attempt failed during native construction, before any student update, because the CuBLAS deterministic workspace environment was absent. Successful synthetic evidence and all 65,884 prior calls were retained; none were repeated. The corrected launch sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`. The reward-shuffle permutation now uses a separate RNG from minibatch ordering and passed its added 16-call check. Failed setup/source/receipts are preserved on NAS; no performance feedback prompted either correction.

## Reward-free initialization

| Domain | States | Alpha | Entropy before | Entropy after | Argmax agreement |
|---|---:|---:|---:|---:|---:|
| RIM_ONE_r3 | 192 | 0.176419057 | 0.140219763 | 0.878889859 | 100% |
| Drishti_GS | 126 | 0.163366355 | 0.092523538 | 0.878889799 | 100% |

Target entropy is 0.8 log(3), with FP32 rounding at about 1e-7. Both scalings used the fixed complete V4 development panel and zero reward accesses.

## Execution status

P1 calibration is running on physical GPUs 5/6/7. The first workers made real student updates and passed process-name/GPU mapping checks. No pilot or confirmation results exist at this startup snapshot.

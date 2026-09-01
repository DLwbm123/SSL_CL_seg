# PRES-DSR-SF V0.2.1 final report

## Outcome

The single authorized validation-only attempt completed and is scientifically adjudicated as `FAIL_SOFT_EXPERT_FUSION_VALUE`. E1, E2, E3, E5, and E6 passed; E4 failed. This is a scientific failure, not an engineering blocker.

The execution used source commit `58ee45b12aae662c8fe61595dc4068094c783f7c`. The exact-source admission suite passed 136 tests, including 83 PRES-DSR-SF cases, with zero failures, errors, or skips. The durable parent recorded child exit code 0.

## Registered gates

| Gate | Result | Registered evidence |
|---|---:|---|
| E1 backend and control | pass | Deterministic backend barriers and clean M1/M2 controls passed; maximum control difference was `3.60901036556e-08` against tolerance `1e-6` |
| E2 snapshot oracle value | pass | Three-domain gain `0.165627`; historical gain `0.248441`; 3 positive seeds; maximum domain drop `0` |
| E3 discriminative routing | pass | Ridge-hard stage-1 macro `0.988333`, minimum domain `0.983333`; stage-2 macro `0.960000`, minimum domain `0.933333` |
| E4 soft expert fusion | fail | Oracle gap `0.005587`, shared gain `0.160040`, historical gain `0.252128`, and gain over M1-hard `0.041137` passed; maximum seed-domain drop `0.039803 > 0.020` and current-domain drop `0.039803 > 0.010` failed |
| E5 stability | pass | Hard macro p10 `0.953000`; soft gain p10 `0.151947`; soft oracle-gap p90 `0.013680`; all domains nonempty and all values finite |
| E6 isolation and memory | pass | All 12 model guards and all 9 checkpoint before/after checks passed; maximum memory size was 200, below the frozen cap of 512 |

Controls do not rescue the failed primary E4 gate.

## Coverage and isolation

The recovered call graph completed 186 descriptor forwards over 1,485 cases and 189 expert forwards over 1,485 case-expert passes. It performed 936 closed-form ridge fits, 75 M1 CV prototype fits, 168 clean-control prototype fits, 30 primary bootstraps, and 60 clean-control bootstraps. The exact registered output coverage was 78 CV rows, 915 router-score rows, 117 routing-confusion rows, 27 cross-expert rows, 120 soft-fusion rows, 90 bootstrap rows, and 9 memory rows: 1,356 rows total.

Candidate predictions were sealed before the 495 evaluator-only validation-GT reads. Optimizer, autograd, backward, parameter-gradient-write, training, and test-GT counts were zero. Router fitting was closed-form CPU float64; no method was registered.

## Archive and hard stop

All 14 ordered stage barriers passed. The durable phase contains 181 files and 4,385,980,959 bytes; the final create-only private bundle exactly covers 183 files and 4,386,018,614 bytes with content identity `05c9008ad4496ccbdc51df6103638024d49fae4b3b4cdc2a9f829c5f3ab165bb`.

The image-only domain-agnostic routing line stops here for independent review. No test evaluation, second V0.2.1 attempt, C0 regeneration, hyperparameter expansion, validation refit, expert fine-tuning, LoRA, adapter, performance training, Prostate, MnMS, full sweep, or main merge is authorized.

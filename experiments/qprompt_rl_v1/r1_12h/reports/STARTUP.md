# R1 startup receipt

Status: RUNNING, not completed. Run ID: r1_12h_20260925T113152Z.

The 12-hour window started 2026-09-25 19:36:10 Asia/Shanghai and ends 2026-09-26 07:36:10. Source E0: edee1ab84175e73c9e055ce904a93831ef7f2a9e. Active training source E1: 41dfa243f15b184d728c19422a9f23ddcb346de9.

All three CPU contract checks passed. Both backbones passed CUDA state/RNG restore and deployment equality checks, then real labeled-data smoke. Physical qualification calls: 14 synthetic (including six consumed before repair), eight smoke, zero recovery at startup. Four formal prefix workers have committed updates; this is execution evidence, not endpoint or utility evidence.

E1 changes strict CUDA determinism to warn-only because the existing cross-entropy and bicubic-backward kernels have no deterministic implementation in this environment. Fixed seeds, initialization, paired sample schedules, augmentation choices and scientific losses remain fixed. Exact cross-run numerical reproducibility is not claimed. No formal updates preceded E1.

A detached service and persistent queue run independently of the client connection. Authorized five-minute maintenance checks inspect failures and repair/recover within this R1 scope. Final results, missing endpoints, resource measures and patch history will be published after the session ends. No R2/R3 authorization.

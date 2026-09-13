# DPR finite response V0.2 launch

Status: **RUNNING_NOT_COMPLETE**. This is a launch/qualification record, not an accuracy result.

Execution commit `91cb00bf2ac7458d82895256fbf71d91af84529d`; new branch `codex/dpr-finite-v0-2`; run ID `dpr_finite_v0_2_20260913_01`. Started 2026-09-13T10:33:00.740623+08:00 on GPU6/7 only, as a detached NAS-backed parent with two neutral-command workers. The first startup check observed committed steps for both seed61/O1 arms and no immediate failure. Closing this session does not stop the queue.

The fixed matrix contains12 target tasks and31,800 formal optimizer updates. Six original SRC_CE students, F_CONV/F_FULL, existing STATIC_SOURCE and all18 DPR V0.1 records are immutable references. No source or baseline retraining. Historical DPR SMALL_POSITIVE_PRIMARY_SIGNAL and CIST terminals are preserved. Main is DPR_FINITE_U only; DPR_FINITE_SIGN_U is its fixed sign control.

Qualification on the exact execution source passed locally and on CUDA, including independent dense KKT, detached raw-position VJPs, non-contamination of supervised gradients, the full-field counterexample's exact Adam fallback, final pending recovery, illegal raw/candidate recovery rejection, native warmup parity, two-case overfit and384/B1 execution. A zero-training fixture also passed all24 queue dispatches and final reporting paths. Real-L-only smoke consumed8 discarded optimizer updates and0 U/val/test accesses. Peak smoke reservation was954MiB; admission requires1961MiB including headroom.

Cumulative qualification cost is78 synthetic optimizer updates:26 in an earlier local pass,26 on final local source,26 on final CUDA source. All attempts remain retained. Eight real smoke updates are separate. Synthetic fixture U records in the qualification counters are generated data, not real-U accesses.

Each new run compares its epoch20 student/Adam/EMA/order hashes with the historical F_CONV warmup records. All12 final students must seal before isolated final validation. The queue then generates aggregate results, terminal, actual cost and NAS archive evidence and stops. GitHub result publication follows completion verification; it is not yet complete at launch. No automated parameter search, extra seeds, CIST recovery or historical-KI search.

Public files contain code, frozen contract, aggregate qualification counters, reference identities and launch evidence. Patient-level records, data and model checkpoints remain private on NAS. Runtime checkout remains at the exact execution commit even when later public documentation commits are added.

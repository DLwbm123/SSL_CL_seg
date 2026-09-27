# Patch log

## E0 — e179e8df5b75acc545d8d69a206c10879d496d69

Initial authorized implementation frozen before CUDA qualification. No post-launch scientific or engineering patch yet. CPU mock budget grant: 1, executed optimizer calls: 0. Native qualification passed both backbones, including exact complete-state restoration; 8 synthetic student, 24 synthetic controller, and 16 real L-only smoke calls. R3a completed 320 disposable student calls. All budgets remain cumulative.

The inherited native runtime uses deterministic algorithms in warn-only mode. CUDA cross-entropy emitted the known nondeterministic-kernel warning; the paired null qualification passed the frozen numerical tolerance. Bitwise repeatability of independent CUDA updates is not claimed.

## P1 — rollback optimizer tensor alias (scientific impact)

The evaluator caught REG optimizer step 3500 at retained suffix 1200 (expected 3200). PyTorch optimizer loading reused CPU step tensor storage from the reusable candidate snapshot. Disposable steps therefore advanced the snapshot counter, changing later Adam bias correction. Clone optimizer state before both student and controller loads. A zero-optimizer regression populates Adam state, restores, mutates live counters/moments and verifies the snapshot remains unchanged.

All R3a evidence and started adaptive R3b trajectories are invalidated and archived, then restarted from their bound supervised prefixes with corrected reward scales. Prior physical grants remain in the ledger; duplicate transaction IDs charge replay. Non-adaptive trajectories are unaffected and resume only with explicit old/new commit and checkpoint hash bindings plus full state equality. Native synthetic qualification is repeated within remaining caps. The already completed 16 L-only smoke calls are bound and reused, never repeated beyond their cap.

Coordinator repair also distinguishes an intentional engineering pause from a worker crash so paused siblings do not acquire spurious failure fingerprints. Original deadline and every cap remain unchanged.

## Final closeout

P1 training commit 195448b93dafb89fda5e05d0a53f997c39ed9ae1 completed the registered matrix. Read-only closeout audits all 108 endpoint states and merges current, qualification, and invalidated P1 ledgers; no new optimizer calls. All physical attempts have success receipts, all caps and the original deadline are respected. Final metrics exclude invalidated P1 trajectories. Primary gain is not established.

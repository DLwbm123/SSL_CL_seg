# Legacy PAS bank recovery: prospective bounded feasibility v1

This is **not a repair to the frozen Gate 1C v2 result or an admission run**.
The user authorized continuing the current method and fixing actual code/data
artifact defects. The missing bank is a persistence gap: a pre-prototype best
checkpoint contains `None`, and the original stage-end in-memory bank was not
saved separately. No placeholder or alternative bank is admissible in old v2.

The companion JSON is the executable scope/budget contract. Publish this plan
before any replay forward or optimizer step. Bind its commit and both file
SHA256 values, the later helper commit, source/config/input hashes, full command
and runtime in each new run. Existing originals and failures remain immutable.

## Source-grounded bounded replay

Resume the complete frozen `B0/seed1/stage1` best checkpoint at stage1 epoch1,
global step3208, using the unmodified **fb55e802** code and config in a clean
detached checkout. The frozen manifest has 16 labeled cases, batch size2.
Epochs1 through25 therefore require exactly **200 supervised optimizer steps**.
The archived trace independently contains exactly steps3209 through3408.
The preserved launch scripts and payload establish one visible CUDA device
(original physical GPU1); each replica exposes exactly one of the two identical
3090 GPUs and restores that single saved RNG stream as logical cuda0.

Run one replica per GPU, at most 200 updates and 30 minutes each; no retry or
parameter search. Both reuse the original shared runner, teacher/student/GAS,
Adam, polynomial schedule, transforms, RNG restoration and `warn_only` numerical
policy. Do **not** apply the diagnostic deterministic-CE repair to baseline
replay. Keep the original validation evaluations; use their GT only inside the
evaluator. Stop immediately after the original `compute_single_prototypes`
returns at epoch25, before any unsupervised update, stage-end test or transition.
Save the candidate bank and complete capture state in separate new directories.

Every new supervised row must match the original identity/schedule fields
exactly and numeric loss/LR fields at the existing Gate0 **atol=rtol=1e-6**.
Stop on the first discrepancy. Compare both replica banks and complete captured
states bitwise. Verify their current-domain case-order traces and label-role
isolation. Failures and numerical differences are retained, never hidden by a
more favorable retry, tolerance, device or checkpoint.

## Interpretation and limitations

Passing both replicas supports reconstruction from a frozen full resume state;
it does not prove bytewise identity to a historical bank whose hash/file was
never saved. The archived training trace contains losses/indices, not historical
per-batch tensor hashes; report that residual limitation explicitly. The new
case trace checks replica ordering but cannot independently recover an absent
historical case/augmentation fingerprint.

The recovered bank is not written into any frozen checkpoint. Consuming it
requires a **new prospective Gate1C input/protocol version**, leaving all original
models, K2, cases, thresholds and old failures identifiable. The old v2 attempt
remains incomplete; its 265 cached cases cannot be mixed into a revised run.
Any current-method training updates here are baseline recovery only, separately
counted from Gate1C's zero model/transport updates. Gate1B and overall original
Gate1 remain `FAIL_TRANSPORT_NOT_SUPPORTED` regardless of this feasibility result.

If either trace or replica comparison fails, retain
`BLOCKED_LEGACY_PAS_RECONSTRUCTION_NOT_VERIFIED`; do not expand the budget or
silently change R1. No new DI-DMPA training, Gate2, full sweep, package install,
paid resource, main merge or frozen HDF5/manifest/split modification is included.

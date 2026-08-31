# Model-Fisher EWC V1: synthetic engineering result

**Status: PASS_SYNTHETIC_ENGINEERING. This synthetic phase is closed.** The third and final registered invocation passed all six groups with zero skips. This is engineering admission for the separate `model_fisher_ewc_v1` implementation. It is not Fundus performance evidence, an EWC efficacy result, independent scientific review, or overall project success.

No real image, label, checkpoint, U-Net forward, or GPU was used. The engineering-only coefficient/decay pair `1.7/0.6` tested arithmetic and state accumulation; it is not a recommended Fundus setting.

## Registered invocations

The protocol allowed at most three complete invocations and required every failure to remain available. All three were used:

| Invocation | Source commit | Child exit | Passed | Result | Finding and fixed action |
| --- | --- | ---: | ---: | --- | --- |
| 1 | `c22bf532dcf8c26985418ac0a64dfa8f5f065052` | 1 | 5/6 | `FAIL_SYNTHETIC_ENGINEERING` | The check fixture captured its expected RNG state before constructing random fixture tensors. Commit `ce9bc1a` moved the capture after fixture construction without changing the registered estimator or gates. |
| 2 | `ce9bc1ad93d1987d8fbfc212559023eeeed07457` | 1 | 5/6 | `FAIL_SYNTHETIC_ENGINEERING` | The stronger state check exposed a method bug: an invalid consolidation count was assigned before active-stage validation. Commit `ab1c40c` validates the complete payload before assigning any method state. |
| 3 | `ab1c40c312dcbebe5b6fd025bc17bb6fe02d68eb` | 0 | 6/6 | `PASS_SYNTHETIC_ENGINEERING` | All frozen groups passed; no fourth invocation is allowed or present. |

Each invocation used 402 counted toy-model calls, 734 images through toy models, 295 `autograd.grad` calls, 117 backward calls, and 117 synthetic optimizer updates. The cumulative totals were 1,206 calls, 2,202 images, 885 gradient calls, 351 backward calls, and 351 synthetic optimizer updates. Every per-invocation and cumulative registered limit passed.

## Final gates

- Closed-form three-class pixel Fisher matched the expected bias diagonal `(0.16, 0.21, 0.25)`, included background and all labels, and left the deliberately unused parameter at zero.
- Empty, short, capped, one-image and two-image cases used the exact actual pixel denominator and were invariant to incidental minibatch partitioning.
- Eight invalid checkpoint/state forms were rejected. Successful and failed estimation preserved model parameters, buffers, gradients, optimizer state, pre-existing EWC state, module modes, and RNG state; failed loading was atomic.
- The quadratic penalty, gradient, first-stage shared objective, golden repeatability, shared-Trainer backward, and online accumulation passed.
- The shared runner's uninterrupted eight updates exactly matched interruption after update six plus two resumed updates, including checkpoint tensors and RNG state.
- The fixed two-case synthetic overfit reduced supervised loss from `1.5638391971588135` to `0.02059922367334366` in 100 updates and reached pixel accuracy `1.0`.

The exact final result SHA-256 is `8cdfafb4f5a5b6635cc4272dff4e8d787095648f93766508ae9114c5790b7db2`. The two retained failure-result hashes are `b6ab14b51d30a4c66b0e90a97706e54a14bee37118b1d6710c417f6632af47b2` and `207131b39a5547b3eb62e379994df29410dcd66599ec44c42fd73312b3b37d67`.

## Artifact audit and archive

The first zero-model artifact-audit attempt exited 1 because a string exclusion for `transform` matched the accessor docstring phrase "applying transforms." It did not identify an implementation defect. That failed audit operation is retained. A create-only second audit used Python's AST to inspect actual calls and attributes, exited 0, and independently recomputed invocation exits, counts, budgets, gates, fix scopes, source hashes, and closed-study invariants. It made zero model forwards, gradient calls, or optimizer steps. See [ARTIFACT_AUDIT.json](ARTIFACT_AUDIT.json); its SHA-256 is `7ebf8d48649d24db8f5487006fd6526ad23c18001b91dbee9806b5ecb5dc35e1`. This is an artifact and arithmetic audit by the same agent, not independent scientific peer review.

The additive private archive is `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/ewc_model_fisher_engineering_v1_20260901/archives/324332da9ff7ffc229f473bcae984a10b42fe06003a2575200511a3bae13dd7a`. Its `evidence.tar` includes all three immutable execution checkouts, all successful and failed operation evidence, both audit attempts, exact audit procedures, and the three root admission records. It contains 12,181 regular files, 1,175,801,997 logical source bytes, zero symlinks, and six preserved hardlink groups. Every source and tar member was hash-verified before atomic promotion; originals remain in place. The sealed four-file bundle is 1,207,413,373 bytes. See [ARCHIVE_RECEIPT.json](ARCHIVE_RECEIPT.json). It is a verified copy on the same NAS, not an independent-device backup.

The legacy EWC file remained at SHA-256 `a832e123c9efc44d474de99d47f63447f28c7baa05d3ae314dcda24f9d706724`. The closed source-audit and LwF result hashes were unchanged, and `/home/jiangsuiyang/SSL_CL/runs` remained the required NAS symlink. No frozen HDF5, manifest, split, checksum, historical method, or closed study was changed.

## Next registered boundary

No more synthetic invocation is admitted under this registration. Before any real Fundus read or model forward, a separate prospective contract must fix the coefficient and normalization convention, U-Net/real-loader admission checks, model-forward and optimizer-update budgets, seeds, matched sequential-SSL control, terminal-stage policy, evaluation roles, success gates, and failure exit. The synthetic values `1.7/0.6` and the closed LwF readout cannot be used as tuning recommendations.

The 30-minute heartbeat may prepare and audit that bounded contract. It must not start real-data computation until the new contract is published, and it must continue to preserve PMGC/LwF/Gate1C/MMPR closure, the ended prototype-derived line, and the prohibitions on C0, Gate2, Prostate, MnMS, unbounded search, paid resources, or main merge.

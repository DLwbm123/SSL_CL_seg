# PRES-JASCL V0.1 final report

## Outcome

The single authorized validation-only attempt is `BLOCKED_PROTOCOL_OR_LEAKAGE` with D5=false. It produced complete M1/M2 evidence, but no registered scientific pass/fail conclusion and no selected M may be claimed.

Formal execution source: `977e0139c6f8ee9ac795606a19ab155b9dd4bd72`. The durable child exited 0. The frozen call graph completed 162 router forwards over 1,287 cases, 189 expert forwards over 1,485 case-expert passes, 60 bootstrap operations, 12 immutable model guards, 495 evaluator-only validation-GT reads, and 2,031 registered CSV rows. Optimizer, autograd, backward, parameter-gradient-write, training, and test-GT counters were all zero.

## Registered gates

| Gate | M1 | M2 | Non-adjudicable observation |
|---|---:|---:|---|
| D1 snapshot-oracle value | true | true | Three-domain gain 0.165627; historical gain 0.248441; 3 positive seeds; maximum drop 0 |
| D2 domain routing | false | false | Stage2 macro 0.853333 / 0.848889, both below 0.90; M2 Stage1 macro 0.942500 is below 0.95 |
| D3 routed segmentation | false | false | Oracle gaps 0.046725 / 0.042668 and maximum seed-domain drops 0.154087 / 0.050178 exceed 0.010 |
| D4 stability | true | false | M1 prototype cosine median 0.999940; M2 minimum occupancy 0.097561 is below 0.10 |
| D5 isolation/protocol | false | false | Official JASCL import changed the registered cuDNN benchmark flag |

These D1-D4 values are diagnostic only because blockers take precedence. Even without the D5 blocker, neither M satisfies all D2-D4 gates in the observed run.

## Engineering blocker and bounded fix

A zero-forward reproduction established the exact transition: before the pinned JASCL model import, `cudnn.benchmark=False`; immediately after import, it was `True`; after the PRES-local guard, it returned to `False`, with deterministic algorithms, cuDNN determinism, TF32-off, and autocast-off all satisfied. The guard and its two passing regression checks are in commit `0c983666da4458d96450ef8121427a823cbaa3b4`.

The guard source has not been used to execute another formal attempt. The existing authorization permits one attempt only, so `attempt2_authorized=false` remains binding.

## Inputs, tests, and archive

The frozen input audit passed for 14,470 files (17,712,127,650 bytes), nine exact B0 checkpoints, and 2,962 data checksum entries before any real forward. The formal source suite completed 51 tests in 2.21 seconds with 0 failures, 0 errors, and 0 skips.

The private archive audit passed: 56 durable phase files (3,441,549 bytes) were preserved; the final create-only manifest covers 58 files (3,452,867 bytes) with content identity `83b480a74acc0fe218c31bfeaf5e5b96617b94733ebcb37f3cbcfd52a148631b`. Raw private evidence remains outside Git.

## Hard stop

No second attempt, test evaluation, data regeneration, training, adapter/LoRA work, MILE reproduction, other benchmark, sweep, Gate2, main merge, or scientific claim is authorized. Independent review or a new explicit retry authorization is required.

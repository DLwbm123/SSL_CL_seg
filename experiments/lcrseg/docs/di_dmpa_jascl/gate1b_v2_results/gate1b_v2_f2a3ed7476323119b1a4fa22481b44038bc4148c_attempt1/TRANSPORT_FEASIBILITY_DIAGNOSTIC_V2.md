# Gate 1B v2 transport diagnostic

Result: **FAIL_TRANSPORT_NOT_SUPPORTED**.

Primary: B0 previous/current EMA, frozen UNet decoder.dec1 post-ReLU16-D; K=2. Gate1A v2 was not rerun.

Selected transport: `T0_identity` (DOWNSTREAM_FALLBACK_ONLY_NOT_EXECUTED). T1 is only a comparator.

| Gate | Result |
| --- | --- |
| B1 | PASS |
| B2 | PASS |
| B3 | FAIL |
| B4 | FAIL |
| B5 | FAIL |
| B6 | PASS |
| B7 | PASS |

B1 held-out full-support error (unrounded):

- RIM_ONE_r3: T0=0.021627976823300834, T2=0.011189261679320597, relative reduction=0.482648711401157.
- Drishti_GS: T0=0.006605478088795815, T2=0.003676713349901207, relative reduction=0.4433842182994092.

B3 immediate foreground angular mean: T0=0.2495489872953954, T2=0.232267583726251, relative reduction=0.06925054578036859.

All B4/B5/B6 per-unit values, T0/T1/T2 metrics, support masses, oracle convergence and full optimizer traces are in the JSON/CSVs.

Coverage:12 paired units /638976 registered pairs;6/6 maps;9/9 historical-val evaluator units;6000 transport updates;0 segmentation model updates. Pair counts: {'AA': 638974, 'A_NULL': 0, 'NULL_A': 2, 'NULL_NULL': 0}.

Oracle warning records: 0; no restart/iteration extension or replacement. Finite nulls remain in all applicable metrics.

GT usage: current-domain train_unlabeled fit has no label access; historical-val membership is diagnostic_evaluator_only; hidden-GT training=none; final-test-GT=none.

Frozen inputs, complete model/classifier/GAS/buffer state and all9 B0 checkpoint files are unchanged. Exact-code tests and every cache/artifact hash accompany this report.

This is mechanism admission only, not evidence of segmentation performance improvement. No method registration, Gate1C, reliability, gradient conflict, teacher noise, theory final, training, other benchmarks or main merge.

Freeze commit `58f19e968700bd7708ec00e44a11759b48ce756f`; preregistration `b20f186deff287843f3c9f18bf4ab5633908f441`; authorization `c6f72b86fdfa3683a6e2c7dbf593f73cab74c592`; exact code `f2a3ed7476323119b1a4fa22481b44038bc4148c`.

Report commit: resolve the commit first adding these exact bytes; the publication receipt records it separately to avoid a self-referential hash.

**STOP_FOR_INDEPENDENT_REVIEW**.

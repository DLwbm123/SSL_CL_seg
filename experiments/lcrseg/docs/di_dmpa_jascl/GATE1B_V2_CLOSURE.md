# Gate 1B v2 closure

The independent review accepts **FAIL_TRANSPORT_NOT_SUPPORTED**. No Gate 1B rerun, additional transport optimization, T1 rescue, changed regularization, or alternative transport is authorized.

| Gate | Frozen result |
| --- | --- |
| B1 | PASS |
| B2 | PASS |
| B3 | FAIL |
| B4 | FAIL |
| B5 | FAIL |
| B6 | PASS |
| B7 | PASS |

B3 reference = 0.2495489872953954, candidate = 0.232267583726251, relative reduction = 0.06925054578036859; the required reduction is 0.10. All 12 B4 units and all 9 B5 units, including the failures, are retained without rounding in the closure JSON and the complete B1–B7 freeze.

Identities: preregistration `b20f186deff287843f3c9f18bf4ab5633908f441`; authorization `c6f72b86fdfa3683a6e2c7dbf593f73cab74c592`; exact code `f2a3ed7476323119b1a4fa22481b44038bc4148c`; report `959e62df5608fe170f6702a7fd1a1f2a42eec8ad`; receipt `4ea4d7723db9cd29295ab000707c7bbb0044d0dc`. Formal artifact manifest SHA-256: `26e69d13935133b1cfa4e3176ff5555ba8bef73755fd8fe3c0157505a92e0ea2`.

Selected transport remains `T0_identity`, historically recorded as `DOWNSTREAM_FALLBACK_ONLY_NOT_EXECUTED`. Gate 1C has not run. R4 is unavailable; no T2 output may feed Gate 1C. The 6,000 completed transport optimizer steps belong exclusively to historical Gate 1B; model optimizer steps and this gate's transport steps remain zero.

Gate 1A's K=2 and B0-EMA identity are preserved. “Drift-calibrated” is not an allowed method claim. Method registration, training and main merge remain false. No historical Gate 1A/Gate 1B file is changed by this closure.

Next: separately publish Gate 1C v2 preregistration and execution authorization, then exact diagnostic code, before any real diagnostic execution.

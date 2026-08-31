# Gate 1C v3: identity-history reliability

**FAIL_IDENTITY_HISTORY_RELIABILITY_NOT_SUPPORTED**

The complete offline diagnostic uses frozen K=2 B0-EMA prototypes with identity history. R4 is unavailable. No T1/T2 output was used.

| Gate | R3 pixel normalized | R3 class balanced |
| --- | --- | --- |
| C1 | PASS | PASS |
| C2 | PASS | PASS |
| C3 | FAIL | FAIL |
| C4 | FAIL | FAIL |
| C5 | PASS | FAIL |
| C6 | FAIL | FAIL |
| C7 | PASS | PASS |
| C8 | PASS | PASS |

Validation:9/9 units (18 foreground units); gradient probes:72/72; teacher draws:576/576. All candidate and control evidence is retained, with explicit unsupported coverage and undefined zero-gradient cosines.

Selected reliability: `None`; normalization: `None`.
Reduced candidate: `NOT_ELIGIBLE`. **Overall Gate1 remains FAIL_TRANSPORT_NOT_SUPPORTED**.

All numeric comparisons use unrounded values. Every gradient admission comparison uses pixel-normalized R1 under the active input contract. Shared validation points and all C1–C8 raw values are in the status/diagnostic JSON and tables.

Model/checkpoint immutability:297 complete guards PASS; all9 B0 checkpoint disk hashes unchanged. Model optimizer steps=0; transport optimizer steps this gate=0; no EMA/GAS/prototype update, backward or parameter.grad write.

GT is isolated: current labeled GT only for the supervised reference; current val GT only in the diagnostic evaluator; hidden unlabeled GT and test GT usage both none.

R0/R2, class-balanced controls, posterior-mean teacher and offline PoE are reported separately. Only independently passing R3 class balancing can be selected after pixel R3 fails. Controls never rescue an R3 failure and no method is registered.

Preregistration `4a21799bc50e4e6c644a89b08808e59de525d6a9`; authorization `9c673bd745a0d35a6c25f772c7aca58502d0e572`; exact code `db4af88eca0dca48025f8884bf7f85e068eabf2a`.

Report commit is resolved in a separate publication receipt. Exact commands, test evidence, all warnings, caches, model audits and SHA-256 artifact manifests accompany this report.

**REPORT_AND_HARD_STOP_NO_METHOD_IMPLEMENTATION**. No reduced-method implementation, DI-DMPA training, Gate2, theory final, Prostate/MnMS, full sweep or main merge in this diagnostic.

Fresh v3 execution: 990 validation forwards, 75 separate integration forwards and 1800 formal forwards; total 2865. All 495 validation caches, raw native tensors/PAS intermediates, 9 validation guards, 12 integration guards and 288 formal guards were generated in this protocol. R1 reads the direct historically hashed PAS bank; no reconstructed bank or old private cache/golden is used. Reduced-method candidates are decisions for a separate preregistration only; no method or C0 is implemented here.

Reduced method candidate: NONE. Historical-bank claim allowed: False. A passing pixel-normalized R2 can nominate a current-only future candidate but cannot change the R3 Gate1C status.

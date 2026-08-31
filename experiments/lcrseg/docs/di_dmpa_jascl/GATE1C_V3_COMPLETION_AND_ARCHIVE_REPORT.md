# Gate1C v3 completion and archive report

**FAIL_IDENTITY_HISTORY_RELIABILITY_NOT_SUPPORTED**. Engineering execution and complete private byte archive passed.

Reduced future candidate: `NONE`; historical-bank claim allowed: `false`. Overall Gate1 remains **FAIL_TRANSPORT_NOT_SUPPORTED**.

| Condition | R3 pixel | R3 class balanced |
| --- | --- | --- |
| C1 | PASS | PASS |
| C2 | PASS | PASS |
| C3 | FAIL | FAIL |
| C4 | FAIL | FAIL |
| C5 | PASS | FAIL |
| C6 | FAIL | FAIL |
| C7 | PASS | PASS |
| C8 | PASS | PASS |

For primary R3, macro AURC decreased by 20.0305% and 12/18 foreground units improved. However, 17/18 units failed the registered foreground-retention guard. Negative global gradient cosines were 39/72 versus R1's 43/72, a 9.3023% reduction against the required 20%. Stage1 (RIM_ONE_r3) median cosine worsened by 0.190000950, exceeding the allowed 0.05. These failed requirements determine the verdict; the passing ranking/median-aggregate results do not override them.

The unchanged criteria require C1: at least 10% relative macro AURC reduction or 0.01 matched precision increase; C2: at least 12/18 improving foreground units; C3: every unit retains at least 80% of the reference fraction at the registered operating point; C4: at least 20% fewer negative global gradient cosines with a nonzero reference count; C5: at least 0.05 higher median cosine; C6: no stage worsens by more than 0.05; C7/C8: immutability and GT isolation. Undefined required comparisons cannot pass.

R3 pixel-normalized observed values (nulls retained):
```json
{
  "C1": {
    "candidate_macro_AURC": 0.2726041752962498,
    "matched_precision_delta": null,
    "reference_macro_AURC": 0.34088502366165246,
    "relative_AURC_reduction": 0.20030462949635253
  },
  "C2_improving_units": 12,
  "C3_failed_units": 17,
  "C4": {
    "candidate_negative_count": 39,
    "candidate_undefined": 0,
    "reference_negative_count": 43,
    "reference_undefined": 0,
    "relative_reduction": 0.09302325581395353
  },
  "C5": {
    "candidate_median": -0.06587548566593054,
    "increase": 0.07204938441912276,
    "reference_median": -0.1379248700850533
  },
  "C6_stage_worsening": [
    -0.053319617478573084,
    0.1900009499703864,
    -0.421816014515849
  ]
}
```

R3 class-balanced observed values (nulls retained):
```json
{
  "C1": {
    "candidate_macro_AURC": 0.2726041752962498,
    "matched_precision_delta": null,
    "reference_macro_AURC": 0.34088502366165246,
    "relative_AURC_reduction": 0.20030462949635253
  },
  "C2_improving_units": 12,
  "C3_failed_units": 17,
  "C4": {
    "candidate_negative_count": 48,
    "candidate_undefined": 0,
    "reference_negative_count": 43,
    "reference_undefined": 0,
    "relative_reduction": -0.11627906976744182
  },
  "C5": {
    "candidate_median": -0.20902415642947034,
    "increase": -0.07109928634441703,
    "reference_median": -0.1379248700850533
  },
  "C6_stage_worsening": [
    0.0701632123576577,
    0.21764366723160722,
    -0.2602528694518893
  ]
}
```

R2 pixel-normalized independent future-candidate admission: `false`. This does not rescue or alter the R3 Gate1C verdict.

The prototype-reliability line is stopped: neither R3 route nor pixel-normalized R2 qualifies.

Coverage is complete: 495 freshly generated validation caches and nine guards; one new three-pair/four-phase integration and 12 guards; nine formal reliability evaluator units, 72 draw0 pairs, 576 teacher-draw records, 72 posterior-mean controls, complete offline PoE controls and 288 formal probe guards. All three completed validation/integration/formal scope parent exit codes are 0; the preserved zero-forward admission failure is recorded separately below. The registered **990+75+1800=2865 new Gate1C forwards** completed. B0 training, engineering tests and 93 batched K2 feature forwards are separate from this diagnostic count.

All three regenerated B0 seeds completed 100 epochs per domain in the fixed domain order (5,295 steps/seed). New three-seed foreground Dice mean: 0.617832663222; old public B0 mean: 0.617832663222; difference: 0.000000 pp. This is descriptive, with no tuning or checkpoint selection from the comparison. Direct historically captured PAS banks are stored in all nine stage-best checkpoints; none was reconstructed.

B0 retained the original deterministic evaluation-matrix policy: test labels were evaluator-only, never training or selection inputs. Gate1C itself used no hidden unlabeled GT or test GT; labeled GT was only the fixed current-domain supervised reference, and val GT only the evaluator.

K2 replication passed: 18/18 foreground units improved, median nine-unit R95 reduction 14.4944%, occupancy-condition pass fraction 100.0%, bootstrap matched cosine median 0.99875748, and 3/3 domains improved. This was K1/K2 replication on new B0-EMA, not a new K search.

The complete private bundle contains **14,470 files / 17,712,127,650 logical bytes**; content SHA-256 `8a82c7b8f0c72eb4faf619f51d7c1eae67a5f81059bc7f283b6b8df22d563526`, manifest SHA-256 `480b627e0f63839ff5430d980020ca026c45838cf5eeb345f2b4cf7c4d578bb2`. It physically contains checkpoints, direct PAS/K2 banks, all raw cache tensors, integration/formal arrays, runtime tests and process receipts. Hardlinks avoid copying verified payloads twice; the complete remote and local bundles were nevertheless SHA-verified in full. Local promotion was atomic and no remote copy was deleted. Original absolute paths in metadata remain untouched; the complete index supplies the relocation mapping.

Exact commands, actual parent/worker exit receipts, per-phase manifests, JUnit, raw logs, all R0/R1/R2/R3/PoE/teacher-noise tables and undefined rows are retained. All scientific comparisons use the original thresholds and unrounded values. R0/R2/PoE/posterior controls cannot rescue an R3 failure. A CURRENT_ONLY future candidate, if present, is a separate R2 decision and does not reverse the R3 verdict.

The unchanged compiler still reads a compatibility filename, GATE1C_V2_INPUT_AUDIT.json, inside each new v3 run. Its content and metadata were freshly generated and hash-bound to v3; that filename is not an old private input or reused historical cache. The diagnostic source commit is distinct from the later documentation/publication commits.

Failure history is preserved. The first v3 administrative cache launch exited 1 before models/workers/forwards because the preregistration digest preceded sorted-key JSON serialization. Its 0-forward output was archived without reuse. An append-only prospective serialization erratum and separate authorization corrected that representation digest, with unchanged 72-pair values/seeds/checkpoints/formulas/budget and a new directory. The original documents remain unchanged; the corrected code passed 248 server numerical tests and 89 real-model/engineering tests, with the one disclosed historical-root duplicate skip. No second integration was run.

Earlier tooling/runtime failures and warnings remain in the versioned evidence directories: initial access/transport issues, the historical-root test and temporary-alias archive issue, and local test assertion/metadata-assembly errors. Local NumPy/BLAS and frozen scheduler/CUDA warnings are preserved, not suppressed; final exact-server checks and immutable-model/numeric barriers passed. Structurally unavailable history/null scores and zero-gradient cosines remain explicit; they are not nonfinite evidence silently dropped from admission.

Old v2.2 full remains UNKNOWN_AFTER_SSH_OBSERVATION_INTERRUPTION, its private references PERMANENTLY_UNAVAILABLE, and its C1-C8 conclusion null. No SSH exit code was used as experiment completion evidence; no old full budget was counted as completed work and no old attempt/private input was retried.

Execution used the existing zmic44 runtime and only authorized physical GPUs4/5/6/7, without stopping/reserving existing jobs or installing packages. Exact diagnostic code: `db4af88eca0dca48025f8884bf7f85e068eabf2a`; operative preregistration: `4a21799bc50e4e6c644a89b08808e59de525d6a9`; separate authorization: `9c673bd745a0d35a6c25f772c7aca58502d0e572`.

**Hard stop for independent review after publication.** No C0 regeneration, reduced-method implementation, method training, learned transport, Gate2, Prostate, MnMS, full sweep or main merge. Any eligible future candidate requires a separate preregistered C0 seeds0/1/2 phase before performance comparison. This is not DI-DMPA reproduced or PASS_CORE_ADMISSION.

The separate publication receipt binds the report commit and verified public branch SHA without a self-referential report hash.

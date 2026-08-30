# Gate 1C v2 offline diagnostic

Frozen B0-EMA K=2 identity-history reliability only. This package does not register a method, construct an optimizer, update a model/bank, or use T1/T2 outputs. Overall Gate1 remains `FAIL_TRANSPORT_NOT_SUPPORTED`.

Reuse: original complete UNet loader and PAS, Gate1A null-aware support and model-state hashing, original immutable pair/seed protocol. New code is isolated here; no Gate0/A/B source or artifact is edited.

Publication order: B closure → C preregistration → separate authorization → synthetic tests → exact-code commit/push/remote verification → real integration → nine validation caches/audit/evaluator → all72 draw0 probes → all576 teacher draws → posterior mean → PoE → unified report/receipt → stop.

```sh
PYTHONPATH=. python -m pytest -q tests/di_dmpa_gate1c_v2/test_core.py tests/di_dmpa_gate1c_v2/test_pipeline.py
```

Real integration requires `GATE1C_V2_CODE_COMMIT` and create-only `GATE1C_V2_INTEGRATION_OUTPUT`. The runner requires the matching successful test receipt and a clean, remotely published exact-code checkout. All raw caches are hash-bound and remain on the execution host; public tables include unsupported/null reasons. No automatic retries, threshold changes, training, Gate2, theory final, other benchmarks or main merge.

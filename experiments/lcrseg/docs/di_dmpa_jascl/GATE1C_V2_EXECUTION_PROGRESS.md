# Gate 1C v2 execution progress

Last observed: 2026-08-30 14:55:56 UTC. This is a dated running-status snapshot,
not a scientific verdict. Consult the remote completion/failure artifacts for
current state; never infer success from an old PID or this file alone.

## Long-running scope

The user clarified that only the current JASCL / DI-DMPA medical segmentation
method is intended. B0/C0 remain controls. The native long-running Goal and
same-thread 15-minute follow-up `ssl-cl-seg` are active. Results will be audited
and analyzed, then a prospective next plan will be recorded and executed.
Original, repaired and subsequent variants remain distinguishable.

Workflow authorization / scope clarification:
`b6d70699599bd89faafb4a1dd22223575c62bbb6`.
The original Gate 1B failure is unchanged; no transport result is rescued.

## Verified test milestone

- Exact repaired diagnostic code:
  `01b3fd0b6cc7648261e5cae03e84f2cef60c363a`.
- Pre-publication synthetic checks: **98 passed**.
- Post-publication complete suite: **99 passed, 0 failed, 0 skipped**, 20.12 s.
- Real integration: original three stage cases, the frozen known-null case,
  exact R1/PAS parity and one original full gradient pair passed.
- Five model immutability guards passed; all nine B0 checkpoint hashes unchanged.
- Strict deterministic CE checks passed on both GPUs; repeated loss/gradient
  bitwise equal, largest independent-reference loss error 1.1920928955078125e-7,
  largest parameter-gradient error 1.4901161193847656e-8.
- Hidden/test GT use: none. Model/transport optimizer updates: zero.

Public evidence (metadata and passing logs only; no feature or label tensors):

- [Unit/integration receipt](gate1c_v2_test_results/01b3fd0/GATE1C_V2_UNIT_INTEGRATION_TEST_REPORT.json)
- [Real integration details](gate1c_v2_test_results/01b3fd0/GATE1C_V2_REAL_INTEGRATION.json)
- [JUnit](gate1c_v2_test_results/01b3fd0/pytest.xml)
- [Passing test log](gate1c_v2_test_results/01b3fd0/pytest_output.txt)

Prior code `68dedea7ccaa9144913dfc50a096364d7d55f2cf` attempt1 (network) and
attempt2 (fused CUDA mean-CE determinism) remain intact. Their evidence hashes
are bound in the passing receipt and the deterministic-CE repair ledger.

## Formal run started, not yet complete

Started 2026-08-30 14:53:58 UTC. Wrapper PID 87258; runner PID 87259;
validation workers observed at PIDs 87280 and 87281 on physical GPUs 0 and 1.
Both workers advanced through validation cases. The first two completed units
were seed0/stage0 (100 cases) and seed0/stage1 (40 cases). Partial progress is
not used for a gate decision.

- Clean execution checkout: `/root/SSL_CL_gate1c_v2_ce_repaired`.
- Run root: `/root/LCRSeg/runs/di_dmpa_gate1c_v2/32d32ab5e491f2e14c3edde6b4f319f978217351/gate1c_v2_01b3fd0b6cc7648261e5cae03e84f2cef60c363a_attempt1`.
- Process receipt: `/root/gate1c_v2_formal_01b3fd0_process.json`.
- Parent log: `/root/gate1c_v2_formal_01b3fd0.log`.
- Exit receipt (expected after exit): `/root/gate1c_v2_formal_01b3fd0_exit.json`.
- Test root: `/root/LCRSeg/runs/di_dmpa_gate1c_v2_validation/01b3fd0b6cc7648261e5cae03e84f2cef60c363a/attempt1`.

The detached launcher was `/root/.venvs/lcrseg-py310/bin/python
/root/gate1c_v2_run_formal_01b3fd0.py --launch`; launcher SHA256
`8cc5c1822957799eff26386083c66c652b716dacba5760f4a82f080b4d8a0436`.
Do not relaunch it or write into the occupied attempt.

The runner enforces validation cache/audits before metrics, all72 draw0 pairs
before 8-draw noise, then posterior-mean and PoE controls, followed by the
complete C1-C8 compilation. Inspect shard logs, phase receipts, failure files,
`EXECUTION_COMPLETION.json` and the artifact manifest before reporting a verdict.
No optimizer training, new method registration or main merge has occurred.

Progress reports are published only on `codex/sslcl-long-running-reproduction`;
the code-validation branch remains at `01b3fd0` while this execution is active.

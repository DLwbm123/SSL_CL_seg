# Gate 1C v2 execution progress

Last observed: 2026-08-30 15:08:51 UTC. The formal run has **exited with code 1**:
one frozen checkpoint lacks the legacy PAS prototype required by R1. This is
incomplete input evidence, not a scientific PASS or FAIL of reliability.

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

## Preserved launch observation

Started 2026-08-30 14:53:58 UTC. Wrapper PID 87258; runner PID 87259;
validation workers observed at PIDs 87280 and 87281 on physical GPUs 0 and 1.
Both workers advanced through validation cases. The first two completed units
were seed0/stage0 (100 cases) and seed0/stage1 (40 cases). Partial progress is
not used for a gate decision.

- Clean execution checkout: `/root/SSL_CL_gate1c_v2_ce_repaired`.
- Run root: `/root/LCRSeg/runs/di_dmpa_gate1c_v2/32d32ab5e491f2e14c3edde6b4f319f978217351/gate1c_v2_01b3fd0b6cc7648261e5cae03e84f2cef60c363a_attempt1`.
- Process receipt: `/root/gate1c_v2_formal_01b3fd0_process.json`.
- Parent log: `/root/gate1c_v2_formal_01b3fd0.log`.
- Exit receipt: `/root/gate1c_v2_formal_01b3fd0_exit.json`.
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

## Formal failure and source audit

The exit receipt records 2026-08-30T14:56:30.638492+00:00, exit code 1.
The wrapper, runner and both workers have exited; both GPUs are idle. Validation
cached 265 cases (seed0: 100/40/25; seed1 stage0: 100). No partial-cache admission
decision is permitted. No gradient phase or optimizer training was launched.

The first shard failed in `di_dmpa_gate1c_v2/binding.py::load_b0` with
`ProtocolError: missing frozen legacy PAS prototypes`; the second shard stopped
after the peer failure. Preserved formal artifact hashes:

- `GATE1C_V2_FAILURE.json`:
  `2b18fa984baeae99fb61afee480e0d58fab8db3c28062d1688ff49e51a241d02`.
- `GATE1C_V2_ARTIFACT_MANIFEST.json`:
  `5f6e63fb4193b417c298f9b70abcbecb8983b43c70390a2d5d38cb4169a4acfd`.

[The nine-checkpoint payload audit](GATE1C_V2_LEGACY_PROTOTYPE_AVAILABILITY_AUDIT.json)
finds eight finite `(3,16)` tensors and exactly one missing bank:
`B0/seed1/stage1`, checkpoint SHA256
`fd61eb6c8c6b1b4e13ce16f9b442572f7c951e03b9403925ad5d898011201b11`.
It was saved after epoch 0 (`epoch=1`, global step 3208), whereas the frozen
runner creates the bank only after the supervised phase of zero-based epoch 25.
`prototypes=None` is real, not a malformed tensor or an overly strict guard.
There is no independently saved stage1 prototype in the original run directory.
All nine checkpoint file hashes are unchanged; the audit used CPU payload reads,
zero model forwards, zero labels and zero prototype refits.

The passing integration checked model hashes for all nine files but real payload
semantics only for its preregistered cases. Future readiness must check required
payload fields for every checkpoint before expensive cache generation.

The original R1 explicitly requires `checkpoint['prototypes']`. Do not fill zeros,
borrow another seed/domain/C0/K2 bank, replace a frozen checkpoint or remove the
guard. A separate prospective bounded recovery feasibility plan may replay the
original trajectory. Any later consumer of a recovered artifact requires a new
input/protocol version; it cannot complete this frozen v2 attempt or reuse its
partial caches. Gate 1B remains `FAIL_TRANSPORT_NOT_SUPPORTED`.

Progress reports are published only on `codex/sslcl-long-running-reproduction`;
the code-validation branch remains at `01b3fd0`. The same-thread heartbeat now
records this failure and will not relaunch the occupied attempt.

## Prospective recovery preflight

Published recovery feasibility plan:
`05946f05484ab3bf612daf20a21e4fee541668ef`; this is not a Gate1C admission run.
The [plan](GATE1C_LEGACY_PAS_RECOVERY_PREREGISTRATION.md) permits only two
original-code replays, 200 supervised updates each, with unchanged numerical
tolerances and a stop before any unsupervised update or test evaluation.

The helper `scripts/recover_gate1c_legacy_pas.py` imports a clean detached
`fb55e802` checkout instead of duplicating its training loop. Guard self-checks
pass for trace identity/tolerance/nonfinite values, roles/domains, the exact
capture budget and byte-exact replica comparison (including signed zero).
Helper SHA256: `1855c6525f1afb2b73a069c028ed2637deb120e9267d1e7aa7966e4f3daabe14`.
The publication check reads the exact public branch ref from GitHub's HTTPS API,
which the compute node can reach. Its `origin` is an older local Git bundle;
Git smart-HTTP timed out, while the API returned the correct public ref. No
network/proxy or shared-remote configuration was changed. Code is transferred
as a verified incremental bundle, and the API response is recorded per replica.

Original-source regression: **25 passed**, 7.31 s, on CPU with synthetic data;
the resume fixtures used their default TinySegNet and the separate model-contract
tests exercised the real UNet/stochastic classifier. No real B0 checkpoint was
trained in these tests. The selected files were `test_resume_v2.py`,
`test_model_checkpoint.py`, `test_deterministic_supervised_smoke.py`,
`test_pas_probability.py`, `test_classifier_stochasticity.py` and
`test_official_model_contract.py` under `tests/gate0`.
JUnit: `/root/gate1c_pas_recovery_source_tests_attempt1.xml`, SHA256
`472e5c80df8f5a3a580042f95e01225abd9e037e57e7e754d62df7f5ab862b47`.
These are engineering checks, not evidence that the missing historical bank
has been reconstructed. Replay has not started at this preflight checkpoint.

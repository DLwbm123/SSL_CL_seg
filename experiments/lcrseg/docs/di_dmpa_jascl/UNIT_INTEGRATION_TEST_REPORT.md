# Gate 0 v2 unit and integration test report

Source commit: `fb55e8022bc379e2515a46214c6fdf45ea818de6`.
Date: 2026-08-30. Environment: existing Python 3.10.21 / PyTorch 2.2.1+cu121.

Remote result: **52 passed**, 0 failed, 0 skipped, 38 warnings, 80.85 seconds.
The actual JUnit cases and transcript hashes are recorded in
`UNIT_INTEGRATION_TEST_REPORT.json`, `pytest.xml`, and `pytest_output.txt`.
The compiler validates those artifacts rather than inserting a PASS constant.

Covered contracts:

- exact probability squared-L2 formula, with no division by class count;
- detached bool PAS validity and student/teacher intersection;
- invalid-pixel zero contribution and graph-connected zero for empty masks;
- nonzero student unlabeled gradients and total-minus-supervised gradient;
- teacher/prototype/mask isolation; lambda=0 removes only the unlabeled gradient;
- explicit classifier stochasticity, posterior-mean repeatability, and no eval RNG consumption;
- real UNet/JASCL CUDA state restoration at mid-supervised, mid-PAS,
  before-best stage boundary, and after-best stage boundary;
- fail-closed reports rejecting zero gradients, hard-index MSE, detached
  consistency, mismatched provenance, and bare PASS without real evidence;
- byte-for-byte preservation of archived v1 status and matrices;
- existing frozen Fundus adapter and real-data leakage/six-step smoke tests.

The four extended resume trajectories use the production model and runner
on explicitly synthetic, hashed 16x16 HDF5 fixtures. They are state-machine
integration evidence, not formal Fundus performance or full-data resume claims.
Real Fundus unlabeled-gradient evidence is independently reported in
`PAS_GRADIENT_AUDIT.json`.

Local complete regression: **264 passed, 4 skipped**, 22 warnings, 27.48 seconds.
The skips are the remote-data-only tests. A separate local actual-UNet
classifier/resume run passed **8 tests** (CPU, 14.54 seconds).

Expected warnings preserve the frozen scheduler order and PyTorch 2.2.1's
undeclared deterministic CUDA NLL kernel; no schedule or tolerance was changed.
The old supervised deterministic comparison is explicitly a smoke test.
Method off-switch parity is NOT_APPLICABLE_METHOD_NOT_IMPLEMENTED.

# Gate1C v3 exact command evidence

The JSON companion retains literal executed child argv, working directories and explicit environment overrides from server-local launch receipts. These are historical command records, not retry instructions. Completed output directories must not be reused.

| Phase | Actual child exit | Literal command receipt |
| --- | --- | --- |
| B0_seed0 | 0 | [gate1c_v3_results/baseline_completed/seed0/LAUNCH_REQUEST.json](gate1c_v3_results/baseline_completed/seed0/LAUNCH_REQUEST.json) |
| B0_seed1 | 0 | [gate1c_v3_results/baseline_completed/seed1/LAUNCH_REQUEST.json](gate1c_v3_results/baseline_completed/seed1/LAUNCH_REQUEST.json) |
| B0_seed2 | 0 | [gate1c_v3_results/baseline_completed/seed2/LAUNCH_REQUEST.json](gate1c_v3_results/baseline_completed/seed2/LAUNCH_REQUEST.json) |
| baseline_authorization_bedf54f | 1 | [gate1c_v3_results/baseline_launch_transport/baseline_authorization_bedf54f/LAUNCH_REQUEST.json](gate1c_v3_results/baseline_launch_transport/baseline_authorization_bedf54f/LAUNCH_REQUEST.json) |
| baseline_bundle_bedf54f | 0 | [gate1c_v3_results/baseline_launch_transport/baseline_bundle_bedf54f/LAUNCH_REQUEST.json](gate1c_v3_results/baseline_launch_transport/baseline_bundle_bedf54f/LAUNCH_REQUEST.json) |
| complete_v3_evidence_bundle | 0 | [gate1c_v3_results/complete_v3_archive/LAUNCH_REQUEST.json](gate1c_v3_results/complete_v3_archive/LAUNCH_REQUEST.json) |
| gate1c_v3_formal | 0 | [gate1c_v3_results/formal_v3_complete/LAUNCH_REQUEST.json](gate1c_v3_results/formal_v3_complete/LAUNCH_REQUEST.json) |
| gate1c_math_tests_90b9d87 | 0 | [gate1c_v3_results/gate1c_math_90b9d87_pass/LAUNCH_REQUEST.json](gate1c_v3_results/gate1c_math_90b9d87_pass/LAUNCH_REQUEST.json) |
| gate1c_math_tests_db4af88 | 0 | [gate1c_v3_results/gate1c_math_db4af88_pass/LAUNCH_REQUEST.json](gate1c_v3_results/gate1c_math_db4af88_pass/LAUNCH_REQUEST.json) |
| gate1c_v3_integration | 0 | [gate1c_v3_results/integration_v3_pass/LAUNCH_REQUEST.json](gate1c_v3_results/integration_v3_pass/LAUNCH_REQUEST.json) |
| k2_geometry_tests_9a33734 | 0 | [gate1c_v3_results/k2_geometry_runtime_pass/LAUNCH_REQUEST.json](gate1c_v3_results/k2_geometry_runtime_pass/LAUNCH_REQUEST.json) |
| K2_replication | 0 | [gate1c_v3_results/k2_replication_pass/LAUNCH_REQUEST.json](gate1c_v3_results/k2_replication_pass/LAUNCH_REQUEST.json) |
| runtime_tests_4e3e274 | 0 | [gate1c_v3_results/runtime_4e3e274_pass/LAUNCH_REQUEST.json](gate1c_v3_results/runtime_4e3e274_pass/LAUNCH_REQUEST.json) |
| runtime_tests_90b4599 | 1 | [gate1c_v3_results/runtime_90b4599_failure/LAUNCH_REQUEST.json](gate1c_v3_results/runtime_90b4599_failure/LAUNCH_REQUEST.json) |
| runtime_tests_90b9d87 | 0 | [gate1c_v3_results/runtime_90b9d87_pass/LAUNCH_REQUEST.json](gate1c_v3_results/runtime_90b9d87_pass/LAUNCH_REQUEST.json) |
| runtime_tests_9a33734 | 0 | [gate1c_v3_results/runtime_9a33734_pass/LAUNCH_REQUEST.json](gate1c_v3_results/runtime_9a33734_pass/LAUNCH_REQUEST.json) |
| runtime_tests_db4af88 | 0 | [gate1c_v3_results/runtime_db4af88_pass/LAUNCH_REQUEST.json](gate1c_v3_results/runtime_db4af88_pass/LAUNCH_REQUEST.json) |
| gate1c_v3_validation | 1 | [gate1c_v3_results/validation_serialization_admission_failure/LAUNCH_REQUEST.json](gate1c_v3_results/validation_serialization_admission_failure/LAUNCH_REQUEST.json) |
| gate1c_v3_validation | 0 | [gate1c_v3_results/validation_v3_pass/LAUNCH_REQUEST.json](gate1c_v3_results/validation_v3_pass/LAUNCH_REQUEST.json) |

Worker commands come from the pinned dispatch implementation and are bound by per-worker start/completion records and server-parent phase exits. Controller launches, including all arguments and executable paths, are retained verbatim above. No command is inferred from an SSH return code.

Local read-only/preflight commands and initial Git publication are preserved in [the original preflight command record](gate1c_v3_results/preflight_20260831/EXACT_COMMANDS.md). Local and server JUnit, failed assertions, raw outputs and successive exact-code runtime reports remain in their original evidence directories. Publication commands and verified public SHAs are in the separate final publication receipt.

The exact production diagnostic source is db4af88eca0dca48025f8884bf7f85e068eabf2a. The original 70ba3dfb4fc989a5149a6343d857e7d10fd2017d files remain unchanged. No old v2.2 attempt or private input was retried.

# Gate 1B v2 pre-publication synthetic validation

Date: 2026-08-30. Development checkout: `/root/SSL_CL_gate1b_v2_dev`, based on authorization commit
`c6f72b86fdfa3683a6e2c7dbf593f73cab74c592`. The existing `/root/.venvs/lcrseg-py310/bin/python` was reused; no dependency was installed.

All inputs to these tests were synthetic tensors/images/checkpoint-byte fixtures or already-frozen JSON contracts.
No new real checkpoint tensor, real model forward, actual transport coordinate plan, real paired census or real transport fit was used.
Synthetic optimizer-step tests are not experiment updates.

| Development run | Result | Retained JUnit artifact |
| --- | --- | --- |
| v1: unit suite | 73 passed, 6.44 s | `/root/gate1b_v2_synthetic_v1.xml` |
| v2: added end-to-end synthetic pipeline | 73 passed, 1 failed, 22.35 s | `/root/gate1b_v2_synthetic_v2.xml` |
| v3: corrected mock target and added boundary tests | 76 passed, 22.70 s | `/root/gate1b_v2_synthetic_v3.xml` |
| v4: final source, balanced GPU assignment and final artifact rechecks | 76 passed, 23.03 s | `/root/gate1b_v2_synthetic_v4.xml` |

The v2 failure was in the test double: patching `di_dmpa_gate1.feature_extraction._images` did not replace the alias
already imported by `di_dmpa_gate1_v2.features`. The synthetic evaluator correctly rejected the nonexistent
`h5/v1/synthetic/image.h5` instead of bypassing its image hash check. The test now patches the actual consumer's alias.
No method, threshold, optimizer, data selection or preregistration was changed to fix this test.

Coverage includes all44 preregistered categories, exact1000-step Adam accounting, no segmentation optimizer,
all four support states, complete zero-AA census rejection, immutable checkpoints and chain inputs,
angular Hungarian matching that differs from maximum-total-cosine matching, raw gate boundaries, file hashes,
and a synthetic cache → twelve-unit census → six fixed fits → chain-oracle pipeline.

Operational setup notes: the fresh development worktree initially lacked its ignored `third_party` directory;
creating that directory allowed the read-only official-reference symlink. The first source tar carried macOS
extended-attribute warnings; subsequent archives use `--no-xattrs`. Neither issue involved diagnostic data.

The next barrier is exact diagnostic code commit/push/`git ls-remote` verification. Real read-only integration
is separately opt-in and must PASS before the full registered coordinate plan or formal attempt can start.

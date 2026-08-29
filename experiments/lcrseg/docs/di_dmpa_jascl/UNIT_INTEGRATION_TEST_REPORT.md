# Gate 0 unit and integration test report

Date: 2026-08-29
Environment: `/root/.venvs/lcrseg-py310`, Python 3.10.21, PyTorch
2.2.1+cu121
Command:

```bash
cd /root/JASCL_gate0_sync
LD_LIBRARY_PATH=/lib/x86_64-linux-gnu PYTHONPATH=. CUDA_VISIBLE_DEVICES= \
  /root/.venvs/lcrseg-py310/bin/python -m pytest -q tests/gate0
```

Result: `17 passed in 3.62s`.

The earlier DeepLab test result is invalidated and excluded. This result
exercises the corrected LCRSeg UNet2D plus official JASCL 3x3 stochastic-head
contract.

Covered contracts:

- fixed independent benchmark/class protocol and all method switches off;
- frozen LCRSeg UNet2D body, official commit subtree unchanged, and official
  classifier kernel 3x3;
- current-domain-only manifest views and hidden-GT isolation;
- val/test rejection by the training adapter;
- unlabeled batch schema without label tensors;
- teacher frozen, optimizer exclusion, and no-grad forward;
- complete classifier required at stage load;
- full checkpoint and Python/NumPy/CPU/CUDA RNG round trip;
- repaired optimizer step after the unlabeled backward;
- off-switch one-step parity;
- uninterrupted versus interrupted/resumed state trajectory.
- report audit accepts intentionally absent LR on unlabeled rows while still
  rejecting a missing or non-finite supervised LR.

The remaining scheduler warnings are expected evidence of the preserved
upstream call order (`scheduler.step(epoch)` before the first optimizer step of
an epoch); this Gate 0 repair does not change that schedule.

The previous DeepLab GPU smoke checks are excluded. Corrected UNet GPU checks
passed for:

- 1-step supervised forward/backward/GAS/optimizer/checkpoint;
- actual-model six-step uninterrupted versus interrupted/resumed equivalence;
- current-domain PAS prototype shape `[3,16]`, unit norms, finite teacher
  tensors, and unlabeled batch without a label key.

## SSL_CL_seg merge validation

After integration into `experiments/lcrseg`, the existing local environment
(`/opt/miniconda3/bin/python`, PyTorch 2.6.0) produced:

- Gate 0 subset: `13 passed, 4 skipped`;
- complete `experiments/lcrseg/tests` suite: `229 passed, 4 skipped`.

The four skips are the frozen-data tests guarded by the absence of
`/root/LCRSeg` on the local macOS host. They are the same tests included in the
remote formal result above, where all 17 Gate 0 tests passed.

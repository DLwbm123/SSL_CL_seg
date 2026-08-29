# LCR-Seg repository integration audit

Generated: 2026-08-19

## Scope and decision

This audit covers the current local workspace and the `jiangsuiyang` training
host before any LCR-Seg training-engine implementation. The frozen data bundle
is treated as read-only input. The correct LCR-Seg method specification was
found in `Downloads/METHOD_SPEC_V0_1 (1).md`; its SHA-256-matched copy now
lives at `experiments/lcrseg/METHOD_SPEC_V0_1.md`. The similarly named
`Downloads/METHOD_SPEC_V0_1.md` is a MedTRACE/LLM-editing document and is not
used by this project.

## Workspace and version control

- Current workspace: `/Users/bominwang/Desktop/codes/SSL_CL_seg`.
- Git status: this workspace is not a Git repository; therefore no branch,
  commit, dirty-file list, or commit boundary can currently be reported.
- A separate remote source-deployment root now exists at
  `/home/jiangsuiyang/SSL_CL/code/SSL_CL_seg`; it is empty until source-only
  synchronization occurs.
- Consequence: implementation can proceed in the current real workspace, but
  the requested per-milestone commits are blocked until the user supplies or
  authorizes an actual repository. No repository is initialized or pushed by
  this audit.

## Frozen data and runtime facts

- Server data root: `/home/jiangsuiyang/SSL_CL`.
- Frozen data inventory: 1,466 image/label records, 2,932 HDF5 files, and
  2,962 SHA-256 manifest entries; prior remote integrity/readability gates
  passed.
- Formal runtime candidate:
  `/home/jiangsuiyang/anaconda3/envs/py38/bin/python`.
- Candidate runtime packages: Python 3.8 environment, PyTorch `2.2.1+cu121`,
  CUDA `12.1`, cuDNN `8902`, `h5py 3.16.0`, NumPy `1.26.4`, Pillow `10.3.0`,
  and pytest `8.3.5`; CUDA is available.
- GPU inventory: seven NVIDIA GeForce RTX 3090 GPUs, each with 24 GiB VRAM.
- The frozen HDF5/manifests/splits/checksums are now owner-readable but
  non-writable (`dr-x------`) after the M0 storage lock. Training outputs must
  use the separate `/home/jiangsuiyang/SSL_CL/runs` directory.
- `/home` has approximately 67 GiB free, so run/checkpoint retention must be
  bounded and raw data must not be duplicated.

## Reusable components

| Component | Status | Audit finding |
| --- | --- | --- |
| HDF5 payloads | reusable | Versioned `h5/v1`, explicit image/label paths, provenance and checksums. |
| Training manifests | reusable | Seeded manifests provide image-only `train_unlabeled` rows with an empty label path. |
| Basic H5 reader | partial | `lcrseg/h5_dataset.py` opens image/label files safely, but it is not the formal labeled/unlabeled batch interface. |
| Preprocessing smoke | replace | `scripts/two_case_overfit.py` uses a tiny ad hoc segmenter and writes under the data root; it cannot serve as the formal M0 overfit gate. |
| U-Net | absent | No LCR-Seg 2D U-Net exists in this workspace. |
| Transforms | absent | No shared weak/strong geometry contract or cutout-valid-mask implementation exists. |
| Trainer / continual runner | absent | No common runner or method lifecycle implementation exists. |
| Evaluator | absent | No Dice/ASD/HD95 per-case or site-matrix evaluator exists. |
| Checkpoint manager | absent | No complete method/anchor/RNG checkpoint schema exists. |

## Data-contract observations

- Training manifests omit `label_h5_relpath` for `train_unlabeled`; diagnostic
  manifests retain labels and are not a training input.
- The existing `H5ManifestDataset` returns a generic dictionary containing a
  `label` key set to `None` for unlabeled rows. The new formal training data
  layer must instead construct `UnlabeledBatch` without any label field and
  must reject hidden-label paths.
- M&Ms uses ED/ES phase records with a patient identifier, so split validation
  must enforce patient-level grouping.

## Required integration work

1. Add the contracts, data package, 2D U-Net, projection head, shared engine,
   methods, evaluator, checkpoint manager, configs, tests, and analysis CLI
   under `experiments/lcrseg/`.
2. Synchronize only code, specifications, configs, tests, and implementation
   reports to the existing server code root. Checkpoints, caches, logs, and
   report artifacts must never be written under `h5/v1` or frozen
   manifests/splits/checksums.
3. Record the frozen manifest and split hashes in every resolved training
   configuration.
4. Replace the ad hoc two-case script with the formal model/loader/checkpoint
   path before treating it as an M0 gate.

## Planned files

```text
lcrseg/contracts.py
lcrseg/data/{h5_dataset.py,batch_types.py,transforms.py,continual_sampler.py}
lcrseg/models/{outputs.py,projection_head.py,unet.py}
lcrseg/methods/{base.py,supervised.py,sequential_ssl.py,uniform_kd.py,lcrseg_v0_1.py}
lcrseg/methods/components/{anchor_bank.py,relation_field.py,pseudo_label.py,learnability.py,compatibility.py,routing.py}
lcrseg/engine/{trainer.py,continual_runner.py,evaluator.py,checkpoint.py}
lcrseg/analysis/
configs/{model,method,experiment}/
tests/
scripts/{run_lcrseg.py,two_case_overfit.py}
```

## Acceptance and run commands to establish

```bash
cd /home/jiangsuiyang/SSL_CL/code/SSL_CL_seg/experiments/lcrseg
export LCRSEG_DATA_ROOT=/home/jiangsuiyang/SSL_CL
export LCRSEG_RUN_ROOT=/home/jiangsuiyang/SSL_CL/runs
export LCRSEG_PYTHON=/home/jiangsuiyang/anaconda3/envs/py38/bin/python

$LCRSEG_PYTHON -m pytest -q tests
$LCRSEG_PYTHON scripts/two_case_overfit.py --root "$LCRSEG_DATA_ROOT" --seed 0 --dataset fundus --steps 200
```

## M0 blockers and next action

- **Blocking for commits only:** no Git repository exists in the supplied
  workspace; do not fabricate a repository, commit, or remote.
- **Blocking for M0 acceptance:** the new formal code must still be deployed
  and run against the server's locked inputs; no M0 result has been claimed.
- **Next action:** deploy source only to the explicit server code root and run
  M0 against the existing `py38` environment.

## Implementation update (2026-08-19)

The initial audit above is preserved as the pre-implementation snapshot. The
following update records the actual state after M0 through M5 engineering work.

- Source-only code is deployed to
  `/home/jiangsuiyang/SSL_CL/code/SSL_CL_seg/experiments/lcrseg`; no frozen
  data path has been modified.
- The formal runtime is `/home/jiangsuiyang/anaconda3/envs/py38/bin/python`
  (Python 3.10.6, PyTorch 2.2.1+cu121, CUDA 12.1, cuDNN 8902).
- The reusable runtime now includes `UNet2D`, typed HDF5 labeled/unlabeled
  datasets, shared transforms, deterministic sampler, trainer, evaluator,
  full-state checkpoint manager, and one continual runner for all methods.
- M0 passed using the formal loader on workers 0 and 4 and the Fundus 2-case
  overfit gate. The data root remained read-only.
- M1 baseline tiny smokes, a real remote interruption/resume equality check,
  M2 relation overfit, M3/M4 routing tests, and M5 golden batch all passed.
- The local and remote test suites each pass 23 tests. The current runner
  serializes expanded method defaults before provenance/checkpoint creation.
- The first strict-warm-up Fundus 5-epoch LCR pilot is complete; it validates
  engineering behavior only. Its metrics and post-hoc hidden-GT diagnostic
  conclusion are recorded here. The engineering-only raw run directory was
  pruned during the 2026-08-20 project storage cleanup.
- **Current blocker:** this worktree is not a Git repository, so the requested
  milestone commits cannot be created. No branch was invented and no push was
  attempted.
- **Current next action:** run the explicit Fundus seed-0 200-epoch formal
  configuration, then check its engineering/reliability gates before the
  Prostate `RUNMC -> BMC` pilot.

## Final Fundus update (2026-08-20)

- The implementation and Fundus seed-0 suite completed with 25 local and 25
  remote tests passing. The full detail and eligible run table are in
  `BASELINE_AND_V0_V3_COMPLETION.md`.
- Two fairness defects were found and fixed before final comparison: Joint-SSL
  now sums per-site rounded schedules to exactly 13,400 steps, and supervised
  controls match the SSL optimizer-step budget without consuming unlabeled
  batches. The earlier 13,200-step Joint and 6,600-step supervised artifacts
  are excluded from comparison and were pruned during the 2026-08-20 project
  storage cleanup.
- Full LCR completed 13,400 steps with no NaN rows, no skipped optimizer step,
  valid anchors, no warnings, and exact golden-batch verification.
- The Fundus research gate is **not met**: full LCR is below both
  Sequential-SSL and Uniform-KD in final average/previous-site performance,
  while compatibility calibration is not strictly monotonic in its upper
  bins. Consequently the conditional Prostate pilot was not launched.
- Frozen input directories remain mode `500`; all experiment writes are under
  `/home/jiangsuiyang/SSL_CL/runs`.

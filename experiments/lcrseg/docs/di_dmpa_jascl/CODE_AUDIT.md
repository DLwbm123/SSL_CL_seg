# DI-DMPA-JASCL pre-implementation audit

Date: 2026-08-29

## Provenance and scope

- Upstream repository: `https://github.com/prinshul/JASCL`
- Audited upstream commit: `3c93ca70784fc3a1d2a887f8d7dce5af6bc75f53`
- Local branch: `codex/di-dmpa-jascl`
- Primary upstream path audited:
  `Semi-Supervised_Natural-FoSSIL/inc/deeplab_gaps_meanT`
- Paper checked: *Continual Segmentation under Joint Nonstationarity*,
  arXiv:2605.20538v1, especially Sections 3.1-3.2, Figure 3, and the
  Semi-Supervised Natural-JASCL benchmark description.

The attached DI-DMPA-JASCL specification is treated as a proposed method
contract. The paper and upstream code are evidence sources; neither is treated
as authorization to weaken the fixed-class, current-domain-only, or hidden-GT
constraints.

## Paper-to-proposal reconciliation

The paper's PAS uses one normalized class prototype per class and accepts a
pseudo-label only when both prediction confidence and cosine similarity exceed
hard thresholds. Figure 3 places PAS inside a mean-teacher loop and applies GAS
to the decoder pixel classifier. The proposed DI-DMPA-JASCL method is not in the
paper: domain-indexed multi-prototypes, feature transport, a continuous soft
product of experts, a history gate, and transported-prototype replay are new
research components.

The paper's Semi-Supervised Natural-JASCL benchmark is also not a pure
fixed-class domain-incremental benchmark: its incremental sessions introduce
new classes. It cannot be used as the main protocol without violating the
fixed-class requirement.

## Upstream code map

| Audit item | Upstream location | Finding |
|---|---|---|
| Student/teacher construction | `train_step2.py:875-884` | Two independently constructed DeepLab models. |
| Mean-teacher update | `fix_seed.py:20-33` | Parameter EMA with alpha 0.99; teacher parameters are not frozen. |
| Previous-stage load | `train_step2.py:901-920` | Loads the previous student state into both models but deliberately omits `conv_logit`, because the upstream protocol expands classes. |
| Checkpoint | `train_step2.py:568-574` | Saves student and optimizer only; teacher, prototype state, scheduler, RNG, and stage metadata are absent. |
| Resume | `train_step2.py:889-900` | References `model` before the mean-teacher wrapper is constructed; the branch is not runnable as written. |
| Augmentation | `train_step2.py:209-245` | One paired image/mask transform for labeled data; unlabeled data uses the non-augmented transform. No weak/strong view pair or transform record exists. |
| PAS | `fix_seed.py:46-111`; `train_step2.py:459-505` | Hard confidence/similarity filtering. The unlabeled loop calls `backward()` but has no `optimizer.step()`. |
| GAS | `models/deeplabv3/deeplab.py:26-72`; `train_step2.py:433-443` | Squared classifier gradients are copied into `grad_update`; inverse-gradient stochastic weights are used on later forwards. |
| Pre-classifier embedding | `models/deeplabv3/deeplab.py:167-174` | Concatenated decoder feature, 304 channels, returned alongside logits. |
| Final classifier | `models/deeplabv3/deeplab.py:162-165` | Base uses a 3x3 Conv2d; incremental sessions use a stochastic 3x3 Conv2d. It is not linear/1x1. |
| Stage dataloaders | `train_step2.py:284-314` | Hard-coded stage-specific pickle lists and global paths. |
| Evaluation | `train_step2.py:639-713` | Concatenates seen-domain test data and reports one aggregate IoU; it does not emit a stage-by-domain lower-triangular matrix. |
| DDP/AMP | target subtree | No DDP initialization/wrapper and no autocast/GradScaler path. |

`train_step3.py` and `train_step4.py` are copied stage-specific scripts rather
than one stage state machine, so fixes made to one script do not automatically
apply to later stages.

## Gate 0 evidence collected

A synthetic model-level sanity check was run on the new compute node with the
upstream code and no source modifications:

- host: `root@162.14.139.38:31192`
- environment: `/root/.venvs/lcrseg-py310`
- PyTorch: `2.2.1+cu121`
- device: one RTX 3090 for this check
- input: `[2, 3, 64, 128]`
- logits: `[2, 13, 64, 128]`
- pre-classifier embedding: `[2, 304, 16, 32]`
- supervised loss: `3.696387529373169`
- classifier gradient norm: `14.321928024291992`
- forward/backward/optimizer step: passed
- EMA parameter update: passed at the tensor-update level
- upstream teacher has trainable parameters: confirmed

Artifacts:

- `/root/JASCL/audit_artifacts/upstream_synthetic_sanity.json`
- `/root/JASCL/audit_artifacts/upstream_synthetic_stage1_checkpoint.pt`
- checkpoint SHA-256:
  `ba224b311b0cbb0397c3ead61403ba0bb82350821bd8d57e6d4191227462168d`
- report SHA-256:
  `f225baf9f4e8c1ae282c44d56dace7fc878d63f7a802338f5536b1f9cc6af52f`

This is a synthetic module sanity check, not a baseline reproduction result.
At the audit point, Gate 0 was incomplete because the released training/resume
paths contained the defects listed above. The later, separately named
`gate0_repaired` UNet runner and its completed Fundus evidence are reported in
`GATE0_STATUS.json`; no paper-number reproduction is claimed from this
synthetic DeepLab check.

The synthetic check above characterizes the released JASCL DeepLab path only.
It does not select DeepLab as the LCRSeg medical benchmark architecture.
For Gate 0, the model contract is the frozen LCRSeg `UNet2D` body
(16/32/64/128 channels with GroupNorm) plus the official pinned JASCL 3x3
`ProbabilisticClassifier` as its final pixel head. The official subtree is not
modified. The upstream `models/unet.py` is not suitable as-is because it
hard-codes three input channels, uses a deterministic 1x1 head, and does not
expose the JASCL GAS state.

## Fixed-class domain-incremental data available on the node

The frozen LCRSeg HDF5 bundle already on the new node matches the requested
access pattern better than the paper's semi-supervised benchmark. Each dataset
can be treated as a separate fixed-class domain-incremental benchmark:

| Dataset | Domain index | Fixed labels | Frozen records |
|---|---|---:|---:|
| Fundus | REFUGE, RIM-ONE-r3, Drishti-GS | 0, 1, 2 | 660 |
| Prostate | six acquisition sites | 0, 1 | 116 |
| MnMS | scanner vendor | 0, 1, 2, 3 | 690 |

For seeds 0, 1, and 2, `/root/LCRSeg/manifests/training/` provides explicit
train-labeled, train-unlabeled, validation, and test roles. The frozen HDF5,
manifest, split, and checksum directories are read-only. Training outputs belong
under `/root/LCRSeg/runs`; diagnostic validation labels must remain isolated
from the training objects.

## Resolved protocol decisions

1. Use a separately named `gate0_repaired` runner with only the authorized
   plumbing repairs.
2. Keep the official JASCL 3x3 classifier; exclude prototype replay from the
   core and keep constant-patch regularization disabled.
3. Use Fundus, Prostate, and MnMS as three independent LCRSeg benchmarks.
4. Use the medical LCRSeg `UNet2D` body rather than the released DeepLab body.

## Recommended implementation boundary

1. Keep all upstream scripts byte-for-byte unchanged.
2. Add a new `di_dmpa_jascl` package and a single config-driven stage runner.
3. Freeze two baselines: `upstream_exact_synthetic` and `gate0_repaired`.
4. Build the LCRSeg manifest adapter with assertions that only the current
   domain's train rows can enter training.
5. Do not register or implement DI-DMPA until Gate 0 passes.
6. Require off-switch parity against `gate0_repaired`, and report separately
   that it is not byte-level behavioral parity with the defective upstream
   script.
7. Stop at every preregistered gate; do not launch a full sweep automatically.

# LCR-Seg baseline and V0--V3 implementation report

**Status:** FUNDUS SEED-0 ENGINEERING COMPLETE; PROSTATE A->B IS GATED OFF.

This report records completed evidence through the Fundus seed-0 suite. It
does **not** claim that the Fundus research gate passed or that the conditional
Prostate pilot was run.

## Scope, inputs, and runtime

- Formal data root: `/home/jiangsuiyang/SSL_CL`
- Code root: `/home/jiangsuiyang/SSL_CL/code/SSL_CL_seg/experiments/lcrseg`
- Run root: `/home/jiangsuiyang/SSL_CL/runs`
- Manifest SHA-256:
  `0622f54f42f05d6ef87f9dc89ee9435cf8da03c6c30cd970db6ea167e00dd8a3`
- Split SHA-256:
  `f250d97aea1f36f21899f5dd40bb6c9a819e7755aee458c8ee27506496b46a88`
- Formal interpreter:
  `/home/jiangsuiyang/anaconda3/envs/py38/bin/python` (Python 3.10.6,
  PyTorch 2.2.1+cu121, CUDA 12.1, cuDNN 8902, RTX 3090).

The frozen `h5/v1`, `manifests`, `splits`, and `checksums` directories still
have mode `500`. No training artifact was written below those directories;
checkpoints, logs, diagnostics, and result tables are under `runs/` only.

The supplied workspace is not a Git worktree. No branch, commit, repository
initialization, or push was fabricated. This remains a version-control
blocker for the requested per-milestone commits, not a training blocker.

## Delivered engine and method scope

The shared engine contains typed HDF5 labeled/unlabeled datasets, shared
weak/strong transforms, deterministic scheduling, a 2D U-Net with relation
features, a single trainer/checkpoint/evaluator/continual-runner path, site
matrices, and post-hoc hidden-GT diagnostics.

Registered methods are `Static-Sup`, `Static-SSL`, `FineTune-Sup`,
`Sequential-SSL`, `Uniform-KD/LwF`, `Joint-Sup`, `Joint-SSL`, standalone
`SS-EWC`, and `lcrseg_v0_1`. LCR-Seg V0.1 implements V0 K=1 anchors, V1
detached current learnability, V2 detached historical compatibility, and V3
continuous assimilation/consolidation. V4 multi-agent, V5 RIC, K>1, replay,
diffusion, VAE, a third teacher, and extra auxiliary losses were not added.

The requested ablations are explicit, provenance-recorded options:

- `use_learnability=false`: unit weighting on valid pseudo-labels only;
- `use_compatibility=false`: unit historical relation-KD weighting;
- `lambda_relation=0`: no historical relation consolidation.

Every new run records fully resolved method defaults in both `config.yaml` and
its checkpoints.

## Engineering gates

| Gate | Result |
| --- | --- |
| Formal DataLoader, workers 0 and 4 | Passed; unlabeled batches expose no label field; M&Ms patient split and `auxiliary25` restrictions passed. |
| Fundus 2-case overfit | Passed: loss `2.576756 -> 0.106565`; mean/min foreground Dice `0.974170/0.965229`; checkpoint reload error `0`. |
| Unit tests | 25 passed locally and 25 passed in the formal remote runtime. |
| Resume | Static-SSL interruption/resume reached exact model equality (`max abs=0`) with equal optimizer/scheduler progress. |
| V0--V3 invariants | Tests cover detached routing weights, old-model no-grad, old-anchor immutability, independent anchor storage, buffer-only anchors, empty-set safety, strict bootstrap, and update-after-successful-optimizer-step. |
| Full-LCR runtime | 13,400 steps, zero NaN rows, zero AMP skipped optimizer steps, valid anchors for every class/site, and zero site warnings. |
| Full-LCR golden batch | Create then independent verify passed with all recorded tensor/loss errors exactly `0`. |

The corrected Joint-SSL run had one AMP-scaled skipped optimizer update but no
NaN rows and completed its 13,400-step budget. That event is retained in its
log rather than hidden.

## Budget corrections and artifact eligibility

Two budget defects were found before final comparison and corrected with
tests. Existing artifacts are retained, but excluded below where noted.

1. A joint merged epoch has batch-rounding different from the sum of three
   independent site schedules. The final Joint-SSL budget now explicitly sums
   per-site schedules: `8,000 + 3,200 + 2,200 = 13,400` steps.
2. Static-Sup and FineTune-Sup now use image-only unlabeled *counts* only to
   match the SSL optimizer-step budget; their training loop still receives
   only visible labeled batches.

Excluded, non-comparative artifacts:

- `fundus_seed0_joint_ssl_full200e_budgetfixed` (13,200 steps, not 13,400);
- `fundus_seed0_static_sup_full200e` and
  `fundus_seed0_finetune_sup_full200e` (6,600 steps each).

These entries are retained in this report for provenance only; their raw run
directories were pruned during the 2026-08-20 project storage cleanup.

## Valid Fundus seed-0 results (validation only)

All rows below use frozen seed-0 splits and evaluation role `val`. Values are
`final average Dice`, `BWT`, `incoming Dice`, and `previous-site Dice`.

| Method | Steps | Final avg | BWT | Incoming | Previous | Run directory |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Static-Sup | 13,400 | 0.5914 | -0.2344 | 0.7476 | 0.5831 | `fundus_seed0_static_sup_full200e_budget13400` |
| Static-SSL | 13,400 | 0.5635 | -0.2628 | 0.7387 | 0.5465 | `fundus_seed0_static_ssl_full200e` |
| FineTune-Sup | 13,400 | 0.6344 | -0.1766 | 0.7521 | 0.6172 | `fundus_seed0_finetune_sup_full200e_budget13400` |
| Sequential-SSL | 13,400 | 0.6473 | -0.1557 | 0.7511 | 0.6552 | `fundus_seed0_sequential_ssl_full200e` |
| Uniform-KD/LwF | 13,400 | 0.6540 | -0.1372 | 0.7455 | 0.6747 | `fundus_seed0_uniform_kd_full200e` |
| SS-EWC | 13,400 | 0.6299 | -0.1770 | 0.7479 | 0.6202 | `fundus_seed0_ss_ewc_full200e` |
| Joint-SSL (upper bound) | 13,400 | 0.7066 | n/a | n/a | n/a | `fundus_seed0_joint_ssl_full200e_budget13400` |
| LCR-Seg full V0.1 | 13,400 | 0.6211 | -0.1596 | 0.7275 | 0.6369 | `fundus_seed0_lcrseg_v0_1_full200e` |
| LCR without learnability | 13,400 | 0.6379 | -0.0965 | 0.7022 | 0.6764 | `fundus_seed0_lcrseg_no_learnability_full200e` |
| LCR with uniform relation KD | 13,400 | 0.6551 | -0.1185 | 0.7341 | 0.6759 | `fundus_seed0_lcrseg_uniform_relation_kd_full200e` |
| LCR without relation consolidation | 13,400 | 0.5897 | -0.2193 | 0.7360 | 0.5963 | `fundus_seed0_lcrseg_no_relation_consolidation_full200e` |

The full LCR Dice matrix is:

| Trained through | REFUGE | RIM-ONE-r3 | Drishti-GS |
| --- | ---: | ---: | ---: |
| REFUGE | 0.8276 | 0.5994 | 0.4735 |
| RIM-ONE-r3 | 0.6952 | 0.7072 | 0.3852 |
| Drishti-GS | 0.6642 | 0.5513 | 0.6477 |

## Full-LCR reliability and ablation diagnosis

The separate post-hoc analysis at
`fundus_seed0_lcrseg_v0_1_full200e/posthoc_reliability/` reads diagnostic
labels only after the final checkpoint was frozen. It reports 41 Drishti-GS
records, 377,856 relation-grid pixels, and 98.47% valid pseudo-label coverage.
No diagnostic label was visible to any training loader.

- Learnability calibration passes its directional check: pseudo-label accuracy
  rises monotonically from `0.6748` to `0.9937` across its ten bins.
- Compatibility has a strong broad association with old-model relation
  correctness (`0.2851`, `0.7889`, `0.9563`, ..., `0.9827`), but is **not
  strictly monotonic** in the upper tail (`0.9827 -> 0.9820 -> 0.9802 ->
  0.9739 -> 0.9356`).
- Eliminating relation consolidation is harmful (final `0.5897`, BWT
  `-0.2193`), so the historical-relation loss has signal.
- Replacing compatibility weights with uniform relation KD improves both final
  average and BWT over full LCR (`0.6551/-0.1185` vs `0.6211/-0.1596`).
- Removing learnability improves BWT (`-0.0965`) relative to full LCR but
  reduces incoming Dice (`0.7022`), indicating the current L/C routing
  calibration—not the presence of relation consolidation alone—is the primary
  issue to investigate.

## Fundus research-gate decision

| Required Fundus condition | Decision | Evidence |
| --- | --- | --- |
| Learnability increases with pseudo-label correctness | Pass | `0.6748 -> 0.9937` across bins. |
| Compatibility is monotonically calibrated to old-model correctness | Fail | Upper-tail decline; not strictly monotonic. |
| Compatibility routing improves/reduces negative transfer over Uniform-KD | Fail | Uniform-KD and uniform-relation LCR both outperform full LCR. |
| Full LCR improves previous-site performance over Sequential-SSL without unacceptable incoming loss | Fail | Sequential previous/incoming: `0.6552/0.7511`; full LCR: `0.6369/0.7275`. |
| Analysis explains the observed behavior | Pass | Golden, L/C bins, branch counts, JS, quadrants, and ablations are saved. |

**Gate decision:** `FUNDUS_RESEARCH_GATE_NOT_MET`. Per the supplied plan,
`RUNMC -> BMC` is not started. V4/V5, RIC, multi-agent, and additional
unregistered mechanisms remain out of scope. The next research action needs a
separate preregistered calibration/routing hypothesis; it must not be an
unannounced method expansion.

## Reproduction commands

```bash
cd /home/jiangsuiyang/SSL_CL/code/SSL_CL_seg/experiments/lcrseg
export LCRSEG_DATA_ROOT=/home/jiangsuiyang/SSL_CL
export LCRSEG_RUN_ROOT=/home/jiangsuiyang/SSL_CL/runs

CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. \
  /home/jiangsuiyang/anaconda3/envs/py38/bin/python scripts/golden_batch.py \
  --checkpoint "$LCRSEG_RUN_ROOT/fundus_seed0_lcrseg_v0_1_full200e/checkpoint_final.pt" \
  --dataset fundus --site RIM_ONE_r3 --seed 0 --device cuda --verify --atol 1e-4

CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. \
  /home/jiangsuiyang/anaconda3/envs/py38/bin/python scripts/analyze_reliability.py \
  --checkpoint "$LCRSEG_RUN_ROOT/fundus_seed0_lcrseg_v0_1_full200e/checkpoint_final.pt" \
  --dataset fundus --site Drishti_GS --seed 0 --device cuda \
  --output-dir "$LCRSEG_RUN_ROOT/fundus_seed0_lcrseg_v0_1_full200e/posthoc_reliability"
```

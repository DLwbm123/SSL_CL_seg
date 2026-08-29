# LCR-Seg status

## Current phase

Remote data deployment, the unified continual runner, baseline engine, and
LCR-Seg V0.1 (V0--V3) are implemented and engineering-validated. The Fundus
seed-0 formal suite is complete. Its V0.1 research gate did not pass, so the
conditional Prostate A->B pilot is not started. The separately preregistered
V0.2 Phase-A routing diagnostic and V0.2 implementation/test preflight are
complete. The literal preregistered Fundus V0.2 R0 completed, but triggered
the prompt-required shared-path stop against its named legacy reference; no
V0.2 downstream formal run is currently authorized.

## Completed

- All three source-data roots were read without modification; a final source
  provenance rehash matched all 1,121 raw case pairs against 1,466 immutable
  HDF5 image provenance attributes.
- Actual HDF5 preprocessing is complete at `/Volumes/DataP/LCRSeg/h5/v1`:
  1,466 image/label pairs (2,932 HDF5 files).
- Fundus: 660 accepted pairs, labels `{0,1,2}`, 100% minimum crop retention.
- Prostate: 116 accepted binary pairs, label `2 -> 1`, and all 31 geometry
  mismatches accepted by `index_geometry_repair` under `auto_rule_v1`.
- M&Ms: 345 patients / 690 ED-ES phase pairs; canonical320=320 patients and
  auxiliary25=25 images-only training patients; frozen FOV=320 mm and minimum
  foreground retention=1.0.
- Runtime manifests for fixed seeds `0`, `1`, and `2` contain 1,466 rows per
  seed. All `train_unlabeled` rows have empty label paths, including auxiliary25.
- Full HDF5 schema, manifest/split, hidden-label, source-provenance, and
  DataLoader (`num_workers=0/4`) acceptance gates passed.
- `checksums/checksums.sha256`, `manifests/transfer_manifest.json`, and
  `h5/v1/FROZEN` were generated and verified locally.
- The formal training DataLoader passed on `jiangsuiyang` with both
  `num_workers=0` and `num_workers=4`; its unlabeled batches contain no
  visible or hidden label field, M&Ms ED/ES remain patient-split consistent,
  and `auxiliary25` remains train-unlabeled only.
- Formal Fundus 2-case overfit passed at
  `/home/jiangsuiyang/SSL_CL/runs/m0/two_case_overfit_fundus_REFUGE_seed0_20260819T103301Z`:
  loss `2.576756 -> 0.106565`, mean foreground Dice `0.974170`, minimum Dice
  `0.965229`, and checkpoint reload max error `0`.
- The shared 2D U-Net, HDF5 labeled/unlabeled datasets, deterministic sampler,
  weak/strong transforms, trainer, checkpoint manager, continual runner,
  evaluator, site-matrix logger, and baseline methods are now implemented
  under `experiments/lcrseg/`.
- Registered baseline methods are `Static-Sup`, `Static-SSL`, `FineTune-Sup`,
  `Sequential-SSL`, `Uniform-KD/LwF`, `Joint-Sup`, `Joint-SSL`, and the
  separate `SS-EWC` control. All share one runner and evaluator.
- LCR-Seg V0.1 implements K=1 current/historical anchor banks, V1 detached
  learnability, V2 detached historical compatibility, and V3 continuous
  assimilation/consolidation routing. V4/V5, replay, diffusion, a third
  teacher, and K>1 are not implemented.
- The original local and remote V0.1 unit suite passed `25` tests. Remote exact
  checkpoint interruption/resume equivalence passed for Static-SSL
  (`model_max_abs=0`, equal optimizer/scheduler progress).
- The post-strict-warm-up 5-epoch Fundus pilot completed with all three sites'
  anchors valid, zero site warnings, a complete 3x3 validation matrix, and a
  verified fixed golden batch (all recorded tensor/loss errors `0`). It was an
  engineering pilot, not a 200-epoch research result; its raw run directory
  was pruned during the 2026-08-20 project storage cleanup after the evidence
  had been recorded.
- Separate hidden-GT diagnostics for the pilot were written under that run's
  `posthoc_reliability/` directory only. On Drishti-GS, learnability-valid
  coverage was `0.811171`; both learnability and compatibility bins increased
  strongly in observed accuracy. These diagnostics never entered training.
- Every new formal run now persists fully expanded method defaults in both its
  `config.yaml` and checkpoints, including LCR routing flags and anchor
  settings.
- The completed Fundus seed-0 suite includes budget-matched Static-Sup,
  Static-SSL, FineTune-Sup, Sequential-SSL, Uniform-KD, corrected Joint-SSL,
  SS-EWC, full LCR-Seg, and all three requested LCR routing ablations. The
  detailed, eligible result table is in
  `reports/implementation/BASELINE_AND_V0_V3_COMPLETION.md`.
- All eligible full runs finished without NaN rows. Full LCR used 13,400 steps,
  had zero AMP skips, valid anchors at all sites/classes, no anchor warnings,
  and a full-checkpoint golden verify with zero numerical error.
- Corrected schedule accounting now makes Joint-SSL and supervised controls
  exactly match the 13,400-step SSL/LCR cumulative budget. Earlier 13,200-step
  Joint and 6,600-step supervised artifacts are excluded from formal
  comparisons and were pruned during the 2026-08-20 project storage cleanup;
  their provenance remains recorded in the implementation report.
- The supplied V0.2 plan was copied unchanged to
  LCRSEG_V0_2_ASYMMETRIC_ROUTING_EXPERIMENT_PLAN.md and its pre-code,
  pre-training acknowledgement is frozen at
  reports/experiment_status/V0_2_PREREGISTRATION_ACK.md.
- The required independent V0.1 routing audit completed on all nine LCR
  final/site checkpoints, with all three Sequential-SSL checkpoints recorded
  as relation-state-not-applicable rather than assigned synthetic L/C values.
  Its frozen post-hoc outputs are under reports/analysis/v0_1_routing/, with
  the evidence and hashes recorded in
  reports/experiment_status/V0_1_ROUTING_DIAGNOSTIC.md. It confirms
  class/region non-monotonicity of raw compatibility and does not expose
  diagnostic labels to the training path.
- V0.2 now has an independent `lcrseg_v0_2` method with class-wise progressive
  admission, a labeled-only PAVA compatibility calibrator, and rejection-only
  relation KD. V0.1 source behavior remains protected by its unchanged golden
  regression: all recorded array and loss errors are exactly zero on the
  frozen RIM-ONE-r3 golden batch.
- The remote V0.2 suite passes `43` tests, including no-hidden-label,
  detached-mask, PAVA/resume, per-class cap, and V0.2 synthetic R3
  create-and-independent-verify golden tests. Evidence is frozen at
  `reports/experiment_status/V0_2_TEST_GATE.json`.
- The four immutable V0.2 Fundus configurations resolve from the frozen
  manifest to exactly 13,400 optimizer steps: REFUGE `8,000`, RIM-ONE-r3
  `3,200`, and Drishti-GS `2,200`.
- Literal V0.2 Fundus R0 completed its full 13,400-step seed-0 run without
  numerical failure. Its required comparison with the named V0.1
  uniform-relation reference differs by Final Dice `-0.0241101275` and
  Incoming Dice `-0.0289927683`. The required investigation found that the
  legacy reference used continuous V0.1 learnability weighting whereas literal
  V0.2 R0 uses unit assimilation. The frozen evidence and hard-stop decision
  are recorded in `reports/experiment_status/V0_2_R0_SHARED_PATH_STOP.md`.

## In progress

- The V0.1 Fundus research gate remains not met: full LCR does not exceed
  Sequential-SSL or Uniform-KD, and raw compatibility is not strictly
  monotonic class-wise. V0.2 R0 is now hard-stopped because its named legacy
  reference is not semantically identical to the literal unit-assimilation
  definition. R1/R2/R3, the automated V0.2 gate, and all Prostate work remain
  blocked pending an explicit protocol amendment or user direction; they must
  not be started from the current evidence.
- Prostate seed-0 `RUNMC -> BMC` is a deliberate gate stop, not an unstarted
  oversight. Do not start it without an explicit decision after the Fundus
  diagnosis.
- M&Ms canonical320 and all multi-seed/three-dataset formal experiments remain
  out of scope for this phase.

## Remote deployment status

- Remote transfer: `complete` on 2026-08-19 at
  `/home/jiangsuiyang/SSL_CL`.
- The direct source-bound endpoint was `jiangsuiyang@10.12.208.180:22` through
  local address `10.75.81.150`; the formal server interpreter is
  `/home/jiangsuiyang/anaconda3/envs/py38/bin/python` (Python 3.10.6,
  PyTorch 2.2.1+cu121, CUDA 12.1, cuDNN 8902).
- The canonical remote layout contains `h5/`, `manifests/`, `splits/`,
  `checksums/`, and `reports/preprocessing/`; raw source data was not copied.
- Remote `sha256sum -c checksums/checksums.sha256` passed all 2,962 frozen
  entries, and the remote HDF5 verifier passed all 2,932 HDF5 files.
- Independent post-transfer checks found zero `._*.h5` files. A transient
  report-only duplicate from the initial path-mapping repair was verified
  byte-identical to the canonical reports and removed; the remote layout is
  now canonical.

## Invariants

- Original DataP directories are read-only input.
- Derived HDF5 is versioned and immutable after acceptance.
- Training manifests omit labels for unlabeled samples.
- Run artifacts, checkpoints, diagnostics, and logs are written only beneath
  `/home/jiangsuiyang/SSL_CL/runs`, never beneath frozen inputs.
- This workspace is not a Git repository, so requested milestone commits are
  a documented blocker. No repository has been initialized and nothing has
  been pushed.

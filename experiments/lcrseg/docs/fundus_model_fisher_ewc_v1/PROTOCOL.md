# Fundus Model-Fisher EWC comparison, V1

Registration ID: `FUNDUS_MODEL_FISHER_EWC_V1`. Registered on 2026-09-01 (Asia/Shanghai), before any real-data access or real-model forward in this study. Source base: `5d4a7e96fac31a93df991d386ee6b8a9fa1adebf`. Branch: `codex/fundus-model-fisher-ewc-v1`. The registration commit must precede the engineering/execution commit; both must be published and byte-verified before computation.

## Authority, inherited evidence, and exclusions

The user's long-running authorization permits finite, prospectively registered Fundus comparisons on GPUs 4/5/6/7 with all generated storage on NAS. This is a new external-baseline adaptation, not a rescue or amendment of PMGC, LwF, Gate1C, MMPR, or legacy `ss_ewc`. Those studies and the prototype-derived method line remain closed. No C0 regeneration, Gate2, Prostate, MnMS, replay, prototype/transport mechanism, parameter search, paid resource, or main merge is introduced.

The closed source audit and synthetic engineering phase are immutable prerequisites. They established the categorical pixel model-Fisher semantics, exact actual-point denominator, image-only accessor, transactional state loading, mode/RNG preservation, shared-runner resume equivalence, and a six-group synthetic pass at source `ab1c40c312dcbebe5b6fd025bc17bb6fe02d68eb`. Their three invocation results and failed evidence are not rerun here. Synthetic coefficient/decay values `1.7/0.6` are not reused.

This study uses `model_fisher_ewc_v1`, not legacy `ss_ewc`. It is a fixed-class Fundus segmentation adaptation of the Fisher-weighted quadratic EWC principle, not a faithful reproduction of the original classification architecture, task protocol, or hyperparameter search. No external implementation is executed or copied.

## Fixed question, arms, and estimator settings

Question: under the same legacy weak/strong sequential SSL framework, does the registered categorical pixel Model-Fisher EWC adaptation improve final accuracy and retention across three Fundus domains and three paired seeds without materially reducing incoming-domain learning?

- S0: `sequential_ssl`, freshly trained control.
- S1: `model_fisher_ewc_v1`, the same shared SSL objective plus `lambda/2 * sum_j running_F[j] * (theta[j]-reference[j])^2` after the first consolidation.
- `lambda=1.0`. This is the unit coefficient under the explicitly retained `1/2` convention, chosen before data access without a search or claim that it came from the EWC paper.
- `gamma=1.0`. The running diagonal is an undiscounted sum across the three stages; no decay grid is used.
- Per stage, uniformly select at most 16 current-domain visible-training images without replacement and at most 16 output pixels per selected image, using `fisher_seed=optimization seed` and the registered site namespace. The actual labeled-image counts are 40/16/10, so the fixed maximum is 16/16/10 images and 256/256/160 pixels per run stage.
- For every selected pixel, use all three categorical classes including background, detached model probabilities, and separately squared scalar log-probability gradients. Divide by the exact actual selected-pixel count. No label, Dice, confidence mask, class balancing, prototype, transport, replay, trace normalization, or training transform enters Fisher.
- Consolidate after every successful stage, including the terminal stage, before the stage-final checkpoint. The terminal state cannot affect an already completed update but preserves a complete continuation checkpoint. A failed estimate publishes no partial state.

Seeds 0/1/2 use their corresponding frozen splits and the same optimization seed within each pair. Both arms start from fresh paired initialization and input order; no historical checkpoint or LwF control artifact is reused. The domain order is REFUGE -> RIM_ONE_r3 -> Drishti_GS with the existing 20% visible-label training view. Per seed the labeled/unlabeled/validation/test counts are 40/160/100/100, 16/63/40/40, and 10/41/25/25.

Training is unchanged from the closed matched LwF comparison: CE + foreground soft Dice, current weak-view hard pseudo-labels at confidence 0.95, strong-view CE under the cutout-valid mask, SSL coefficient 1 with a 1,000-step per-domain ramp, Adam at 0.0005, weight decay 0.00001, FP32, labeled/unlabeled batches 2/4, and 200 epochs per domain. Each run receives 8,000 + 3,200 + 2,200 = 13,400 optimizer updates. Six runs permit at most 80,400 formal optimizer updates. No AMP skip, nonfinite value, shortened schedule, added arm, or changed configuration is admissible.

## Prospective computation budgets

Each shared training update makes exactly three current-model calls: labeled, unlabeled weak, and unlabeled strong. Validation evaluates all three domains after every stage at batch size four, for 126 calls per run. Therefore S0 permits 40,326 training-plus-validation model calls per run. S1 permits those calls plus at most 42 Fisher image calls, for 40,368 per run. The six formal runs permit at most 242,082 model calls in total, including 126 Fisher calls. S1 permits exactly at most 2,016 Fisher `autograd.grad` calls per run and 6,048 across three seeds. These budgets exclude the separately bounded preflight and post-training test readout below.

At most three formal workers run simultaneously, one seed queue each on GPU4/5/6; GPU7 is reserved for the one engineering check and the eventual readout. Each formal child has a 12-hour limit. At most two infrastructure-only resumes per run may use a verified `checkpoint_last.pt` from the same run/config/source. Observation loss alone never authorizes restart, and a scientific or code failure is not an infrastructure resume.

## Engineering admission before formal training

There is one create-only real visible-training engineering attempt, after the registration and exact check source are published and verified. It uses seed0 REFUGE only, no validation/test role, no hidden label, and the formal U-Net, FP32, method settings, image cap, and point cap.

For each arm it performs two no-update golden calls and one shared-Trainer update on the same labeled batch of two and unlabeled batch of four. S1 additionally performs one full registered Fisher consolidation, checks exact 16-image/256-point/768-gradient counts, finite nonnegative state, nonzero Fisher, immutable parameters/buffers/optimizer/RNG during estimation, then applies a deterministic 0.001 perturbation at a positive-Fisher entry and performs one shared-Trainer update with a positive finite EWC penalty. Maximums: 37 real-model calls, 768 `autograd.grad` calls, three optimizer updates, one invocation, and 900 seconds. Input bytes are rehashed afterward. A failure closes this version as `FAIL_ENGINEERING`; it cannot be retried with changed images, points, coefficient, tolerance, or code under this registration.

The synthetic phase is inherited rather than re-executed. Formal training starts only if the real check exits 0, all input hashes and source bytes match, its observed duration fits the formal 12-hour bound, and an engineering-admission record is published. No overfit rerun is needed because the exact shared Trainer and new method path already passed the frozen two-case synthetic overfit and resume checks.

## Evaluation and success rule

Formal training reads only train-labeled, train-unlabeled, and visible validation roles. Validation metrics are descriptive and cannot change settings, checkpoints, schedule, or continuation. There is no checkpoint selection: use every stage-final checkpoint and the fixed full budget.

Only after all six runs exit 0 and pass source/config/budget/checkpoint/Fisher-state/input-immutability gates may a separately published evaluator open the test role once. It evaluates each stage-final checkpoint on every seen test domain, producing all 36 required cells and at most 612 model calls / 2,430 case predictions, with zero optimizer updates. Test values cannot trigger a retry, coefficient change, threshold adjustment, added seed, or checkpoint choice. Private case tables and predictions remain on NAS; public evidence is aggregate.

Use patient-mean hard foreground Dice exactly as in the closed LwF comparator: empty/empty class Dice is one; average classes 1/2 within patient and then patients within domain. For each seed/arm, F is the mean of the final row's three domains, I is the mean of the three diagonal entries, and BWT is the mean final-minus-diagonal Dice for the first two domains. Pair S1-S0 within seed.

All five conditions are required: mean delta F >= 0.01; delta F > 0 in at least 2/3 seeds; every seed delta F >= -0.01; mean delta BWT >= 0.01; mean delta I >= -0.01. These are the already registered one-Dice-point operational margins, reused unchanged for external-baseline comparability. `PASS_EWC_FEASIBILITY` requires every condition. Any failed condition after valid execution is `FAIL_EWC_FEASIBILITY`; incomplete, corrupt, nonfinite, over-budget, or mismatched evidence is `FAIL_ENGINEERING`. Three seeds support no significance or clinical claim. A pass is baseline feasibility only, not overall project success.

## Storage, provenance, and closure

Canonical create-only NAS root: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/fundus_model_fisher_ewc_v1_20260901`. All runs, checkpoints, Fisher state, logs, predictions, caches, scratch, temporary files, and archives live there and are launched through `with_nas_storage.sh`. The actual NFS mount and a write/read probe must pass before writes. The existing Python environment and Torch stay unchanged; there is no HOME output fallback.

The exact execution checkout, resolved configs, registration/check hashes, input inventory, commands, parent-observed child exits, latest resume checkpoint, all final checkpoints, Fisher summaries, and failures are retained. Frozen HDF5/manifests/splits/checksums are read-only and rehashed; the HOME runs path remains its NAS symlink. Publish sanitized source and aggregate reports only on this branch. Before closure, run a separate zero-model artifact/arithmetic audit and create an additive verified NAS archive. A same-NAS copy is not an independent-device backup. Keep the 30-minute heartbeat and never launch duplicate workers.

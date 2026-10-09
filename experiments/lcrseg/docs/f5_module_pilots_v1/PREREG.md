# F5 module pilots V1 — fixed incremental experiment

User authority, 2026-10-09: “按照优先级，对这几个方法的模块都进行尝试，设置每小时监测”. This authorizes this new finite pilot after pausing RL. It does not reopen old DAGs or rewrite their stops/negative results. No external-review approval is claimed; current user authorization and the new code-bound qualification admit this new runner.

## Parent and scientific scope

Parent F5_C02: NATIVE_LR_SRC_A_3DOMAIN_V1 input-side projected A/B adapters, LCTX CE+Dice, current EMA/PAS, and `F_prev @ (I + Q R Q^T)`. Current L updates original A/B and R. U KL+0.2 SWD is multiplied by ramp*0.25 and updates R only. Rank ratio .25; initial A/B learning rate .0005, R .001; original Adam/decay/polynomial schedule and .2 warmup/ramp unchanged.

Fixed seeds 163/164 and orders REFUGE→RIM→Drishti / REFUGE→Drishti→RIM. Source S163/S164 (8000 updates) is reused from the completed F5 confirmation; no source training. Target stages are 3200 RIM and 2100 Drishti updates. No new patients, splits, label budgets, augmentation search, best-checkpoint selection, image banks, hidden U labels, test readout, RL or replay.

Run four fresh F5 trajectories, then four boundary trajectories, then four OT trajectories, then four adapter trajectories. All modules start from the same respective source, not the previous module's result. Each stage2 inherits its own arm's sealed stage1. Baseline costs are necessary matched controls, not counted as a new module.

## Three paper adaptations

1. [Combining Boundary Supervision and Segment-Level Regularization](https://arxiv.org/abs/2604.01859), §3. The original is temporal segmentation with an extra boundary channel. Our explicit 2D loss adaptation derives edge probabilities as half the class-probability L1 change between adjacent pixels, taking the larger horizontal/vertical change. Current L GT supplies binary transitions. Edge BCE is restricted to a 2px boundary band, with weight .05. The interior term matches x/y marginal CDFs of predicted class probability restricted to GT class interiors to the uniform pixel distribution, weight .05, after original warmup. Exclude image edges, ignore neighborhoods and 2px mixed-input seam neighborhoods. No new head, teacher or inference-time computation; this is not topology preservation or the original temporal CDF reproduction.

2. [Optimal Transport Metric Learning for Feature Alignment](https://arxiv.org/abs/2609.19176), distribution-triplet component. Keep F5 SWD, add foreground-class positive/negative Sinkhorn-divergence margin .1. Use existing projected clean-U student features and clean-L teacher features from F5 forwards. Normalize features, sample at most32/class with an independent stateless stream; require8/class. Current L references are two detached centroids/class from5 fixed k-means iterations, not the paper's learnable prototype bank. No persistent features or historical samples. Uniform entropy OT epsilon .1,100 log-Sinkhorn iterations, marginal residual≤.001, envelope gradient; anchor self-cost cancels in the positive-minus-negative divergence. Add weight .2 inside the unchanged ramp*.25 U objective. Missing support yields graph-connected zero with explicit counts, not fabricated labels. U gradients remain R-only.

3. [Per-Loss Adapters for Gradient Conflict in PINNs](https://arxiv.org/abs/2605.10136), shared-output loss-indexed low-rank mixing. A linear, mergeable adaptation keeps the full original R and adds three rank1 branches indexed supervised/KL/SWD: `R_eff=R+(1/3)sum(up_k @ down_k)`. Up starts0; down is normalized deterministic private Gaussian initialization. Losses share the output and can update every branch: there is no invented exclusive gradient routing. Orthogonality of normalized down vectors adds .001 to L. U permission expands only within the same Q-sidecar; original A/B still receive no U gradients. EMA averages the effective R matrix, never the factors; deployment seals the same single16×16 transform. Effective linear function capacity equals original full R; parameterization and optimization change. This is not the paper's nonlinear adapter reproduction or proof of persistent conflict.

## Budgets and admission

- Formal: 4 arms ×2 seeds ×2 orders ×5300 = **84800** physical/scientific updates, 32 target stages; source0.
- Native generated CUDA qualification cap64, planned28 including4 deliberate after-optimizer failures. CPU module qualification cap768, planned192. Discarded real current-L smoke cap16, planned16; source synthetic forwards2, patient source forwards0. All failed costs remain; engineering repairs may not reset any cap.
- Formal gradient audit: four predefined points/stage, L versus total U in the Q-sidecar, two VJPs/point, cap256. Qualification VJPs are separately counted, overall diagnostic cap272. Sparse gradient samples are diagnostics, not a full conflict-regime proof.
- Check original historical F5 warmup/active state parity; meaningful module mathematical/finite-difference tests and two-case module overfit; native one-batch backward, U-gradient permissions, teacher freeze, checkpoint continuation, deployment merge, own-prefix stage transition and uncommitted failure accounting. No formal worker starts before all qualification succeeds. If qualification fails, retain ledgers and repair only within unchanged scientific definitions/caps.
- Native outputs, checkpoints, logs and temporary artifacts use a new create-only canonical NAS root with the storage wrapper. GPUs4–7, one trajectory/GPU, minimum4GiB free at each batch admission. Do not kill or reset any existing process. No failed formal retry or extra experiment.

## Readout and decision

Seal all32 target students before any target validation readout. Then evaluate each stage on its seen domains (80 domain evaluations), using existing patient-mean rim/cup macro Dice. Final=(2Old+Incoming)/3; Forget is the mean source/first-target early-to-final decline. Stage1 is evaluated for Forget only after the entire matrix is sealed. No diagnostic labels enter training. Historical B0/B2 seed163/164 aggregates are reused with zero new evaluation and reported separately from the fresh F5 control.

Primary increment is each module minus fresh F5: average orders within seed, retain both seed effects, all orders/domains/classes and negative outcomes. Practical advancement requires both seed Final increments>0 and mean≥.005, each order mean Old and Incoming≥−.005, and every individual seed/order/domain/class delta≥−.05. Apply the same descriptive gate against B0/B2 and disclose all tradeoffs; a useful improvement over F5 alone does not establish the strongest method. No automatic combination/tuning/extra seed follows any outcome.

This reuses repeatedly observed development patients. Optimization-seed checks are not independent patient confirmation, SOTA, clinical safety or proof of novelty. The observed seed164 F5 weakness and historical failed gates stay in the comparison; no favorable subgroup selection.

## Execution qualification and startup

Execution commit `c86ae7fe77d5afb2bc6744b392fa441f81b62775` passed native qualification and started the four matched F5 control workers on GPUs4–7. `STARTUP.json` records the one brief startup check; it is not a completion receipt. The hourly heartbeat id is `f5`; prior RL monitoring remains paused.

Three failed engineering qualification attempts remain in `QUALIFICATION.json`: a generated image too small for the fixed interior-support rule (0 optimizer calls), missing deterministic CuBLAS configuration (192 cumulative CPU, 0 CUDA calls), and unsupported deterministic floating CUDA cumsum (192 cumulative CPU, 10 CUDA calls). The final passing attempt brings totals to384 CPU,38 generated CUDA and16 discarded real-L smoke calls, within the original caps. These include failed and deliberately interrupted optimizer invocations. No formal work was retried or validation read before startup.

Only the generated test grid was enlarged. Deterministic CUDA uses `CUBLAS_WORKSPACE_CONFIG=:4096:8`; spatial CDF transfers the two small marginals to CPU for cumulative summation while preserving gradients. Successful CPU tests may be reused only when the tested mathematical function and module source signatures match. All frozen scientific definitions, coefficients, data boundaries and optimizer caps remained unchanged.

Every hour, check compact status/ledger/exit evidence once. Stay quiet without a meaningful change. On completion, audit once and publish only source, protocol, aggregate results, costs and report to this project's GitHub branch through the required proxy. Keep raw data, patient rows, weights, private configuration/logs on NAS. After verified delivery, pause this pilot's heartbeat; the prior RL automation remains paused.

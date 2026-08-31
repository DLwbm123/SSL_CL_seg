# EWC source audit V1

Scope: one non-prototype candidate, source reading and closed-form arithmetic only. Base commit `a4dd264e813dc477fcc7ceea21975d3d4a9850c5`; branch `codex/fundus-ewc-audit-v1`. No Fundus image, label, model checkpoint or new model is used. Real and synthetic-network forwards, gradient calls and optimizer steps are all capped at **zero**. At most two invocations of the standard-library arithmetic check are allowed, retaining any failure. This is not a registered EWC performance experiment.

**Admission decision: the existing `ss_ewc` implementation is not admitted as a standard model-Fisher EWC comparison.** This is a source/engineering finding, not a negative EWC performance result. No EWC training is launched. The closed LwF comparison and all older prototype-line results remain unchanged.

## Primary evidence and implementation identity

[Kirkpatrick et al., Eq. 3](https://arxiv.org/html/1612.00796v2#S2) specifies a Fisher-weighted parameter penalty with coefficient lambda/2. It does not provide a Fundus segmentation protocol or a transferable Fundus coefficient. The original task, model, loss and training schedule differ from this repository's SSL setting.

[Schwarz et al., Section 4, Eqs. 7–9](https://proceedings.mlr.press/v80/schwarz18a/schwarz18a.pdf) distinguishes online accumulation at the latest parameter reference from separate historical penalties. Its online formulation includes decay and discusses per-task Fisher normalization. This is not the whole Progress & Compress architecture, and implementing only a penalty must not be called a P&C reproduction. The paper's discussion does not by itself fix a complete Fundus normalization contract.

[Van de Ven's 2025 primary study](https://iclr-blogposts.github.io/2025/blog/fisher/) explicitly separates the model-label expectation, empirical-label estimator and batched approximation. Its author-linked [code](https://github.com/GMvandeVen/continual-learning/tree/e6d795aa81b9cef742b8de76cb71222d4d1ce00b) is pinned at `e6d795aa81b9cef742b8de76cb71222d4d1ce00b`. The `fisher_labels='all'` branch weights each separate class-gradient square by its detached predicted probability. Its classification protocol and coefficient search are not imported into Fundus. This is the 2025 study author's implementation, **not verified original 2017-author code**.

A bounded search did not establish an original 2017-author EWC release. The complete `google-deepmind/deepmind-research` tree at `f5de0ede8430809180254ee957abf36ed62579ef` had no path matching `ewc`, `elastic`, or `progress_compress`; this is not a proof of absence across all repositories/history. Its [continual_learning directory](https://github.com/google-deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/continual_learning/README.md) describes a different 2021 classifier-ensemble method and is not accepted as EWC source. No other candidate or prototype mechanism was pursued.

Exact reviewed file hashes and source roles are recorded in [SOURCES.json](SOURCES.json). No external implementation is executed, installed or copied into the training engine.

## Local findings

The reviewed local estimator is `experiments/lcrseg/lcrseg/methods/ewc.py`, SHA-256 `a832e123c9efc44d474de99d47f63447f28c7baa05d3ae314dcda24f9d706724`. These findings concern this exact version:

| Contract | Observed implementation | Consequence |
| --- | --- | --- |
| Differentiated quantity | `estimate_fisher` differentiates `loss_sup`, which the shared base defines as CE plus foreground Dice by default | This composite-loss gradient square is not a categorical log-likelihood Fisher |
| Aggregation | Each minibatch loss is differentiated, then the aggregated gradient is squared and averaged over batches | Cross-example cancellation and cross terms prevent equivalence to averaging individual gradient squares; a scalar lambda cannot generally repair it |
| Target distribution | Uses visible ground-truth labels through the training batcher | Even a per-example version would be an empirical estimator, not the model-label expectation |
| Coefficient and online state | Local defaults are lambda=0.1, gamma=1, eight batches; penalty has no factor 1/2; references are reset to the latest parameters | These are repository adaptation choices, not verified paper hyperparameters; the coefficient convention and Fisher units need explicit registration |
| Estimator input and RNG | Reuses the augmented labeled training batcher; flips consume the global Torch RNG; no local RNG save/restore wraps estimation | Fisher work can change later training augmentation draws; a matched comparison must explicitly isolate or disclose this effect |
| Checkpoint admission | Missing EWC state becomes empty dictionaries; missing parameter entries are silently skipped by the penalty | Corrupt/incomplete state can silently disable regularization; new execution needs strict key/shape/finite checks |
| Existing test | Checks that importance exists, penalty is zero at the reference and positive after a perturbation | It does not establish correct Fisher values, aggregation, sample counting or complete resume semantics |

The shared runner invokes estimation after each domain and before its final checkpoint, including the terminal domain. With the existing labeled batch size 2 and visible counts 40/16/10, the current eight-batch setting would imply **8+8+5 extra estimator forwards/backward calls per run**, beyond optimizer-update counts. This is a static budget derivation; none were executed in this audit. Any future registration must count these operations explicitly and decide terminal-stage behavior before computation.

## Closed-form checks and external-source caveat

For a scalar Bernoulli logit with probability 1/2, two opposite labels give gradients +1/2 and -1/2. Averaging their squares gives 1/4, while squaring their average gives zero. Splitting the same observations into singleton batches changes the latter statistic. This proves a structural difference rather than a missing constant multiplier. With probability 0.8 and two observed zero labels, the model Fisher is 0.16 whereas the empirical gradient-square statistic is 0.64.

The pinned 2025 reference also needs care: in `models/cl/continual_learner.py:225`, the stopping rule is `index > fisher_n`; at line 288, normalization divides by the zero-based `index`. With no cap, one datum gives divisor zero, and two data give divisor one. With cap=1 and a longer loader it processes two samples; cap=500 can process 501. These are control-flow consequences of the inspected source, not measurements from its paper. We must not port that counting logic unchanged.

[check_fisher_statistics.py](check_fisher_statistics.py) verifies the pinned local/external bytes and these arithmetic/control-flow examples using only Python's standard library. It creates no network, invokes no data loader, and reads no experiment dataset. The actual execution receipt is recorded separately in `STATUS.json` and the NAS evidence packet; source reading alone is not labeled a completed runtime check.

## Next bounded engineering contract

The next allowed work is a **separately recorded synthetic engineering phase**, before any new real-data registration or forward. It should reuse the shared engine, checkpoint/RNG helpers and existing environment, preserving this legacy implementation and all historical execution checkouts.

The engineering target is a clearly named model-Fisher EWC adaptation, with a declared categorical sample unit. For segmentation this requires choosing and documenting image-versus-pixel likelihood, point sampling, all three classes, and exact normalization; it cannot be inferred from a classification batch. The all-label expectation must square individual log-probability gradients before weighting/summing them, with detached probabilities. No Dice, supervised target labels, confidence mask, prototype, transport or old-image replay enters the Fisher quantity.

Before real-data admission, require closed-form value and gradient agreement on tiny linear models; exact actual-sample denominators for empty/one/two/capped/short-loader cases; no dependence on incidental minibatch partition; finite nonnegative diagonals; no model, optimizer or reference update while estimating; strict complete-state reload; preserved training/evaluation mode and RNG; and a shared-runner uninterrupted-versus-resumed synthetic comparison. Preserve failed checks and use a finite suite budget defined before invoking that phase.

Only after these semantics and checks are complete may a separate Fundus performance study freeze its coefficient/normalization pair, point/sample budget, stage policy, seeds, controls, evaluation role, success bounds and failure exit. No coefficient search or performance claims are authorized by this audit. The already published LwF test results must not become tuning targets. A 2017 or 2025 classification coefficient cannot be transplanted and called a verified Fundus setting.

All source captures, logs and execution evidence use a new create-only NAS root. PMGC remains closed; the prototype-derived line remains ended; no C0, Gate2, Prostate, MnMS, unbounded sweep or main merge is introduced.

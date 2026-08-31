# Model-Fisher EWC V1: synthetic engineering registration

This is a separate, prospectively bounded engineering phase, not a Fundus performance study or an EWC success claim. It follows the closed [source audit](../ewc_audit_v1/AUDIT.md). Base commit: `a9a6ff3113f0774be2bfb306be030c28069c4758`; branch: `codex/ewc-model-fisher-engineering-v1`. No result from the closed LwF comparison is used to choose settings. Legacy `ss_ewc`, LCR-Seg equations, frozen inputs, and historical execution checkouts remain unchanged.

The later [long-running authorization](../pmgc_v0_1/LONG_RUNNING_FOLLOWUP_AUTHORIZATION.md) permits this finite external-baseline engineering work. Only CPU synthetic fixtures are authorized here. Real Fundus images, labels, checkpoints and model forwards are prohibited. There is no new performance registration or GPU training authorization in this document.

## Fixed estimator contract

The new method identity is `model_fisher_ewc_v1`, version `1.0`. It inherits the shared sequential SSL objective and uses the existing continual runner and optimizer step. It never creates an old teacher, prototype, anchor, transport, replay store, or independent training loop.

For a selected image `x`, an output pixel `u` is one categorical observation with three mutually exclusive classes, including background. The target is the conditional marginal pixel model, not the log likelihood of a whole label map. For the selected pixel set `S`, at the current parameter vector:

```
F[j] = sum_(x,u in S) sum_(c=0..2) stopgrad(p(c|x,u))
       * (d log p(c|x,u) / d theta[j])^2 / actual_number_of_selected_pixels
```

Each scalar log-probability gradient is squared separately before weighting or summing. Use all three labels, stable log-softmax, no temperature, no ground-truth labels, Dice, confidence/validity mask, class balancing, or trace normalization in this quantity. A trainable parameter unused by logits has zero Fisher; it still has an explicit reference and state entry. Only float32/float64 model parameters are admitted in this version; no AMP estimation.

The image source is an image-only view of the current site's existing labeled-training dataset. Reuse its normalized HDF5 image reader without invoking its label reader or training transforms. The shared dataset gains only an additive `image_at(index)` accessor; the historical `__getitem__` path is unchanged. Test image providers implement the same small accessor. Existing `DeterministicBatcher` is passed by the shared stage-end hook, but its incidental minibatch partition does not define the statistic.

The required explicit settings are `ewc_lambda`, `ewc_gamma`, `fisher_max_images`, `fisher_points_per_image`, and `fisher_seed`. No Fundus defaults are supplied. Lambda is finite and nonnegative; gamma is finite in [0,1]; both caps are positive integers; seed is an integer in [0, 2^63). Reject the legacy `ewc_fisher_batches` option for this new identity.

Select images uniformly without replacement using a dedicated CPU Torch generator seeded by the existing `_namespace_seed(fisher_seed, 'model_fisher_ewc_v1:' + site_id, 0)`. First generate `randperm(dataset_length)` and retain at most the image cap; then generate one `randperm(H*W)` per selected image, retaining at most the pixel cap. Image and pixel selection do not inspect labels, predictions or scores. The denominator is the actual total pixel count, including short datasets/images; reject empty input rather than divide by zero. Each selected image is forwarded separately, once, in evaluation mode, so grouping those images into different training batches cannot change the estimator. An estimate uses `M` model calls and `3*P` autograd calls, for actual image/pixel counts `M/P`.

Capture and restore the existing Python/NumPy/Torch CPU/CUDA RNG state and every module's training/evaluation flag in `finally`, including failure paths. Use `autograd.grad` without writing or clearing parameter `.grad`. Parameters, buffers, optimizer state, and existing reference/Fisher tensors must not change during estimation. Build new importance and reference dictionaries locally; publish them only after a complete, finite, nonnegative estimate passes validation. Do not execute this routine in inference mode.

## Consolidation and checkpoint contract

After each successful stage, including the terminal stage under the unchanged runner hook:

```
running_F = gamma * previous_running_F + current_stage_F
reference = detached_copy(current_parameters)
loss = shared_sequential_SSL_loss + lambda/2 * sum_j running_F[j] * (theta[j]-reference[j])^2
```

The first stage has no penalty. There is one successful consolidation per stage; duplicate or out-of-order consolidation is an error. Missing state is valid only for a newly initialized first stage, not for a checkpoint. New settings and state schema identify this estimator separately from legacy EWC.

Serialize the estimator schema, complete resolved method settings, consolidation count, and complete trainable-parameter reference/Fisher dictionaries under a distinct method-statistics key. Check exact keys, shapes, dtypes, finite tensors, nonnegative Fisher, detached ownership, compatible settings and stage count before assigning imported EWC state. Reject old-teacher/anchor state, incompatible method identity/version, incomplete state and changed configuration. State exports and imports must not alias the model, each other, or their source payload. Previous-stage loading and same-stage resume must both use this validation. Retain shared optimizer/scheduler/scaler and RNG restoration; do not implement a second resume engine.

## Frozen synthetic checks and admission gates

Only a tiny 1x1 three-class linear segmentation model, optionally with BatchNorm/dropout for state/RNG checks, is used. An unused trainable scalar exercises zero-gradient entries. The shared runner's model constructor is replaced by that toy model only inside the check harness. This does not establish U-Net or real-data readiness.

The finite check groups are:

1. **Closed-form Fisher:** zero linear weights and biases giving probabilities (0.2,0.3,0.5), with nonconstant synthetic inputs. Check bias Fisher `p*(1-p)` and weight Fisher `p*(1-p)*mean(x^2)` at float64 tolerance 1e-10. Check the unused parameter is zero and every class contributes. The image provider's label-bearing accessor must raise if touched.
2. **Counting and partition:** empty, one, two, capped and short image sequences; caps larger than the available pixels; no repeated/extra image access. Check exact actual `M/P/3P` counts and identical Fisher under different batch sizes. Invalid/nonfinite inputs must fail without consolidated-state mutation.
3. **State/RNG isolation:** preserve mixed module modes, BatchNorm buffers, parameters, existing gradients, optimizer state and all RNGs on success and on an injected image-read failure. No old-reference/Fisher mutation before successful publication. Export/load without aliasing; reject missing/extra/wrong-shape/wrong-dtype/nonfinite/negative/incompatible state and duplicate stage consolidation.
4. **Penalty/backward/golden:** engineering-only lambda=1.7 and gamma=0.6 exercise nontrivial scaling and accumulation. Check penalty and its parameter gradient against the quadratic formula, zero penalty at the reference, exact preservation of the first-stage shared SSL objective, and repeatable logits/losses. Use the shared Trainer for a backward/update check; Fisher/reference tensors never acquire gradients.
5. **Shared runner resume:** reuse the existing synthetic HDF5 fixture builder for two toy sites (two labeled, one unlabeled and one validation record per site). Use four shared optimizer updates per site; compare one uninterrupted eight-update run with a run interrupted after update six and resumed for two. Use stochastic transforms and toy dropout/BatchNorm. Require exact checkpoint tensor/RNG/optimizer/scheduler/method-state equality (atol=rtol=0), excluding only the differing output run name/config invocation control. Verify all synthetic input bytes remain unchanged after the run. Evaluation is only the synthetic validation role.
6. **Two-case synthetic overfit:** two fixed 8x8 three-class one-hot input/label patterns, 100 shared Trainer updates, Adam learning rate 0.1 with the shared cosine schedule, weight decay zero, AMP off, assimilation weight zero. Require final shared supervised CE+Dice <=0.1, at least 90% reduction from its initial value, and pixel accuracy >=0.98. This is a toy engineering gate, not medical-segmentation performance.

Sampling tests use at most three synthetic images and four selected pixels per image unless testing caps larger than available input. Shared-runner Fisher settings are image cap=2, pixel cap=2, seed=271828; coefficient/decay are 1.7/0.6. Test-only variations enumerate API boundary conditions and are not a hyperparameter search. They confer no recommendation for a real-data coefficient.

## Execution budget and evidence

Before the first invocation, publish this registration and the exact implementation/check commit, verify the NAS checkout matches that commit, and save the command, hashes, limits and environment. Do not perform exploratory model forwards to develop the tests before this step.

There are at most **three complete check invocations**, stopping after the first full pass. Per invocation the enforced upper bounds are **2,048 toy model calls**, **4,096 images through toy models**, **2,048 autograd.grad calls**, **512 backward calls**, **256 optimizer updates**, and **300 seconds**. All model calls in the harness use the counted toy model; wrap the gradient APIs and Adam step before checks begin. Record actual counts, each group's outcome, and parent-observed exit status even on failure. Each invocation gets a new directory. A bug fix within these frozen semantics requires a new source commit and keeps the failed evidence; do not change tolerances, data, goals or budget after seeing outcomes. Exhaustion or a contract gap blocks engineering admission.

All six groups must pass with zero skipped checks for `PASS_SYNTHETIC_ENGINEERING`; otherwise the phase is incomplete/failed, never scientific success. That result alone does not authorize real Fundus forwards. Any next real-data engineering/performance phase requires a separate prospective contract, finite limits, matched control, seeds, evaluation-role isolation and failure exit, published before computation.

Canonical new NAS root: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/ewc_model_fisher_engineering_v1_20260901`. Use a create-only code/operation/run layout, the existing storage wrapper, a verified NFS mount and write/read probe, and the existing server Python environment. Set CUDA visibility empty for this CPU phase. All fixtures, checkpoints, logs, caches and temporary files go to NAS; no HOME fallback, dependency installation or large local artifact. Verify source/result hashes and preserve originals before sealing a separate NAS evidence copy; do not call it independent-device backup.

The source audit and LwF/PMGC/Gate1C/MMPR studies remain closed. The prototype-derived method line remains ended. No C0 regeneration, Gate2, Prostate, MnMS, unbounded search, paid resource or main merge is introduced.

# PRES-DSR-SF V0.2 execution authorization

This document authorizes one validation-only execution of `PRES_DSR_SF_V0_2_DESCRIPTOR_MEMORY_SOFT_ROUTER` on `codex/pres-dsr-sf-v0-2-feasibility`. It is bound without amendment to preregistration commit `c4767688e01ee9106d172a88a95f7e6c8a5de0eb`; its Markdown and JSON hashes are recorded in the adjacent authorization JSON.

## Authorized work

The run may recompute clean M1/M2 controls, extract frozen image-only style descriptors, create capped train-only domain memories, select ridge lambda and temperatures only from deterministic train folds, solve the CPU-float64 closed-form router, evaluate hard routing and primary soft expert-probability fusion on validation, execute five cached train-case bootstraps, and produce the registered tests, reports, manifests, private archive, sanitized public artifacts, publication receipt, and hard stop.

This authorization becomes effective only after this authorization itself is committed, pushed, and remote-verified. Exact source and tests must then be published and remote-verified. Before the first real model forward, the canonical NAS mount/write probe, frozen private-input audit, zero-forward backend import-order audit, and exact manifest-derived call graph must all pass. Required tests must finish with zero failures, errors, or skips.

## Isolation and prohibitions

The pinned JASCL classifier must be imported before the registered deterministic backend is applied and frozen, while model construction, checkpoint/HDF5 reads, forwards, and avoidable CUDA initialization remain zero. Any later backend-state change blocks the run.

No segmentation optimizer step, model autograd/backward/parameter-grad write, training, EMA/GAS/PAS update, expert fine-tuning, validation-driven router fitting, test evaluation, C0 regeneration, LoRA/adapter, prototype pseudo-label, feature transport, relation loss, gradient projection, Prostate, MnMS, full sweep, main merge, or historical-artifact modification is authorized. Router fitting is CPU-float64 closed-form linear algebra only. Model optimizer/autograd/backward/grad-write counters stay zero; `router_is_closed_form=true`, `method_registered=false`, and `training_launched=false`.

## Attempt, storage, and stop

This is one formal prospective attempt, not PRES V0.1 attempt 2. A durable create-only resume may only complete missing cells after infrastructure interruption with identical source, inputs, folds, draws, gates, and existing artifacts. Outputs, caches, logs, model downloads, and temporary files belong in a new create-only NAS directory and every experiment command runs through `experiments/lcrseg/scripts/with_nas_storage.sh`; there is no home fallback.

After the validation report and publication receipt are pushed and their exact remote SHAs are verified, execution stops for independent review. No test evaluation, regeneration, router retraining on validation, expert fine-tuning, training, other benchmark, sweep, or main merge may start automatically.

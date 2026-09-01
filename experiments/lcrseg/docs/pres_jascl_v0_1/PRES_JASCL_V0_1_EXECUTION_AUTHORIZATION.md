# PRES-JASCL V0.1 execution authorization

This document authorizes one validation-only execution of registration `PRES_JASCL_V0_1_SNAPSHOT_DOMAIN_ROUTER` on branch `codex/pres-jascl-v0-1-routing-feasibility`. It is bound without amendment to preregistration commit `cd797d55362fd997beb6a9b7d5878aa790392831`, whose local SHA, `git ls-remote` SHA, and GitHub branch API SHA matched and whose Markdown/JSON files both returned HTTP 200. The preregistration file SHAs are recorded in the adjacent JSON.

## Authorized scope

Only the following work is authorized: verify and read the complete frozen Gate1C regenerated-B0 bundle and exact nine stage-best checkpoints; build an immutable three-snapshot expert bank for each seed; extract frozen image-level RGB/enc1/enc2 style descriptors; build M1/M2 domain prototypes from `train_unlabeled`; run Stage1/Stage2 validation routing, the complete 3×3 cross-expert validation segmentation matrix, fixed Oracle-snapshot/Prototype-routed/Shared-final comparisons, and five registered case bootstraps; then produce the registered tests, reports, manifests, CSVs, audits, archive, commit/push verification, and hard stop.

This authorization becomes executable only after this authorization itself is committed, pushed, and remote-verified. Exact execution source must then be committed, pushed, and remote-verified; the canonical NAS mount and a real write/read probe must pass; the frozen bundle-content/manifest SHAs, nine checkpoint SHAs, and 2962 data checksums must all match; required tests must have zero failures, errors, or skips; and the exact manifest-derived `PRES_JASCL_CALL_GRAPH.json` must be frozen before the first real model forward.

## Prohibitions and counters

No optimizer may be constructed or stepped. `autograd.grad`, backward, parameter-grad writes, model training, EMA/GAS/PAS/K2 updates, C0 regeneration, test construction/GT, hidden GT, validation GT outside the independent segmentation evaluator, expert reselection, forbidden router inputs, PMGC virtual-step outputs, LoRA/adapters, MILE claims or implementation, Prostate, MnMS, Gate2, full sweeps, method training, main merge, and historical-artifact modification are prohibited.

The mandatory terminal counters are `model_optimizer_steps=0`, `router_optimizer_steps=0`, `autograd_calls=0`, `backward_calls=0`, `parameter_grad_writes=0`, `method_registered=false`, and `training_launched=false`. Model/checkpoint state must remain bitwise unchanged and all required artifact hashes must be complete.

## Attempt, storage, and stop

This is one formal registered attempt. A durable create-only resume may complete missing cells after infrastructure interruption only when source, inputs, seeds, gates, and existing artifacts are unchanged; it is not a new scientific attempt. Scientific failure does not authorize truncation of M1/M2 evidence.

All outputs and temporary/cache files go to a new create-only directory under `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg`, launched through `bash experiments/lcrseg/scripts/with_nas_storage.sh`; no home fallback is permitted. After the validation report is committed, pushed, and matched to the remote branch SHA, execution stops for independent review. No test evaluation or other follow-up is authorized.

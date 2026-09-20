# MAIN_HEAD_HIERARCHICAL_KL_V1 — preparation review

Status: **STOP_AWAITING_EXTERNAL_CODE_REVIEW**. This is a local, preparation-only remediation, not a production-ready or approved experiment. The attached review requested R1/R2 fixes and did not authorize execution. No approval or launch receipt has been generated.

## Delivered

- P0-A: independently recomputed five recipe means, 60 per-domain/class contrasts, all 48 strata across eight ranking points and both cumulative same-state U summaries from the existing public aggregate files. Reproduce with `python3 experiments/lcrseg/docs/main_head_hierarchical_kl_v1/recompute_p0_a.py`.
- Fixed C0/C1/C2 loss kernels: logsumexp parent/conditional KL decomposition, explicit zero-mass limits, unchanged PAS mask and denominator, detached main-teacher weights, E threshold0.9, maximum attenuation0.5, per-image/predicted-class C1 mass matching. C0 and all-one weights use the original masked_kl path.
- Shared B2 StageTrainer remains the sole update loop. Its default fine_kl delegates unchanged to masked_kl; new adapter overrides only fine_kl. No auxiliary heads, DS, H, replay, new parameter group, projection or gradient-path restriction is added. Frozen historical checkouts remain untouched.
- Checkpoint semantic binding includes a loss contract only for the new adapter; original trainers' metadata schema remains unchanged apart from the existing source-fingerprint mechanism. Arm and source identity are checked before restore mutation.
- Generated CPU tests passed on first attempt: 15 Adam calls, 32 autograd.grad calls, one intentionally failed post-optimizer update counted in15. C0 bitwise two-update equivalence covers student, EMA, Adam, scheduler, scaler, prototypes, telemetry, data counters and RNG. C1/C2 uninterrupted versus restored continuations match; wrong-arm restore and saving/reusing failed state are rejected. Mathematical tests cover float32/64 values/gradients, zero mass, extreme logits, empty masks, weight bounds and detached teacher gradients, and C1 mass matching.
- R1 remediation: `HierarchicalTrainer.for_resume(..., *, arm=...)` is an explicit arm-aware factory; the native/data/CUDA rejection remains in the constructor. A server-side signature check passed without constructing a trainer.
- R2 remediation: CPU attempts and optimizer calls are charged in a study-global, atomically updated POSIX ledger. Missing, corrupt, mismatched or exhausted ledgers stop; changing `TASK_EVIDENCE` cannot reset the quota. The ledger regression passed with zero optimizer calls.

CPU fixture is SyntheticParentBridge, not the native medical segmentation model. Its generated readout bias [-3,1,1], size8 and permissive PAS settings exercise foreground supervision and are not proposed production options. CPU checks do **not** prove native B2 checkpoint equivalence, actual frozen-environment equality, real data semantics, CUDA behavior or scientific benefit. New native model runs, private reads and CUDA calls are all zero.

## Review order

1. Read SCIENTIFIC_REVIEW.md, PLAN.json and BUDGET.md; current CPU budgets and future proposed production budgets are separate.
2. Inspect `main_head_hierarchical_kl_v1/core.py`, `trainer.py`, `tests.py`, and the small shared changes in `five_frameworks_v1/train_stage.py` and `semantics.py`.
3. Inspect CPU/REPORT.json, CPU/COUNTS.json and SOURCE_BINDINGS.json. Tested source hashes are bound, including inherited Python source. No test rerun is necessary merely to open this review.
4. Inspect P0_A_RECOMPUTED.json: it explicitly cannot supply conditional-entropy/fine-PAS ranking or patient confidence intervals from aggregates.
5. Review the proposed P0-B preregistration and sufficiency conventions below before authorizing any private manifest/payload access.

## Remaining gates — not disguised as completion

The adapter **rejects every native model and every non-generated provider**. There is no data loader, P0-B runner, CUDA qualifier or formal CLI for this new study. A later native implementation, including model/state/prototype reconstruction and execution authorization enforcement, must be reviewed under its own actual commit. This preparation does not silently enable the proposed10600-update matrix.

P0-B: exact forward count remains UNBOUND because no private manifest has been read. Existing four B2 state candidates (two stage1 prefixes and two stage2 endpoints) and historical receipt/tensor bindings are imported in PLAN.json; live tensor availability and role have not been independently revalidated here. Do not substitute deployed student for dense EMA or invent historical prototypes. Obtain separate metadata-access authorization first, bind manifest, preprocessing, model-role/prototype construction and finite image/batch list, compute exact forward budget, then obtain appropriate payload execution approval. Missing frozen state is a stop, not permission to reconstruct it by training.

For proposed P0 scoring: retain fine-PAS and E support per image and predicted rim/cup. Evaluate conditional error on GT disc and parent mistakes on all eligible GT classes, including background. Rank within each image/GT-class/boundary stratum with stable pixel-index ties; report actual k/n (ceil(.25n)), captured errors, total errors, random expected captured errors=k*errors/n, and support. For a supported image, divide summed captured errors by its summed conditional errors; use equal-image averages across supported images. No-error images are reported separately. Apply >=1.25x matched random expectation and >=.95x 3-class entropy capture at **both B2 endpoints**. Start states are descriptive. Patients are not inferred from filenames; absence of validated patient grouping means image-balanced only and no patient CI.

PLAN.json adds explicit proposed support conventions (>=5 supported images, >=20 errors per endpoint) and a near-empty E criterion (<0.1% of legal pixels OR <1% of fine-PAS predicted-disc pixels). These are preparation choices for external review, not empirically established thresholds or user-claimed facts. They have seen no new private outcomes. Do not revise them after observing new payloads to rescue a failed gate. Sparse support is INSUFFICIENT_EVIDENCE; negligible E is NO_PRACTICAL_SCOPE. No training follows either.

P1: only if P0 passes and separate real external code approval plus new user execution authorization are present. Four C1/C2 stage2 nodes,10600 formal updates; C0 import conditional on native/environment/data/no-op validation.16 L-only smoke,15 generated CUDA,0 additional VJP proposed, all currently unauthorized/unexecuted. Diagnostic readouts must reuse current tensors at each node's four fixed quarter points without adding model forwards; enforce actual counters in the future runner. C0 fallback5300 and all C3/new-seed/full-trajectory work require separate proposals. Native production scheduling and report/export code are deferred until P0 warrants them, rather than exposing an unapproved runnable path.

## Scientific conclusion retained

OBS0 recovers approximately B2/A1 while reversing most original A5 degradation; it has not established new baseline gains. Existing conditional-disagreement ranking captured fewer errors than 3-class entropy overall; new binary conditional entropy is a different, unvalidated score. Do not claim novel KL identity, stop-gradient or entropy selection. Do not reopen the observer's default production use or alter any historical gate/result.

## Delivery boundary

Publication was previously authorized on branch `codex/main-head-hierarchical-kl-v1`; this remediation is a new local commit. No experiment launch or monitoring is authorized by publication. CPU temporary checkpoints remain on NAS; the local review package includes only generated-test evidence, source changes, preregistration and aggregate public analysis. Original source documents/results remain intact. This is ready for another external code review, not ready for a production launch approval.

# AGMS_OBSERVER_V1 — independent code review request

Status: **STOP_AWAITING_EXTERNAL_CODE_REVIEW**. This is preparation, not an experiment result or an approval. Review the exact commit containing this file. Production requires both a genuine external review bound to that commit and a separate new user launch confirmation. No old review, HALF launch, autonomous-work instruction or this preparation prompt grants production authority.

## Scientific change and implementation

Reference: original A5 (`89dc7657c4f0b7c6f51de339d0f6bc2a621da55f`), DS=.25. Only the auxiliary readout consumes `features[name].detach()` in `agms_observer_v1/model.py`. Main features remain connected. Two 1x1 heads (294 parameters) still learn via .25 times their mean CE+Dice, lr=.001, weight decay=4e-5, EMA=.99. Main L/PAS/KL and H=.5, A/B whitelist, A-only projection, risk lag, thresholds and data geometry are retained. Original auxiliary RNG namespace is intentionally retained.

The new package reuses native parent, main model, StageTrainer orchestration, gradient splitting, kernels, integrity verification, cost sessions and durable physical counters. Study-specific model/trainer/state/admission and finite execution/report interfaces are separate, so old type checks, authority and files remain untouched. `SHADOW` is generated-only and cannot appear in the canonical production nodes.

Source map: `model.py` detached readout; `trainer.py` unchanged scientific objectives plus detached diagnostics and atomic commit; `diagnostics.py` aggregate math; `state.py` exact new-study reconstruction; `authority.py` new review AND user launch; `qualification.py` finite native qualification; `execution.py` two-node controller/prefix and target acceptance; `reporting.py` ten-endpoint report; `tests.py` bounded generated CPU evidence.

## Test evidence and limitations

- Attempt 1: **FAIL, 0 optimizer calls**. The generated negative report fixture changed macro Dice without changing rim/cup; the existing metric consistency guard correctly rejected it. The fixture was corrected; original failure/ledger/session retained.
- Attempt 2: **PASS, 12 optimizer calls**: native head/EMA checks 2 + uninterrupted/resumed continuation 5 + B2/shadow 4 + predefined after-optimizer failure 1. The injected failure is charged; retry of its uncommitted state is rejected.
- Cumulative new CPU: **12/32**, attempts **2/2 used**; no third attempt and no remaining capacity may be used automatically. Prior CPU **235 = 134+80+21**, separate; combined historical/new **247**.
- Original auxiliary forward values are exact before/after detach on generated native features. DS upstream main A/B gradients are None/zero; head gradients are nonzero. Original L/KL/H main paths are checked on a generated route fixture, without asserting every actual coarse mask is nonempty.
- H-disabled generated shadow: main model, dense main EMA, main A/B optimizer state and schedule, prototypes, reads, RNG and backbone-forward telemetry match B2 exactly; observer heads learn. A teacher-L readout on cached features is expected additional work and is explicitly checked separately from backbone forwards.
- Native continuation matches all tracked state; wrong study/prefix/non-detach semantics (even rehashed) are rejected. Deployment keeps merged main only. Actual generated checkpoint integrity and wrong-hash/lost-tail refusal are tested.
- Zero-update checks cover KL parent/conditional identity (extreme and empty/masked support), gate counts/partitions, diagnostic RNG purity, original authority refusal and report 10/30/12/8 coverage. Native checks additionally verify diagnostic grad/risk state purity.
- CPU cost session: 60 native feature calls, 54 native readouts, 52 autograd calls, 12 optimizer calls; zero CUDA allocation. Generated model constructors read reference source only, not real model weights or patients.

See `CPU/TEST_REPORT.json`, `CPU/ATTEMPTS.json`, both attempt ledgers and closed cost sessions. All generated tensors/checkpoints remain outside Git on NAS. CPU evidence binds the exact code tree, plan and execution plan; documentation can be committed after qualification without changing those source semantics.

## Frozen future proposal — not executed

OBS0 seed163: O1 original B2 RIM prefix to Drishti (2100); O2 original B2 Drishti prefix to RIM (3200). Only two new second-target nodes, total5300. B2/A1/original A5/HALF have two historical endpoints each (8 imports); no historical model reevaluation. Prefix starts never come from A5/HALF final models.

CUDA10 = continuation5 + B2/shadow4 + predefined failure1. Smoke8 = 4 L-only updates per order, discarded. Formal5300; real total5308. Fixed8 diagnostic points ×4 VJP =32. New source, stage1, P0, extra seed, extra coefficient and other arms: zero. All production phases **PENDING**, new CUDA0, real optimizer0, patient reads0.

The future report has 10 endpoints,30 domain rows,12 primary paired rows and8 new gradient points. Historical new diagnostics are unavailable, explicitly NA. Recovery requires both orders' Final >=B2-.001 and Incoming >=B2-.005, and **each individual old domain** macro Dice >=B2-.002. This is descriptive recovery, not statistical noninferiority, mechanism success or permission to continue.

## Evidence anchors and current conclusion

HALF results `6e65a9b987f8818f7b10fa11a2cc205d11283779`; HALF implementation `44c1da8a5a8a75021a0298d42bd8f9088e9b059a`; original AGMS results `ef6888dc81519ce9e9ca13ea497bbf6397747d70`; original AGMS implementation `89dc7657c4f0b7c6f51de339d0f6bc2a621da55f`; B2 prefixes `3034b2199aa7d6e54ea67499f391a0dd4ea3d21e`.

Read-only recomputation confirms HALF minus original A5 mean source +.00967495698, first target -.00146987360. Thus old-domain average recovery is driven by source, not consistent first-target improvement. HALF's original `DS_PRESSURE_SUPPORTED=false` remains unchanged. No OBS0 improvement is measured or claimed.

Some inherited `PLAN.source` entries identify older provenance of the immutable proposal; they are not initialization sources or authorized runs. Actual starts and all imported endpoints are exclusively `PREFIX_BINDINGS.json` and `IMPORT_BINDINGS.json`.

Review priorities: detached auxiliary-only graph; shadow main-state equality; all-step U counts and no-label boundary; current-L ranking limitations; exact resume/cost accounting; independent old-domain gate; new authority binding. No `APPROVED` artifact is supplied. Stop after this delivery until genuine new review and user confirmation.

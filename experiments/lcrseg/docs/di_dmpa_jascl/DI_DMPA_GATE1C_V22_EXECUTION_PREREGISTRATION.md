# Gate1C v2.2: prospective full-diagnostic execution and retention

Registered 2026-08-31 Asia/Shanghai. The binding machine contract is
[DI_DMPA_GATE1C_V22_EXECUTION_PREREGISTRATION.json](DI_DMPA_GATE1C_V22_EXECUTION_PREREGISTRATION.json).
**This registration is not launch readiness.** A separate execution
authorization, successful live source/cache/input/resource checks, published
exact code, clean tests and new exact-code real integration must precede the
formal run. At registration the old SSH endpoint is unavailable and cache reuse
is **not approved**. Never change these historical readiness fields after the fact;
later readiness belongs in separate receipts.

## Scope and immutable interpretation

Only the current JASCL / DI-DMPA medical-segmentation line is in scope, under
the user's [current-method-only long-running authority](LONG_RUNNING_REPRODUCTION_20260830.md),
commit `b6d70699599bd89faafb4a1dd22223575c62bbb6`. B0/C0 are controls, not
additional methods. This is offline Gate1C evidence collection, not training.

Keep the original C science registration `32d32ab` and the v2.1 input amendment
`9d8ecc6`: B0 EMA probability/features, student gradient receiver, Fundus
REFUGE -> RIM_ONE_r3 -> Drishti_GS, K=2 original train-labeled fits and identity
history. R4 remains unavailable. All 72 pairs, case IDs, forward seeds, eight
teacher draws, labels/splits, preprocessing, null rules, R0-R3/PAS/PoE equations,
metrics and C1-C8 thresholds are inherited exactly by their file hashes.
The model remains the frozen medical UNet adaptation with the official
stochastic 3x3 classifier, not a claim of matched DeepLab paper performance.

The sole reconstructed legacy PAS input remains
`RECONSTRUCTION_SUPPORTED_NOT_HISTORICAL_HASH_VERIFIED`; no historical bank
hash exists. The 400 earlier baseline recovery updates are separate history.
No original checkpoint is replaced. Gate1B and original overall Gate1 remain
`FAIL_TRANSPORT_NOT_SUPPORTED`; original C v2.1 remains
`BLOCKED_INCOMPLETE_EVIDENCE`. A new result never erases either failure.

## Numerical implementation boundary

The three-pair [precision pilot](GATE1C_V22_PRECISION_PILOT_REPORT.md) passed
under preregistration `6357317`, exact code `7fdd431`, with an independently
verified private archive. Its later operator-refusal repair `d6bd070` passed
135/135 exact-code tests without repeating the pilot.

Keep `execution.py`, `gradients.py` and `precision.py` byte-identical to that
tested numerical code. Only shared-runner orchestration, explicit reused-cache
provenance, versioned reporting, guards and tests may change. Existing pilot
observer logic may be extracted into shared helpers without changing its math.
Do not add a second training loop, dependency or new loss. The native validation
path and reliability/metric formulas stay unchanged; report-compiler changes are
limited to version/provenance presentation, not candidate conditions or selection.

Retain native FP32 scoring and original Gaussian returns. An isolated FP64
student replays those exact draws for labeled/unlabeled VJPs. Targets and class
strata are first canonicalized through the original FP32 rule. Native student,
EMA, PAS, banks, features, class strata and RNG must not change.

Every global objective and supervised control must satisfy relative L2 <=1e-3
and cosine >=0.9999, with FP64 norm as the relative-error denominator. Keep
component-sum `atol=1e-6, rtol=1e-4` on every block. Two exact-zero gradients
agree only as a numerical control; their scientific cosine remains null.
One-zero-only comparison fails. No averaging, fallback or relaxed tolerance may
rescue a failed comparison. Numerical validation precedes scientific selection.

## Conditional native-validation cache reuse

The only proposed source is the complete native validation portion of the
original v2.1 formal attempt at:

```text
/root/LCRSeg/runs/di_dmpa_gate1c_v21/9d8ecc65730bee5bec46a1f098c9fe96a67a59b9/gate1c_v21_44a25254697fa535d2b48b64e27ecb226436f7d0_attempt1
```

Its immutable artifact manifest SHA256 is
`0d652551711e0a3ceff6ac8bdb0001355f4ec6083882460d740784ee837420d9`.
The published index lists **495 NPZ / 4,856,574,421 bytes**, nine validation
units and 72,990,720 pixels. Per seed the domain case counts are 100/40/25.
This represents **990 historical native validation forwards**, not new v2.2
computation. New validation-forward budget is zero.

The [local source preflight](gate1c_v22_preflight/SOURCE_AUDIT.json) proves
unchanged validation AST, unchanged protected scientific sources, unchanged
numeric files and exact published case inventory. It does **not** certify
current remote files. Before reuse, a new read-only audit must:

1. Rehash every referenced file and check byte length, metadata, exact case/seed
   identity, image/split/manifest provenance, all nine checkpoint and bank hashes,
   the recovered PAS proof and all nine original model guards.
2. Read all 495 caches through the existing `read_arrays`/`validate_scores`:
   validate array hashes, shapes/dtypes, finite or structural-null scores,
   support census and unchanged field set. Validate the known cached real null
   at seed2/stage0/REFUGE_train_n0038, coordinate [185,180].
3. Preserve original source metadata and descriptors verbatim. A new derived
   index may add new-run metadata only alongside explicit source path/hash and
   original metadata. Old model guards and forwards must never be presented as
   newly executed. Any new index must match its source content apart from these
   registered provenance additions.
4. Recheck all current-role input identities and label isolation. Builder APIs
   never receive GT. Validation labels remain evaluator-only; unlabeled hidden
   GT and final-test GT remain unused. Dataset filenames containing `test` are
   not a substitute for the frozen manifest's role assignment.
5. Check source/input/cache hashes before integration, before formal phases and
   after completion. Stop on any mismatch; no automatic replacement/recompute
   or alternative cache is authorized by this version.

Recompute all nine reliability metric units and all nine PoE evaluator units
from the verified caches with the unchanged evaluator. Old metric rows may be
used for exact reproducibility comparisons, never substituted as new admission
inputs. Do not reuse old partial gradient rows, old v2 partial caches, pilot
rows, failure markers or STOP files as new run state.

## Exact-code integration and full execution

First publish the new preregistration, finish a live metadata/file-hash/resource
precheck without loading checkpoint tensors or running forwards, and publish a
separate conditional execution authorization. Implement the minimum runner
changes and publish exact code. Its clean synthetic suite must include all
existing checks (at least 138 after the three source-audit tests), all new
provenance/counter/budget/refusal checks, and zero failures/errors/skips. Only
after authorization/code publication and those tests may the exact-code input
payload and all-cache numeric audits load tensors/arrays; these audits still
perform zero model forwards.

Then run a new create-only real integration using the same three fixed pilot
pairs and all four phases. GPU0 receives seed0/stage0/pair00 and
seed1/stage1/pair00; GPU1 receives seed0/stage2/pair01. Exact four case IDs and
all seeds are in the JSON. These are **new** integration computations, not a
relabeling of old pilot results. Require old native/FP64 hash goldens, exact PAS,
every numeric check, all 12 guards, source/cache checks including the cached
real null, and no parameter/bank/RNG mutation. Budget: **75 real forwards**,
276 native +366 FP64 autograd calls, at most ten minutes per worker/phase.

Only a passing integration allows a new full attempt. Use the original frozen
72-pair list index modulo two, preserving ascending pair order within shards:
**36 pairs per GPU**, with two workers and batch sizes unchanged. For each unit
the even/odd pair indices go to GPU0/GPU1. This is a preregistered scheduling
choice, not a data selection. Shared cached forwards remain identical across
candidates and phases.

Order: verified cache reference indexes -> fresh nine-unit reliability metrics
and barrier -> draw0 -> eight-draw noise -> posterior mean -> PoE with fresh
nine-unit PoE evaluation -> complete audits -> unchanged C compiler. Each phase
requires all 72 pairs, every counter and every numerical/isolation receipt.
The PoE evaluator may overlap the independent PoE GPU workers; no earlier phase
may be skipped. Both GPUs use one CPU thread per worker; CPU metric workers=2.

| Full-run evidence | Required count |
| --- | ---: |
| New native FP32 forwards | 1,224 |
| New FP64 shadow forwards | 576 |
| Native / FP64 autograd calls | 6,624 / 8,784 |
| Pair-phase records | 288 |
| Global rows and precision comparisons | 6,912 |
| Blockwise alignment rows | 41,472 |
| Class-component rows | 15,120 |
| Supervised global precision comparisons | 288 |
| Teacher draw records | 576 |
| New probe / reused validation guards | 288 / 9 |
| Native PAS scoring checks | 720 |

Full forward budget is **1,800**; integration plus full is **1,875**. The
297-guard full evidence set must explicitly separate 288 new from nine reused
guards. Do not count the integration guards as formal-probe guards.

One integration attempt and one formal attempt are allowed. Full elapsed budget
is three hours, GPU worker/phase budget 30 minutes, CPU metric-phase budget one
hour; retain the existing 600-second owned-worker shutdown grace. Checkpoint
state is read-only. Interrupted observation is not process failure: inspect
the original process handle before any action. Actual failures retain their
logs/partial artifacts and require a new prospective repair/attempt decision.

## Storage and private retention

Use only the existing `/root/LCRSeg` volume and environment; do not use `/tmp`,
shm or overlay as new experiment storage in this version. Create-only paths:

```text
/root/LCRSeg/runs/di_dmpa_gate1c_v22/<preregistration-commit>/integration_attempt1
/root/LCRSeg/runs/di_dmpa_gate1c_v22/<preregistration-commit>/attempt1
```

Require at least **8 GiB free** at preparation. Cap new formal artifacts at
6 GiB and integration at 512 MiB; preserve a 2 GiB root reserve and an extra
64 MiB per-pair headroom. Check before every pair/phase. The raw full gradient
payload bound is 3,906,994,176 bytes, plus JSON/container overhead. Referencing
the old 4.86 GB native cache avoids duplicating it into the new remote run.
Any capacity failure stops with evidence retained; no deletion, truncation,
unregistered spillover or batch change.

Private local archive root:
`/Users/bominwang/Desktop/codes/SSL_CL_seg/runs/di_dmpa_gate1c_v22`.
Verify the existing ignore rule and at least 16 GiB local headroom before
transfer. Retain new outputs plus the referenced original validation caches,
units, guards and input metadata needed to interpret them. Keep original runtime
paths unchanged; a separate receipt maps remote roots to local copies. Use an
incoming directory and independent full hash/size verification before promotion.
A partial transfer is never a verified backup; never overwrite verified copies.
Remote and local readiness must be reported separately.

Publish only code, registrations, scrubbed diagnostics/tables, tests and
manifests, not tensors/checkpoints/images/masks or sensitive logs. No new paid
resource, environment, mount, permission, proxy or global network change.

## Decision and reporting

All original C1-C8 conditions use unrounded values and complete 18-unit /72-pair
denominators. Primary is R3 pixel-normalized vs original R1 pixel-normalized.
Only independently passing R3 class-balanced vs that same pixel R1 reference
may be selected after primary failure. Zero/undefined required comparisons
cannot earn admission; R0/R2/PoE/posterior-mean/C0/student cannot rescue R3.

Include original and new registration/authorization/code hashes in every run
metadata object, source-cache provenance, exact commands, current test evidence,
new/reused counters, all reliability/gradient/noise/PoE outputs, failure notes,
complete reference/new-output manifests and private-archive audits. Publish
explicit `GATE1C_V22_STATUS.json` and `GATE1C_V22_FINAL_REPORT.md`; compatible V2
filenames must identify v2.2 execution and v2.1 inputs without rewriting old files.

No optimizer, backward, parameter.grad, EMA/GAS/prototype/transport update,
method registration, training, Gate2, Prostate/MnMS, full sweep or main merge.
Publish the complete scientific result, analyze it and state the next finite
current-method-only step under the existing user authority. This protocol does
not promise a passing gate or successful method reproduction.

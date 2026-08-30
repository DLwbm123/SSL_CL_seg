# Gate1C v2.2 diagnostic-precision repair: finite pilot preregistration

Registered 2026-08-30T17:02:58.339Z. Binding machine contract:
[DI_DMPA_GATE1C_V22_PRECISION_PREREGISTRATION.json](DI_DMPA_GATE1C_V22_PRECISION_PREREGISTRATION.json).

## Scope and hypothesis

Only the current method's offline diagnostic is in scope. The previous
same-draw FP64 reference (report `ee0b9df`, code `d87b7cb`) supports float32
network VJP accumulation as the cause of the component-sum discrepancy.
This prospective numeric version tests the repair across all three stages and
all four existing probe phases. It is **not a full Gate1C run**, scientific
admission, segmentation training or a revision of any historical failure.

Keep the v2.1 input contract, B0-EMA, K=2, identity history, original cases/seeds,
losses, PAS, R0-R3/PoE formulas, class strata, null rules, thresholds and C1-C8.
Gate1B/overall Gate1 remain FAIL_TRANSPORT_NOT_SUPPORTED. Original Gate1C v2.1
remains BLOCKED_INCOMPLETE_EVIDENCE. Registration/training flags stay false.

## Minimal registered changes

- Keep native float32 forwards and their original grad modes. Capture each
  student's original Gaussian draw without changing its return or RNG.
- Use an isolated frozen-student copy in float64 for labeled and unlabeled
  VJPs; replay the corresponding captured float32 Gaussian values cast to
  float64. Original teacher/student/banks and native scoring do not change.
- Use unchanged supervised and consistency losses. Every gradient target is
  first canonicalized using its original float32 conversion; only then cast
  for FP64 differentiation. PoE reporting remains unchanged. No target or
  predicted-class change is hidden inside the precision repair.
- Compute a native float32 total-gradient control for every objective and
  supervised reference. These are engineering comparisons, not admission
  fallbacks. The unchanged native mode still has its original decomposition
  guard; its known failure is not relabeled as success.
- Reuse the current executor and gradient functions. Share the tested replay,
  RNG and comparison helpers instead of copying them. Add exception-path
  isolation receipts. Do not change training/modeling, reliability, metrics,
  reporting, old contracts or scientific conditions.

## Fixed pilot and finite budget

| Pair | Coverage reason | Physical GPU |
| --- | --- | ---: |
| B0/seed0/stage0/REFUGE/pair00 | Original stage0 integration and empty history | 0 |
| B0/seed1/stage1/RIM_ONE_r3/pair00 | Original recovered-input integration | 0 |
| B0/seed0/stage2/Drishti_GS/pair01 | Sole native decomposition failure | 1 |

The JSON includes the exact original four case IDs and all forward/draw seeds
for each pair. No pair replacement, seed selection or scientific-score-based
selection. Run **draw0 -> noise -> posterior -> PoE**, with all three pairs and
their audits complete before the next phase starts. No new validation forward
or partial scientific-admission calculation is authorized by this pilot.

Bounds: **75 real forwards** (51 native, 24 shadow), **366 FP64** and **276
native-control autograd calls**, 2 workers, at most 10 minutes per worker/phase.
Expected evidence: 12 pair-phase records, 2,016 alignment rows, 288 global
objective precision comparisons, 12 supervised global comparisons, 630 class
decomposition rows and 12 model immutability guards. Draw0 entries reused in
the noise phase are identified as reuse, not additional forward/gradient calls.

Output: `/root/LCRSeg/runs/gate1c_v22_precision_pilot/<preregistration-commit>/attempt1`,
create-only. Require >=1 GiB root headroom, pilot artifact budget 512 MiB.
Raw arrays can be archived with verified hashes only under the existing
git-ignored local root `runs/gate1c_v22_precision_pilot`; never public docs.
The separate `/tmp` mount is not used by this small pilot. No new dependency,
paid resource, storage reconfiguration, deletion or full retry.

## Acceptance fixed before implementation

All FP64 component-sum checks keep **atol=1e-6, rtol=1e-4**, for every block and
registered component. For every global objective and supervised total gradient,
require native/reference relative L2 <=1e-3 and cosine >=0.9999; denominator is
the FP64 norm. Both exact-zero gradients agree as a precision control, while
their alignment cosine stays null and is never credited as a scientific
improvement. One-zero-only comparison fails. No averaging or candidate rescue.

Native forward, supervised-gradient and eight draw0 total-gradient hashes must
match the two bound original golden receipts. The failed pair must reproduce
its available native forensic hashes, and its FP64 draw0 R2/class-balanced
global hash must match the separately published reference. Require native R1
pixelwise parity with the unchanged Gate0 PAS objective on every scoring call,
all source/shadow/GAS/bank/gradient/RNG guards, complete coverage and no GT leak.

Prerequisite synthetic tests include full-phase default-native parity against
the frozen executor/gradient source, same-Gaussian replay, shadow isolation,
target-stratum preservation (including float32 ties), zero-gradient behavior,
FP64 decomposition and exception-path isolation. Publish and verify exact code
before any real pilot forward.

Only complete passing engineering evidence yields PASS_NUMERIC_PRECISION_PILOT.
Preserve incomplete evidence and all numeric/protocol failures. This three-pair
pilot cannot yield a scientific Gate1C pass or method-reproduction success.
Publish results and record the next finite plan; a full diagnostic requires a
separate prospective execution and artifact-retention authorization.

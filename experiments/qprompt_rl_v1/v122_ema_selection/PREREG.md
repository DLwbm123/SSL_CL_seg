# V122: independent EMA conflict selection and prototype ranking

User explicitly authorized continuing after reviewing the proposed independent EMA-only control. Single mechanism question: do deployment-visible EMA/mixed-target conflicts support useful corrections with independently specified locations/budgets, and does prototype ranking add reliable value at the same budget? This is the next development diagnostic; no claim of model-level holdout or training benefit.

## Frozen conditions and targets

Reuse all8 V116 recovery seed601 ENTRY100 states, same40canonical current A_fit image/context exposures, existing photo conditions, seed119 computations and fixed cyclic other-image support as V119–V121. Models already fitted the target images; they are excluded only from their own prototype construction. n=2 has1support image,n=8 has2. No sample/condition/seed/checkpoint search, new domains/patients or model training. Reused development transition choices1→0 and2→1 remain explicitly outcome-informed by V120; current preregistration does not make the data independent.

M is the existing original-EMA>.7, half-eligible COVERAGE selected set. Let c0 be original action11 EMA/memory target argmax and e be EMA argmax. Define CE=M AND[(c0=1 AND e=0)OR(c0=2 AND e=1)]. This candidate and all EMA-only masks are computed by ema_masks(base,q,selected,seed), which has NO prototype or label argument. Its behavior must remain identical when arbitrary prototype scores change.

Every selected correction uses the same existing probability exchange toward e; outside that correction mask the original target remains unchanged. Original per-pixel probability multisets, original EMA weights[.5,1,1.5] and original M remain exact. This restores an EMA winning class, not its complete soft probability vector. Actual changed masks must equal intended masks, otherwise fail closed.

## Five arms and independent budget

BASE: no correction.
EMA_ALL: correct every CE pixel; completely independent of prototype locations and count.
EMA_HALF: within each target image and each of the two CE transition strata, K=floor(number of CE pixels/2). Rank by EMA winning probability minus its second highest probability, descending. Use exact K; ties resolve by increasing flattened pixel index.
RANDOM_HALF: same independently defined K per image/transition; one uniform randperm with CPU generator seed122000+100*context+image_index, strata processed1→0 then2→1. No redraw or alternative seed.
PROTO_HALF: same K and CE, rank by cosine similarity to the current-label prototype of e minus the largest cosine similarity to either other prototype. Prototypes use the same frozen cross-image normalized feature means as prior diagnostics. Score changes ONLY ranking, never count, CE eligibility or destination. It may select negative-margin entries when K requires them; no adaptive rejection/threshold. Ties use the same flattened index order. All3support classes and norms>1e-8 are required by this fixed diagnostic; violation stops rather than changing fallback/denominator.

The50% quota is a single predetermined constant, not fitted to previous prototype counts or searched. K=0 stays zero; no minimum-one intervention. Three HALF arms exactly match counts, transition destinations and original EMA weight sums. They need not match continuous probability amplitudes; publish weighted target L1 differences and overlaps. EMA_ALL intentionally uses its own full coverage and is not a budget-matched ranking comparison. No new U-loss weight or loss term.

## Frozen readout and decision

40metric rows(8contexts×5arms) and8control summaries. Existing additive target argmax accuracy and true-label NLL(clamp1e-8), whole-M denominator, equal-context means. Report all6pre-registered comparisons: EMA_ALL−BASE(primary independent rule), EMA_HALF−BASE, EMA_HALF−RANDOM_HALF, PROTO_HALF−BASE, PROTO_HALF−EMA_HALF, PROTO_HALF−RANDOM_HALF.

Each sign comparison requires equal-context weighted accuracy delta>0 and NLL delta<0, both jointly in≥6/8contexts, and both directions in each n=2/n=8/source2000/source8000 subgroup. Ties/zero effects fail strict positivity and remain in the8context denominator. Primary status DEVELOPMENT_EMA_CONFLICT_SIGNAL iff EMA_ALL−BASE passes; otherwise NO_RELIABLE_EMA_ALL_CONFLICT_SIGNAL. Record EMA_HALF quality and its ranking increment separately; neither can silently substitute for the primary full-rule result. Prototype ranking increment requires allthree PROTO_HALF comparisons pass; no requirement to propose a different class from EMA. Publish all positive/negative axes, not a selected winning method. No permutation significance claims or treating pixels as independent.

Summaries include both CE transition counts and independently calculated K, each arm's actual changes/changed class-weight/L1, EMA/prototype and random/prototype overlaps. BASE rows must exactly reproduce V121. No previous prototype selection counts enter masks or budgets. This fixes V121's inherited-budget limitation while preserving historical failure judgments.

Any result is development target quality on model-seen images, not independent confirmation, pseudo-label quality on U, student Dice or actual continual-training gain. A future model-level holdout must exclude targets from relevant fitting before model creation; n=2 leave-one-fit is1labeled training image and must be reported as such. Do not relabel ENTRY100 diagnostics as model-unseen, open Q_dev/test/hidden U labels, or jump to fullGRPO. Preserve history-free trained-parameter memory and transient current-domain prototypes.

## Costs and operation

Exactly40newcurrent-labeled image/context reads/evaluations,160model-image-forward attempt/success pairs(40eachEMA/memory/flippedmemory/student),40prototype assignments; all5arms share this inference and source feature extraction. Zero native/actor optimizer,linear solve,Q_train/Q_dev,U-image,newannotation calls. V119–V122combined640segforward160labeled exposures, all separately charged. Cumulative841061native excluding8000source(849061including),181593actor,57solves, all prior failures and512/146CPUactor-forward diagnostics unchanged. This shared diagnostic does not measure standalone latency; a deployed EMA-only selector needs no prototype/support-label calculations, while PROTO_HALF does. Do not present matched diagnostic forwarding as equal deployment compute.

Fresh create-onlyNASrun; verify V121rootEXIT0/completion/final+posthocpublication/free lock and no overlapping project EXEC_RUN. GPU4admission>=12GB,NASmount/capacity/write-readprobe,existingNASwrapper,neutralargv/ps/nvidia checks. Publish source/protocol before model operations. CPUsynthetic selfcheck0model/image/query/optimization for independent masks, exact budgets/weights/ties/empty sets/strict gates. Once-only audit160forwardpairs40supportexclusions40rows8controlsummary/reference/state/readout. No implicit retry/extra seed/fraction/split/next experiment. Publish source/protocol,allanonymousaggregates/decision/costs/audit/report with proxyGitHub push/remote/anonymous verification and NAScommitted-file receipt. Per-image data/metrics/IDs/features/weights/rawlogs remainprivate.

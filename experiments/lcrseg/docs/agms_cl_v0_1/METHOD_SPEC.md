# Implementation contract and operation correspondence

Scientific source: the unchanged `inputs/01_EXPERIMENT_PLAN.md`,
`inputs/02_IMPLEMENTATION_SPEC.md`, and `inputs/05_PROPOSED_PLAN.json`.
The canonical resolver binds the complete B2 options, two original B2 prefix identities,
two historical imports, environment, dataset hashes, geometry, risk, and all ten nodes.
Recomputed attacker-supplied hashes cannot authorize different semantics.

## Native model and loss

`AGMSModel` retains native LR_SRC_A (14 layers, rank8), linear valid3×3 main readout,
A-only *effective increment* right projection, the original main CE+Dice and B2 fine PAS/KL.
No R/F, SWD, old teacher, or historical feature memory is added. A0 physically instantiates
no new heads/loss/state updates and serves as the all-new-mechanisms-off equivalence path.

A2–A5 attach two bias-enabled 1×1 heads to decoder.dec3 (64 channels) and dec2 (32),
294 parameters total. Heads use an arm-independent namespaced CPU generator under
`fork_rng`; initialization does not seed CUDA or alter the base CPU stream. Their output
is bilinearly mapped to output coordinates with align_corners=True, without the main
valid-readout crop. Each original LCTX forward captures both scales. The exact original
complementary mask collects the anchor's auxiliary log probabilities. Both heads use
native `supervised_parts` (CE+Dice), average across heads, coefficient .25.

One Adam receives the original A/B groups plus the explicit auxiliary group at lr=.001,
weight_decay=4e-5 and the original polynomial schedule. Only A/B belong to the U whitelist.
Current teacher heads are .99 EMA; the frozen main classifier stays frozen.

H uses `-logsumexp(logp_U[:,1:3])`, only legal geometry outside unchanged fine PAS.
A1/A3 require main disc≥.9; A4/A5 also require fused disc≥.9 and weighted variance≤.01.
Each image divides by its legal geometry count, then valid images receive equal weight.
No coarse/valid support returns a graph-connected zero. No hard rim/cup pseudo-label is made.

A4 is uniform; A5 alpha=.1+.7*softmax(-R/.1). R starts .25, beta=.9, and observes current-L
background/disc balanced Brier only. The current step uses previous committed R. Invalid
L does not update it. All predictions, masks, R and alpha used as targets/weights are detached.
This is training selection state, not independently calibrated uncertainty.

## B2 operation correspondence

| Operation | Old B2 | AGMS |
|---|---|---|
| Stage entry, prototype initialization, sampling | original | inherited, same exposure/RNG |
| Warmup L backbone | one | same one, capture auxiliary features |
| Active LCTX backbone | two complementary | same two, same mask, extra heads only |
| Teacher clean L | prototype estimation | same backbone; main/aux readouts added |
| Teacher clean U | one | same one; auxiliary readouts added |
| Student UL | two complementary | same two, main only |
| Gradient split | L allowed parameters + U whitelist | same split, new L heads excluded from U |
| Adam / constraint / scheduler | one update | one update; explicit third parameter group |
| Commit | validate, EMA, prototypes, cursor | validate heads/R too, then commit heads/R at same success boundary |
| Exit | merged main network | inherited deploy; heads/R/optimizer/prototypes discarded |

No new full-backbone forward is needed. Readouts and four diagnostic VJPs are separately
counted; semantic subsets overlap the operation recorder and must not be added twice.
A0 generated tests compare main state, EMA, Adam, schedule, prototypes, reads and RNG exactly.
This does not assert cross-machine numerical identity; the production environment gate remains.

`AGMSTrainer._update` is a narrow copy of the inherited B2 transaction body because it has
no auxiliary EMA/pending-state hook. It adds only candidate auxiliary EMA/R validation and
success-boundary commit. The old class and global type/authority behavior are not patched.
This is an explicit local transaction adapter, not a second independent scientific engine.

Failures retain physical calls and mark the in-memory trainer unusable. Risk, diagnostics
and successful cursor do not commit after a failed optimizer attempt. Recovery is only from
its own semantically bound checkpoint with exact physical cursor agreement. Partial tails
and failed runtime sessions stop; no automatic replay or budget borrowing is implemented.

Resume and stage exit are distinct. Checkpoints bind study, arm, seed/order/stage, prefix,
complete options, hierarchy/geometry/heads/risk, provider and randomness. They restore the
main/aux student and EMA, Adam/schedule/scaler, prototypes/support, R/count, cursor, telemetry
and all diagnostic points. Deployment keeps only the original main network.

## Diagnostics and analysis

40 fixed points, four VJPs each (160), no Adam counterfactual copies. Record raw/weighted
loss, AB/upstream/head gradient norms and cosines, coarse/fine/ignore counts and fractions,
per-scale selected disc probabilities, variance, lagged risk/alpha, current-L shadow
selection errors, missing support and uniform/risk disagreement. Absent losses are marked
not_applicable. Training-L labels evaluate a completed selector and cannot change its mask.

P0 is separately gated: fixed current-L images only, at most16/order (actual bound counts
10+16=26), no optimizer, confidence-only max(q)<.6 & disc≥.9, not current-stage PAS.
Five-pixel Euclidean boundary bands are diagnostics only; no patient-level records are public.

Reports export all twelve final trajectories,36 domains,27 paired rows and40 diagnostic
points. Nine fixed contrasts include the interaction; best simple control uses two-order
mean, never a per-order splice. P_perf/P_joint are development rules, not significance or
journal-acceptance standards. Incomplete scope is NOT_ASSESSED_REDUCED_SCOPE.

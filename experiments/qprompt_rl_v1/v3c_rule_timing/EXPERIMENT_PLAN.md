# V3C-RULE-TIMING — frozen execution protocol

Authorized by the user on 2026-10-01: “好的，那就按照你的想法继续设计实验并执行”. This authorizes this finite four-arm mechanism study, including three new source models. It does not reopen V3B, launch a full CL sequence, or authorize an automatic next round.

## Question and design

Does the V3B EFFECT_RULE improve fixed-budget current-domain Dice over the designated native reference, and does its state-action correspondence contribute beyond its phase-specific intervention frequency?

- Reference: the same audited LR_SRC_A / NativeLRParent and one LCRSegUNet2DJASCL backbone. Original KI historical identity remains unverified. No substitution or renewed identity question.
- Optimization seeds: 165, 166, 167, fixed numerically before outcomes; distinct from V3B development161/pilot162/163 and existing native164. These are new executions, not independent patients and not a claim of globally unused integer seeds. An earlier unrelated proposal mentioned these integers without executable nodes.
- Train each source from initialization for exactly 8000 REFUGE updates using the existing native source_task. Never borrow a different seed's source. Three source models cost 24000 student updates. Source validation is deferred; it cannot select a source checkpoint.
- Two separate stage-1 targets: RIM_ONE_r3 and Drishti_GS, from the matching immutable source. Same source/domain/seed entry state for all arms.
- Arms: ORIGINAL, FINE, EFFECT_RULE, PHASE_SHUFFLE. No learned controller, development reward, reward tuning, added labels, or new losses.
- H=1200 fixed, 24 target trajectories, 28800 target updates. No shortening or extension based on throughput or score.
- Preserve V3B training engine, rank8, trainable A/B, frozen head, A-side constraint, native Adam with coupled decay4e-5, effective initial LR0.0005, original polynomial horizons3200/2100, EMA0.99, U weight0.5, confidence0.7 and valid-pixel denominator. No learning-rate rescaling to H.
- ORIGINAL is native L-only. Other arms choose one U action per five-step block, followed by four FINE updates. SKIP still performs the supervised optimizer update.

## Matched permutation control

Fixed phase edges are RIM [0,300,600,640,900,1200] and Drishti [0,300,420,600,900,1200]. The native supervised LCTX switch is a boundary, so action counts never move across that regime change.

For each seed/domain/phase, advance ORIGINAL, FINE and EFFECT_RULE in five-step round-robin blocks. Rule uses its current student's full read-only update-effect extraction and the unchanged V3B deterministic score/tie rule. After the phase, save the complete rule action list and a single deterministic uniform permutation of its indices. Save this receipt before PHASE_SHUFFLE takes any update in that phase. Then PHASE_SHUFFLE advances from its own saved preceding state using the permuted list. All four trajectories catch up at each phase boundary.

The permutation RNG is keyed by the protocol/seed/domain/phase and isolated from student RNG. Do not resample an identity permutation, force an action difference, tune phase edges, or choose a shuffle after seeing results. Report constant-action phases, unchanged positions and exact action counts. Never extract control features for PHASE_SHUFFLE. Its data/augmentations, optimizer, EMA and student are its own; it receives only action IDs from the paired rule.

This is an offline, paired mechanism control with access to the rule's within-phase action multiset, including future rule actions in that phase. It is not a deployable online baseline. A difference tests temporal/state alignment conditional on these phase counts; one permutation per cell does not establish broad optimality or estimate all randomization variance. Retained updates and labels are matched, FLOPs are not.

## Binding, checks, resources and stopping

Code commit, native upstream, immutable source receipts, data manifest/split, SESSION, RUN_LOCK and launch process identity are bound before target training. NAS only; no raw artifacts in Git. Training uses existing stateless streams and full state snapshots.

New orchestration has one runnable CPU check for exact action counts, deterministic permutation, RNG isolation and phase coverage. Reuse native Adam preview and V3B CUDA integration checks for actual optimizer parity, feature immutability, full-state resume and snapshot ownership; measure four five-step blocks per domain. The unchanged training engine's existing qualification remains applicable; no new scientific qualification sweep.

Caps: source24000, main28800, smoke80 (planned50), development0, controller0. Base maximum52880. Recovery reserve2644 (5%); no automatic retries, no conversions to extra cells/seeds. Failed and replayed calls remain charged. An identical engineering error occurring twice stops the affected path. Partial work is never silently rerun; recovery requires original/new code and complete state binding under the same caps and deadlines.

New window starts 2026-10-01T03:25:00Z (the first execution preparation), optimizer cutoff 17:25:00Z, hard deadline19:25:00Z. This is a newly authorized window; V3B deadlines are unchanged. Every real optimizer call checks the cutoff. Guardian enforces hard deadline, never restarts. Save complete target states at least every100 steps and at phase boundaries; retain deployment checkpoints0/300/600/1200. Source checkpoints additionally include RNG state. Early completion ends execution. No automatic next experiment.

Use GPU4 if available with sufficient memory; one finite serial worker. Estimate runtime from source and target progress; do not infer convergence or change H. Peak memory, wall time, source, smoke, main, failures/replays, feature extraction/VJP/preview, and operation counts are reported separately.

## Endpoints and interpretation, fixed before scores

Use the existing isolated evaluator only after all target training. Fixed1200 final merged student with deterministic native readout, never best/EMA selection. Report case-mean rim/cup Macro Dice with domains and seeds equally weighted, class components, and REFUGE entry-to-endpoint change. Existing validation population is development evidence, not a new patient test set. Two domains are not independent seed replications; three seeds are limited.

Primary contrasts: EFFECT_RULE−ORIGINAL and EFFECT_RULE−PHASE_SHUFFLE, both on new-domain Macro Dice. Report all six cell deltas, three domain-averaged seed deltas, mean and sample SD; retain every negative cell. FINE and RULE−FINE are context controls. Old REFUGE Dice/change and relative old-domain differences are prespecified secondary outcomes; do not change the primary to a new/old average.

Descriptive continuation rule, not significance: both primary means positive and at least2/3 seed-level deltas positive for each contrast support a later independent confirmation proposal; if old-domain mean relative to ORIGINAL is negative, explicitly report a tradeoff. Any stronger claim requires further evidence. A small or mixed difference versus shuffle leaves state-dependent benefit unresolved, not proof of equivalence. If all methods finish300/600 but not1200, those remain secondary incomplete endpoints.

After completion audit the matrix, physical ledger, sources, and exact phase action counts once. Produce anonymous ENDPOINTS.csv, SUMMARY.json, MATCH_AUDIT.json, ALL_COSTS.json and a report under this package and Downloads/V3C_RULE_TIMING_20261001. Keep checkpoints, patient data, per-case scores, raw logs and features private on NAS. Public delivery follows the user's current project-wide completion instruction; push only this completed study's safe source/results, never raw data or unrelated changes. No intermediate or automatic next-round publication.

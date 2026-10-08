# V109: training-only supervised action margins

Frozen before new runs. V108 coverage improved actual deployment, but RL did not beat target-matched CE or 1NN. Sampled RL still lost about 4.803 percentage points against the old memory reference. All existing actions 0–11 activate the unlabeled loss. Hypothesis: a true supervised-only action adds useful learning/retention options absent from the current action set. This round measures training action margins only; it cannot establish campaign success.

## Action semantics

Add action 12 (OFF): call the existing native StageTrainer.losses for B0_PARENT_LCTX, returning its labeled loss with unlabeled=None and pending=None. It performs no U-data read or U forward during the native update. Keep the same native optimizer, LR schedule, constraints, EMA update and full snapshot lifecycle. Original actions 0–11 dispatch unchanged through the original transfer trainer. Feature extraction and behavior decisions still observe allowed U_adapt through the unchanged extractor, even when OFF is the selected update action. Do not equate action0 (alpha=0 with ordinary U consistency) to OFF.

## Frozen training data and branches

Use exactly V107's 80 keys: original8 + added12 contexts, streams1/2, steps100/200. Preserve all960 old returns and original/stable features. Do not collect new states for selecting retained prefixes. Reuse each stored ENTRY100 and fixed-behavior retained ENTRY200, restoring the original provider seed168+10000*stream and archived full state. Current behavior is the unchanged V99 four-actor CE/RL ensemble (601/602), with generator860901+100*stream+context_index where context_index0..19 follows original8 then V10712. Replay the original first selection and retained ENTRY200 state exactly before new branches.

For each of40 context-stream jobs:

1. Read Q_train_new at the already saved behavior-selected FIRST final snapshot once (4 images) to recover the old final-new baseline needed for the second-action delta. This baseline was not retained as a scalar in the original table. No retraining.
2. At stored step100 take OFF for100 updates; at step200 select a continuation among the original12 actions using the frozen behavior and private generator after its original first draw, then run100 more updates. Score Q_train_new at150/300 and Q_train_old at300 (12 images). Save the OFF prefix at200 for possible separately frozen future work; it is not used to choose this round's retained state.
3. At the previously stored behavior-retained step200 take OFF for100 updates. Score Q_train_new/old at300 (8 images). Its gain is the old selected continuation gain +0.75*(new final score−rescored old baseline final score). The first-action gain remains0.25*(new150−entrynew)+0.75*(new300−entrynew). Dense return is gain+oldfinal−oldentry, unchanged.

All Q_train entry/old baselines come from existing frozen records; old baseline final-new needs the explicit readout above. No Q_dev, hidden U labels, new annotation, sealed test, new source or later actual domain. No actor fit, source training, new stable probes or linear solves.

## Qualification and complete matrix

Before collection, two old training contexts0/7 stream1 verify: OFF two-step continuation equals a direct native supervised-loss two-step continuation exactly (full snapshot and RNG), with zero U reads and inactive U; original action9 one-step continuation equals the unwrapped existing trainer exactly. This is6 native updates per context,12 total, zero query calls. All old action mathematical checks remain active. Qualification admits the new branch only after these assertions.

Each collection uses300 native updates (200 OFF and100 behavior continuation),24 Q_train image evaluations and3 original behavior/state extractions. Full matrix40jobs:12,000 collection+12 qualification=12,012 native updates;960 Q_train image evaluations;120 original extractions;0 actor/linear/stable-probe/Q_dev calls. Count all failures and do not retry silently. GPU4–7 with memory admission, NAS create-only storage, neutral argv and native storage wrapper; no GPU wall-clock limit.

Seal80 new action returns and80×13 total returns with old columns unchanged. Report every state and context/stream/step margin of OFF versus the existing12-action maximum and mean, as well as gain and old-retention components. This hindsight training comparison is descriptive, not a deployed-policy gain or a replacement success threshold. A subsequent13-action fit and actual deployment require separate preregistration and the original matched-control practical criteria. Development-informed photometric overlap and shared cases remain disclosed.

Publish source, protocol, every new aggregate return, counts, qualification and failure records; keep images, features, checkpoints and raw logs private on NAS.

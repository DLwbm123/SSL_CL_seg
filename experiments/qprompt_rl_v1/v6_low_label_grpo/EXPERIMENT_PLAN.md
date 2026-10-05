# V6: low-label GRPO and group-size research

## Authority and scope

User authorized continuing experiments without a wall-clock limit, hourly monitoring, technical repair and continued research toward positive results on 2026-10-04. This supersedes the quick-screen four-hour limit for **new V6 runs only**. Prior V5 runs/results remain immutable. Fixed existing REFUGE source S168; only new-domain supervision varies. Physical GPUs5/6/7, neutral process argv, all outputs/checkpoints/logs/cache and source snapshots on NAS through the project wrapper. No test or hidden U labels, source retraining, unrelated processes or protected data changes.

Unlimited time does not predetermine a positive outcome. Every candidate, technical failure and consumed update is retained. Freeze each candidate before its measurements; do not alter thresholds, select a best seed, discard unfavorable domains/channels or describe tuning-val reuse as independent confirmation.

## Label budgets and controls

| Nominal budget | RIM labels /79 training images | Drishti labels /51 training images |
|---|---|---|
|20%|16 (20.25%)|10 (19.61%)|
|10%|8 (10.13%)|5 (9.80%)|
|5%|4 (5.06%)|3 (5.88%)|

For each subset seed, shuffle the existing authorized labeled patients deterministically and take nested prefixes. All images are single-patient images, asserted at runtime. One chosen patient supplies online controller reward, all remaining selected patients fit the student during policy development. No separate training-audit labels are used (audit values explicitly null, no audit ranking). All chosen labels may fit the final endpoint student after its policy freezes. All methods in a paired cell share the same chosen patients, source and complete entry state. Labels removed from the original L pool are stripped from the accessor metadata; those images join U. Frozen original HDF5/manifests/splits are unchanged. Original steps-per-epoch and full horizons3200/2100 are retained across budgets.

Annotation cost is the unique union of student, initialization and reward labels, not only the labels used by the student. Historical target-trained policies/scalers are not reused. New 25-step action panels, offline policies, softened RL initializations, normalization, calibration and references are built independently within each budget/subset. Panels cover every50 steps; three counterfactual25-step branches plus one retained full-U trajectory. Calibration uses4 complete initial-policy trajectories plus1 U0. Reward is unchanged online quality, fixed calibration scale; the reward provider holds one patient and is a possible overfitting risk measured by separate final val results.

Matched endpoints: ORIGINAL (lambda_U=0), FINE_05 (fixed0.5), OFFLINE_25, and each GRPO/PPO configuration. Equal complete student horizons; endpoint controllers frozen with deterministic argmax. Every training job in a campaign must finish before **any** campaign val evaluation. Existing historical val is development evidence, never new independent patients.

## First registered campaign R0

- Source/segmentation seed168; label-subset seed5101; controller401.
- Three label budgets, two target domains.
- G=2,4,8; respectively4,2,1 groups, **8 total trajectories per learner/domain/budget**.
- Matched PPO at each G.4 epochs per group, **G minibatches**: each batch has one trajectory horizon's worth of decisions;32 actor steps and4 presentations of every sampled decision in every variant. PPO additionally has32 critic steps. Different policy refresh frequency remains an inherent difference, explicitly reported.
- Actor/critic learning rate0.0003, entropy0.01, symmetric ratio clip0.2, gradient norm1. Fixed-scale GRPO; same softened OFFLINE initialization within each cell.
- Complete-state U0 reference may be reused across groups only after the existing learner verifies identical full entry state and provenance. Group index is normalized in this V6 reference identity because every group uses the identical fixed source/seed/data entry. Never treat U0 as an on-policy sample.
-54 endpoints,126 training/preparation jobs, plus54 separate evaluation jobs.
- Student updates:48 qualification,39750 panel,79500 calibration,15900 reference,763200 development,143100 endpoints = **1041498**. Additional CPU optimizer calls:1536 offline,384 critic warmup,1152 actor,576 critic;96 synthetic actor-check calls are separate. Attempts from failed jobs remain additional consumed cost.

## Prespecified interpretation and continuation

Low-label primary cells are both domains at10% and5%, equally weighted. For each G, compare GRPO to the **best per-cell** of ORIGINAL/FINE_05/OFFLINE_25 and to same-G PPO. Screen candidate requires all:

1. Mean new-domain macro Dice gain >=0.5 percentage points versus best non-RL.
2. Mean gain >=0.2pp versus matched PPO.
3. Mean old REFUGE loss relative to ORIGINAL no worse than0.5pp; no individual cell worse than1.0pp.
4. No low-label cell loses >0.25pp against best non-RL.

Report rim, cup, disc-union, old-domain change, every method, all adverse cells, costs, policy KL/argmax changes, reward dispersion, and low-label gain minus20%-budget gain. A nonpositive interaction does not support a specifically stronger low-supervision advantage.

A passing candidate proceeds to **fixed-hyperparameter replication**, new subset seeds6101/6102 and controllers402/403 (four repeats, both domains/all budgets). Keep the same source model and the same registered thresholds; additionally require each repeat's mean new-domain gains versus best non-RL and PPO to be positive. A passing four-repeat replication is a replicated positive direction within this benchmark, not independent-patient or clinical validation. Freeze the selected configuration before starting replication. Publish the screen whether positive or negative.

If no candidate passes, diagnose retained reward/action/normalization/gradient evidence. Initial next candidates, each in a new create-only run with a frozen amendment and all three supervision budgets:

- R1: G2/G4/G8,8 trajectories, learning rate0.001; other settings unchanged.
- R2: G4/G8,24 trajectories, learning rate0.0003; other settings unchanged.
- R3: G4,8 trajectories, learning rate0.0003, entropy0; other settings unchanged.

A failed replication returns to the remaining candidate sequence; retain and publish the failure. For later parameter experiments, write a rationale and parameter scope before launch, preserve comparator budgets and all earlier results, and use fresh registered subset/controller replication seeds. Do not lower the positive-result criteria, retune against sealed test data, modify segmentation losses/architecture or silently broaden source/data/GPU scope. If a valid positive result cannot be established, say so; never manufacture success. Hourly research upkeep remains active while this authorized effort continues, until replicated success is published or the user stops it.

## Qualification, repairs and monitoring

Before performance: CPU G2/G4/G8 accounting/gradient test; every domain/budget passes native8-step qualification for data roles, feature/reward state immutability, exact repeated native updates and checkpoint restore. Original native engine/losses are reused.

Scheduler restart adopts its own live workers using PID+start ticks and skips completed jobs. Healthy work is not restarted. On technical failure, stop new launches while other active workers finish. Diagnose first; fix source/dependency/storage/entry issues, run the minimal relevant check, and record a repair. Preserve the old code revision and failed directory. An explicit `RETRY_REQUEST.json` containing failed job IDs and a diagnosed reason authorizes the coordinator to move those attempts into `failed_attempts/` and retry only those jobs. Their work stays charged. A scientific setting change creates a new campaign instead of mutating failed or completed results. No blanket process kill, data deletion or clock/cost reset.

The hourly Codex heartbeat checks this campaign's actual processes, receipts, logs, GPU memory and NAS capacity; technical problems are repaired within these boundaries and the research sequence continues. Notify on meaningful changes, repairs, result availability, completion or required user action; remain quiet on unchanged healthy states. Publish anonymous aggregates and reports to the project GitHub branch through127.0.0.1:7897 and verify remote SHA/anonymous access. Private artifacts remain on NAS. Local heartbeat execution depends on the desktop app being available; remote workers are independent of the chat.

## User-authorized R0 timing amendment, 2026-10-05

See [EARLY_EVALUATION_AMENDMENT.md](EARLY_EVALUATION_AMENDMENT.md): evaluate all 18 already completed 20% endpoints early, reuse their registered evaluation receipts, keep remaining training settings and all gates fixed. This supersedes the global validation timing barrier for the 20% scope only. The final audit must disclose this amendment.

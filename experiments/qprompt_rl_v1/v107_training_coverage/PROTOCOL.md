# V107: training-state coverage collection

Frozen before collection. This is a training-only preparation round, not a claim of RL improvement. V106 actual sequential deployment failed: RL was below original CE and its matched-target advantage was negligible. Six of 24 ENTRY100 feature dimensions were outside the old training ranges on average; this descriptive observation motivates coverage, not a proven causal diagnosis.

## Hypothesis and scope

Add 12 contexts: memory steps {2000,8000} × labeled support {2,8} × {identity(1),brightness(1.2),contrast(0.8)}. Existing eight contexts, their V101 returns and V103 features remain frozen. Photometric conditions overlap the previously viewed development conditions; this is expressly development-informed training design. Cases remain exclusively existing A_fit, U_adapt and Q_train roles. No Q_dev, hidden U labels, sealed test, later actual domain or new annotation access.

V107 collects and seals the new 48 states and 576 returns, then combines them with existing 32 states/384 returns to produce 80 states × 12 actions. It fits no actors and reads no development outcomes. Following completion, separately preregister matched WARM/CE/RL/target-CE and simple controls and actual sequential deployment; no success gate applies to this preparation. Smaller alternative: fit old features again; rejected because V105 showed the objective was already nearly solved.

## Fixed collection

Reuse original native trainer, source checkpoints, 300-step horizon and 12 expanded actions. Build one shared ENTRY100 per new context by action 0 for 100 steps with seed/provider 168. No source training. Evaluate Q_train_new under context condition and Q_train_old under identity once per entry (4+4 images). Entry scoring is read-only.

For each context use streams 1,2: provider 168+10000*stream; Python/NumPy/Torch/CUDA seed 860100+stream. Fixed behavior is the V99 expanded CE/RL four-policy ensemble (601,602), using original single native state extraction as in V101. Private generator 860901+100*stream+context_index; context index is 8..19 in the listed Cartesian order. Select first action, then branch all 12 first actions for 200 updates each; choose second action using the same restored private generator state at step 200. Retain the behavior-selected first prefix. At this retained step 200 branch all other 11 second actions for 100 updates; reuse the already evaluated selected second return. Do not choose the retained state using labels/rewards.

For each first branch read Q_train_new at150/300 and Q_train_old at300. For each additional second branch read Q_train_new/old at300. Dense reward = .25*(new150-new100)+.75*(new300-new100)+old300-old100. Second-action gain replaces only the .75 final-new term. Store original behavior states and four stable probe states at each retained step100/200; use unchanged V103 probe provider168, cursors0..3, seeds861030+j, float64 mean and exact full-state/RNG restoration. Returns are aligned by explicit context/stream/step/action keys.

## Counts and qualification

12 entries ×100 =1200 native updates. 24 collection jobs ×3500 =84000. Qualification uses two old contexts/actions9/11, twice two-step replay =8. Total85208 native; zero actor optimizer/linear solve calls. Q_train image evaluations:96 entry +5568 branches =5664. Stable probes:192 new +16 qualification =208. Native original behavior extraction counts are additionally recorded. No GPU wall-clock limit; these operation counts prevent silent expansion. Use only GPUs4–7, >=12000MiB free on admission, NAS create-only directories and neutral parent/child command lines. No automatic retries, preserve failed attempts and historical costs.

Before new entries, reproduce two V103 stable states exactly, verify full snapshot/provider seed/global RNG unchanged by probing and behavior selection, and paired two-step continuations exactly restored. Synthetic assembly self-check rejects missing/duplicate action rows and nonfinite values. Then collection must match all counts, keys, finite features/returns and old arrays exactly on merge. Collection success is engineering completion only, not evidence of held-out gain. Publish source, protocol, counts and aggregate tables; retain private states/checkpoints/raw logs on NAS.

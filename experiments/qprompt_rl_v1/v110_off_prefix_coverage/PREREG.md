# V110 — frozen OFF-prefix training-state coverage

Hypothesis: the 13-action policy must see the new step-200 states created by its OFF first action. V109 found heterogeneous training margins (35/80 positive), but appended an OFF return only at the old retained states. This collection closes that specific coverage gap before policy fitting. It is not a performance or RL success test.

## Fixed scope and roles

Use all20 previous contexts (8 original +12 V107), streams1/2, all40 V109 OFF_ENTRY200 snapshots. Do not select by margin sign. Only existing A_fit/U_adapt/Q_train, no Q_dev, hidden U labels, sealed tests, new cases or actual later domains. V109 native action12 supervised loss and old0–11 action dispatch stay unchanged; provider seed168+10000stream and snapshot RNG/state restore exactly. Fixed source, H300, optimizer/scheduler/EMA and native constraints remain unchanged.

At every OFF-prefix state, collect all13 action returns for the remaining100 updates. Reuse exactly one original-action continuation already executed and scored in V109 (FIRST_OFF_FINAL plus first-OFF result). Execute each of the other12 actions once, including action12, with identical starting snapshot. Evaluate each new endpoint once on Q_train_new4 and Q_train_old4. Dense gain = prior V109 first-OFF gain +.75*(new endpoint−V109 selected endpoint); add old endpoint−old-entry. Save all new endpoint snapshots. Reused action has precisely its V109 recorded gain/return, zero new queries/updates. No actor-guided/adaptive collection.

At each of40 prefixes extract the original24-state once to confirm exact replay of V109's saved behavior state; then stable probe j0..3 using the existing V103 state/provider/RNG isolation, float64 storage and averaging. Qualification first: contexts indices0/7 stream1, repeat4probes twice with exact arrays and fullsnapshot/RNG equality, paired two-step OFF continuation with and without repeated probes. This costs8native plus16probes and zero queries. No duplicate fitting or re-running V109.

## Matrix, accounting, acceptance

40 jobs×12 non-reused actions×100 =48000 collection native updates; plus8 qualification =48008 native. Collection OFF4000 +original44000; qualification OFF8. Q_train40×12×8=3840 image evaluations (960 query calls). Stable probes160+16=176; original behavior-state extractions40. Zero actor/linear solves/Q_dev/newannotations. Physical operation counts are fixed reproducibility boundaries, not GPU time limits. GPU4–7 via NAS wrapper, create-only storage, neutral process arguments; preserve failures and spent work.

Append40 states×13 returns to immutable V10980×13 table and V107 original/probe features:120states1560returns. Keys explicitly include prefix ORIGINAL/OFF to distinguish two step200 states per context/stream. Old80 rows, features and1040returns must remain exact. Validate unique keys,13 actions per newstate, finite24-d features, completephysical pairs, queries and formulas, reused40returns match V109. Publish all520new return rows and summary including negative results. State vectors/weights/snapshots/rawlogs remain private. Engineering completion is the only current endpoint.

## Required next stage

After completing and publishing this frozen collection, separately freeze13-action WARM/CE/RL/target-matchedCE fitting and real sequential deployment with fixedOFF and original12-action historical controls. Use unchanged success/practical thresholds, bothactorseeds601/602, and retain allcontrols. This collection itself must not trigger a success conclusion or another automatic coverage expansion. Development-informed factors/shared source are not independent evidence. A candidate still needs separately frozen fresh-stream confirmation.

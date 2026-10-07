# V8.2 independent episode-alignment diagnostic

Authority: the user explicitly requested continued analysis and improvement after poor outcomes, with image-level exploration and no patient-independent claims. Frozen V8 and V8.1 remain stopped and published; this is a new bounded diagnostic, not a continuation past their failed gates.

## Evidence and one intervention

V8.1 reduced skipped actor groups to 30/64 and 32/64, but both mean utilities remained below uniform and matching V8 priors. Retire the floor-only intervention. Training decisions occurred at student steps 100–800 under a registered 900-step native schedule, while development decisions are at 100/200 under a 300-step schedule. Progress features and optimizer schedule therefore differ; states also have different accumulated adaptation lengths. This is a plausible training/use mismatch, not a proven cause or an original-protocol violation.

Change policy training to independent native 300-step episodes: 100 native entry steps, then decisions at 100 and 200, each lasting 100 updates. Each context is visited eight times, giving four two-decision episodes; reset that context to its immutable native entry before each episode. Within episodes retain branch zero after each group, never the best reward. A fresh qualified A-only native entry is prepared for each of eight original training contexts, because restoring the old 900-step snapshot would silently restore its scheduler and horizon.

Use the same two freshly zero-output actors 601/602, same 64 groups each, four candidates, rewards and query roles, nine actions, calibrated V8.1 floor, fixed old-memory reference, GRPO epochs/optimizer, teacher, source snapshots, photo conditions and support amounts. Native entry preparation never queries Q_train or Q_dev. No extra clean-source or action-audit training.

## Checks and outcome

Run the qualified paired-branch/readout/actor-isolation check under horizon 300 before entry preparation. Verify each saved training entry has step 100, horizon 300 and progress 1/3; runtime decisions must alternate step 100/200 and final context states must equal step 300. All 20 unchanged development endpoints finish and seal before query readout. The controls are native, fixed, uniform and both final priors. Q_dev is now reused for diagnostics; it cannot establish independent generalization.

At least one final prior must improve mean utility by 0.0005 over both uniform and matching V8.1 prior, while new Dice stays within -0.0025 and old Dice within -0.005 of uniform. Otherwise STOP_EPISODE_ALIGNMENT_NO_PRACTICAL_GAIN, preserve all results, retire this episode intervention and do not start targets automatically. If successful, a separately frozen target experiment and native qualification are still required. No checkpoint, condition, floor or seed selection.

## Finite budget and accounting

New student cap 56,408: qualification 8, training entries 800, prior branches 51,200, development entries 400, development training 4,000. New actor cap 520: qualification 8 and priors 512. All attempts, including failures, count. Carry the complete V8+V8.1 prefix once: 134,437 student, 412 actor; common historical source 8,000 student is recorded separately. No unused quota transfer, extra seeds or retries. Private artifacts stay in a new NAS directory; GPU 5/6/7 only. Publish complete anonymous results and source through the proxy.

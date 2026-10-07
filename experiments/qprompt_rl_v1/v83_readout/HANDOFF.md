# Current handoff — 2026-10-07

V8.3 finished with STOP_DENSE_REWARD_NO_PRACTICAL_GAIN. V8.3-R completed with STOP_V83_PRIMARY_GATE_FAILED. The original screen and all cumulative physical ledger counts were reproduced, all 288 rows audited, both final actors retained and all 20 endpoints confirmed before their development readout. No V8.4 training is permitted by this result.

The independent readout run is v83_readout_20261007T093208Z on the existing project NAS protocols root. The successful script is readout_recovery_01.py from commit e0a14c9; the original failed readout.py and worker.log remain intact. Root readout artifacts contain the frozen manifest/plan; results contains the anonymous text outputs and private LOCO checkpoints. First failure consumed zero optimizer calls; successful run consumed 1,025 CPU actor and zero student updates. See PATCH_LOG.md and OPERATIONAL_COSTS.json.

Main evidence: conditional state selection passes the cross-stream gate (+0.000811794 over global, +0.000648789 over time); LOCO fails the time-fixed comparison for both seeds (about −0.00025). Full-table actors nearly ignore context and underperform fixed controls. A clearly exploratory exact-soft-target check also underperforms time-fixed expected reward. See FINAL_INTERPRETATION.md for limits, costs and all eight requested answers.

The next authorized work is publication of this completed round, then a separately frozen minimal CPU-only investigation of target sharpness and input conditioning. Do not rerun V8.3-R folds, change its thresholds, or call that new experiment V8.4. The user's later hourly-maintenance and continued-research authorization remains active, without a GPU wall-time deadline. Keep each new experiment bounded and preregistered; do not unlock new datasets, hidden labels, real subsequent domains or the low-label full matrix without the required concrete authorization.

## Latest active work

V8.3/V8.3-R were published and anonymously verified at 9b01b3f; V8.5's full negative 2×2 CPU attribution was published and verified at 88a4a17. V8.5 used 3,459 CPU actor updates and zero student updates. It did not unlock a GPU test.

V8.6 is the next independently frozen, single optimization-budget diagnostic. Protocol and implementation are in v86_cpu_fit_budget; preregistration ccf4915, execution046e334. NAS run v86_cpu_20261007T100027Z is CPU-only, with a 36,866 actor-update cap and zero students. It compares 1,024-update raw/standardized sharpened-target fits against reused 64-update counterparts, with an exact update64 tensor-prefix check. Do not launch a duplicate or repeat any failed training blindly. ACTIVE_STAGE.json is the current stage pointer; this section supersedes earlier tentative next-step wording. V8.4 remains NOT_RUN.

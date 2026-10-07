# Current handoff — 2026-10-07

V8.3 finished with STOP_DENSE_REWARD_NO_PRACTICAL_GAIN. V8.3-R completed with STOP_V83_PRIMARY_GATE_FAILED. The original screen and all cumulative physical ledger counts were reproduced, all 288 rows audited, both final actors retained and all 20 endpoints confirmed before their development readout. No V8.4 training is permitted by this result.

The independent readout run is v83_readout_20261007T093208Z on the existing project NAS protocols root. The successful script is readout_recovery_01.py from commit e0a14c9; the original failed readout.py and worker.log remain intact. Root readout artifacts contain the frozen manifest/plan; results contains the anonymous text outputs and private LOCO checkpoints. First failure consumed zero optimizer calls; successful run consumed 1,025 CPU actor and zero student updates. See PATCH_LOG.md and OPERATIONAL_COSTS.json.

Main evidence: conditional state selection passes the cross-stream gate (+0.000811794 over global, +0.000648789 over time); LOCO fails the time-fixed comparison for both seeds (about −0.00025). Full-table actors nearly ignore context and underperform fixed controls. A clearly exploratory exact-soft-target check also underperforms time-fixed expected reward. See FINAL_INTERPRETATION.md for limits, costs and all eight requested answers.

The next authorized work is publication of this completed round, then a separately frozen minimal CPU-only investigation of target sharpness and input conditioning. Do not rerun V8.3-R folds, change its thresholds, or call that new experiment V8.4. The user's later hourly-maintenance and continued-research authorization remains active, without a GPU wall-time deadline. Keep each new experiment bounded and preregistered; do not unlock new datasets, hidden labels, real subsequent domains or the low-label full matrix without the required concrete authorization.

## Latest active work

V8.3/V8.3-R were published and anonymously verified at 9b01b3f; V8.5's full negative 2×2 CPU attribution was published and verified at 88a4a17. V8.5 used 3,459 CPU actor updates and zero student updates. It did not unlock a GPU test.

V8.6 is the next independently frozen, single optimization-budget diagnostic. Protocol and implementation are in v86_cpu_fit_budget; preregistration ccf4915, execution046e334. NAS run v86_cpu_20261007T100027Z is CPU-only, with a 36,866 actor-update cap and zero students. It compares 1,024-update raw/standardized sharpened-target fits against reused 64-update counterparts, with an exact update64 tensor-prefix check. Do not launch a duplicate or repeat any failed training blindly. ACTIVE_STAGE.json is the current stage pointer; this section supersedes earlier tentative next-step wording. V8.4 remains NOT_RUN.

## V8.7 active — supersedes V8.6 pointer

V8.6 completed with its primary gate PASS, all36 update64 tensor prefixes matching, and36,866 CPU actor updates; full results are publicly verified at5a20b07. The Z1024 held-context expected reward exceeds time control by about0.00117 for both seeds. This is a saved-table diagnostic, not deployment or independent evidence.

V8.7 was separately frozen atf6f6eb7 and implemented at7103305. Runv87_dev_20261007T101829Z is now on the project NAS protocols root; initial coordinator1371631, startup qualification1371830. Verify current identities before acting; these PIDs are clues only. The new root owns run.py, CONFIG.private.json, jobs, PHYSICAL_LEDGER.jsonl and coordinator.log. The native owning engine remains the audited originalV83 exported source. All12 existing controls passed reuse audit.

Matrix: native, fixed-global8, uniform (all reused), time-fixed6/8, Z64 bothseeds, RAW1024 bothseeds, Z1024 bothseeds. Four original development contexts,40 endpoints,28 new; exactly8 qualification+5,600 development student calls, zero actor updates. Entry100 and private categorical seed860301+context are paired; H300, decisions100/200, final-only T1 deployment. No wall-clock deadline. All endpoints lock before new Qdev scores. Outputs remain onNAS and checkpoints stay private.

The coordinator automatically proceeds only if qualification passes, queues jobs onGPUs5/6/7 when at least12GB free, and stops on any job failure without retry. A finished stage still requires complete public collection, interpretation, costs, proxy push and anonymous verification. Check every hour per the existing authorized automation; do not create duplicate automation or retrain a failed prefix blindly. Preserve V83/R/V85 STOPs and V84 NOT_RUN; currentD1 remains repeatedly observed development.

Startup follow-up: qualification PASS with8 updates and two exact full-state pairs; train0/train1/train2 running onGPUs5/6/7, train3 queued. Last startup snapshot205 development calls, no failed jobs. Neutral worker command lines and nvidia-smi names checked. No formal new development score read yet.

## V8.7 complete — 2026-10-07 18:43 Asia/Shanghai

V8.7 completed40 endpoints and all9 jobs exited0. New physical cost5,608 student /0 actor, no failures. All new scoring began after the endpoint lock. Frozen result STOP_V87_NO_PRACTICAL_DEPLOYMENT_GAIN: Z1024 mean new/old soft Dice73.940168/81.183862 versus uniform73.775267/81.289067. Utility gain+0.000183593 fails0.0005, and new gain0.164901pp fails0.20pp. Both seeds improve over Z64; uniform comparison remains mixed across seeds and contexts. Full results and audited ledger are in v87_deployment_confirmation. Actual final receipts override stale per-job progress snapshots.

No next experiment is launched. At the next authorized research continuation, independently freeze a zero-update analysis of saved step100/200 deployment features to diagnose table-to-deployment shift before deciding another optimizer intervention. OriginalSTOPs and V84NOT_RUN remain. Do not treat this diagnostic suggestion as a frozen plan or claim.

## V8.8 complete; V8.9 active — 2026-10-07

Latest human request explicitly asks to think carefully and improve the RL method. V88 zero-update coverage diagnosis completed and was publicly verified atc07dd30. Its broad-shift gate did not pass:1/4 unique primary step200 states flagged, saturation median increase0.15625 below0.25. Dev2 is the localized outlier and worst V87 context. All48 saved actor probabilities replay; no new image query/optimizer update. Do not turn this into a broad OOD or causality claim.

V89 preregistration07d2ba3, execution3a65f93, NAS runv89_episode_20261007T115349Z, initial coordinator1439151. Verify currentPID/start/EXEC_RUN/lock rather than trust these IDs. It uses the original V83 native source, with a new create-only run.py and budget_helper.py; CONFIG.private.json owns input paths.24 historical development endpoints passed reuse audit. The two-update CPU CE/RL objective qualification passed; native8-update qualification gates collection. No new actor fitting occurs during rollout collection.

V89 tests local versus full-episode reward and explicit expected-return RL. Frozen behavior is the mean probability of both V86 Z1024 actors. For every8 training contexts x2 original stochastic streams, enumerate9 first actions100->200 plus behavior continuation200->300; retain the initially sampled first-action state200 and enumerate all9 second actions, reusing the already executed continuation.2600newstudents perjob,16jobs=41600. Both local25/100 and full-episode150/300 readouts come from identical rollouts; Qtrain image evaluations5056. Private RNG860901+100stream+context, student stream asV83. NoQdev information in actor fit.

Three arms LOCAL_CE / EPISODIC_CE / EPISODIC_RL, both601/602, start from their own V86 model, fixed old scaler,1024updates each. RL directly optimizes expected clipped normalized full-return advantage with fixedKL.25 to the initial policy and entropy.01; this is full-action policy improvement, not GRPO or renamedV84. All three see identical32states and9-action return vectors. Native engine/loss, architecture24->32->9, A-only, actionspace, H300, T1sampling stay fixed. No hyperparameter/seed search.

Primary onlyEPISODIC_RL: must beat both CE arms, frozenZ1024 and all four simple controls by>=.0005 meanutility, eachseed>matchingEPISODIC_CE, and practical tradeoff versusuniform/EPISODIC_CE. SecondaryCE improvement cannot replacefailedRL.48endpoints (24new+24reused) locked before newQdev scoring. Newtotalcaps46408students and6146actors; students8qualification+41600collection+4800development, actors2qualification+6144fits. NewQdevimageevals288. No wallclocklimit. OriginalSTOPs andV84NOT_RUN remain, currentD1 is not independent validation.

Coordinator runs27 bounded jobs: CPUactorqual, GPUqual,16collection, CPUfit,4devtrain,4deveval. GPU jobs use5/6/7 only with>=12GB free, neutral env-based commandlines, NAS wrapper. Stops onfailure without retry; preserve physical/query ledgers and originalcode before any engineering recovery. Public delivery after completion must include all48endpoint results, fit/reward summaries and provenance, query/optimizer costs, failure history and frozen decision. Private states/modelweights/rawfeatures/roleidentities remainNAS.

V89 startup check: both qualification jobs PASS (2CPU actor+8native students). Collection jobscollect0_1/collect0_2/collect1_1 running onGPUs5/6/7,13jobs queued; snapshot370collection updates, nofailedjobs. WorkerPIDs1440228/1440239/1440315. Coordinatorlock held, processrun identities andneutral ps/nvidia commandlines verified. A monitor-only read raced with an exited CPU qualification process and got/proc environ PermissionError; that did not affect experiment execution or consume updates. No experiment engineering recovery occurred.

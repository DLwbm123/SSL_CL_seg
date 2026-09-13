# DPR V0.1 completed result interpretation

All18 target trainings and18 final evaluations completed, with36 successful child exits,47,700 formal optimizer updates/supervised backwards and76,320 response VJPs. The run ended2026-09-12 18:37:16+08:00 after3h41m10s. Execution source12ba9232204d5f183f48b67261a91efc0e4f6392 remains immutable and clean. The fixed matrix has stopped.

The pre-specified terminal is **SMALL_POSITIVE_PRIMARY_SIGNAL**. DPR_U Final0.607148 minus reused F_CONV0.606879 is only+0.000269, below the+0.005 priority-replication threshold. The paired development-patient95% interval[-0.005602,+0.006105] includes zero. The three order-averaged seed differences are-0.012578,+0.010843,+0.002542; paired-seed SD is0.011875. This is a near-tie in observed overall accuracy, with no established stable gain over F_CONV. The primary was not redefined after observing the controls.

| Arm | Final | Incoming | Old | Absolute Forget |
|---|---:|---:|---:|---:|
| F_CONV (reused primary reference) |0.606879|0.719677|0.494081|0.190738|
| DPR_U (only primary) |0.607148|0.715738|0.498558|0.186261|
| RESPONSE_U |0.604956|0.718256|0.491655|0.193164|
| DPR_L |0.602582|0.715281|0.489883|0.194935|
| F_FULL (reused secondary reference) |0.600145|0.718077|0.482212|0.202606|

DPR_U trades Incoming-0.003939 for Old+0.004477, hence Forget-0.004477. The O1 Final delta is-0.003430 and O2 is+0.003968. The old-domain macro improvement mainly comes from cup: old cup+0.012986, old rim-0.004032. Current rim-0.003009 and current cup-0.004869 both decrease. These are reported tradeoffs, not additional conjunctive gates.

DPR_U minus RESPONSE_U Final is+0.002192, with interval[-0.001555,+0.005762]. The supervised-progress equality has a positive observed mean contribution but its empirical increment remains uncertain. DPR_U minus DPR_L is+0.004566, with interval[+0.000016,+0.008990]. The current-U query has some positive paired-development evidence relative to current-L query; its interval lower bound is barely above zero. This component result does not establish overall superiority to F_CONV or independent-patient generalization. The secondary F_FULL difference is+0.007003 with interval[-0.002361,+0.015191], also uncertain.

The engineering behavior holds. Every committed numerical check passed. For DPR_U the sampled linear-response norm ratio averaged0.71547 (about28.45% reduction); the correction norm averaged0.8966% of the raw Adam increment. Actual applied progress residuals stayed below9.77% of the independently frozen FP32 rounding budget. DPR_L also passed its equality/budget checks. RESPONSE_U intentionally has no progress-equality constraint. No active step had a zero-probe or zero-task-gradient degeneracy. About4.62% of DPR_U's raw Adam proposals lacked supervised first-order descent; preserving those proposals' progress is not a descent guarantee.

Finite steps show a smaller effect than the sampled linear proxy. At21 pre-specified DPR_U diagnostic points, mean pooled contrast-logit RMS ratio was0.94137 (about5.86% reduction), median0.97973;20/21 had smaller response changes. Only13/21 had lower actual same-batch supervised loss than the raw Adam proposal. RESPONSE_U and DPR_L mean response ratios were0.94083 and0.94914. These diagnostic comparisons use only the already-read current training batch, never determine acceptance/lambda, and cannot establish historical memory preservation. See results/MECHANISM_SUMMARY.json and results/ACTUAL_STEP_RESPONSE.csv for scope and denominators.

Training made85,860 ordinary supervised forwards,38,160 response-query forwards and567 diagnostic forwards. Evaluation added1,170 forwards, giving125,757 model calls and249,384 image-forwards total. U file accesses were49,920, L accesses95,400, and pseudo-labels0. The extra response queries, two VJPs per active step and NAS checkpoints add training cost; matching optimizer updates does not match FLOPs or wall time. The sum of task training wall times is48,397.7seconds, including I/O and shared-GPU effects, not measured GPU compute time.

Peak reserved CUDA memory was1,004,535,808bytes (958MiB). The run retains student+current EMA during training; parameter snapshots, gradient rows, FP64 matrices/vectors and Adam state are accounted separately. Deployment is the unchanged ordinary single-student architecture with438,192 adapted convolution scalars; no response solver, controller, routing or additional input is deployed. All18 weights were sealed before any final evaluation.

Qualification cost remains separate:105 synthetic optimizer updates across all retained attempts, plus12 discarded real-L smoke updates. There were0 formal failed-attempt updates and0 source/baseline retraining updates. Eighteen final students,36 epoch20/100 recovery checkpoints, per-step logs and private patient metrics remain on NAS. Completion checked receipts, successful exits, recorded identities, checkpoint availability and all47,700 scalar diagnostic rows; it did not rerun training/evaluation or rehash every model. Public delivery contains only code/configuration/aggregate results and provenance.

The patient intervals use2000 shared-weight paired bootstrap draws across physical domains with seed2026091204, conditional on these trained models and repeatedly exposed development patients. They do not constitute independent-patient confirmation; the three-seed variation is a separate uncertainty source. Both historical CIST NO_PRIMARY_ACCURACY_GAIN terminals and all earlier evidence remain unchanged. No new seed, rank, parameter search, old-KI search or other experiment was started.

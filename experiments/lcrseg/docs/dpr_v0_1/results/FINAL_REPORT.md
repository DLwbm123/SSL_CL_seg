# DPR V0.1 final report

18/18 target tasks;47,700 formal updates and supervised backwards;76,320 response VJPs. Engineering complete. SMALL_POSITIVE_PRIMARY_SIGNAL.
Unique primary DPR_U minus reused F_CONV Final: +0.000269; development-patient95% interval [-0.005602,+0.006105]. Seed order-averaged effects: [-0.012578061025428744, 0.010843391892128962, 0.00254178656791032].

| Arm | Final | Incoming | Old | Absolute Forget |
|---|---:|---:|---:|---:|
|DPR_U|0.607148|0.715738|0.498558|0.186261|
|RESPONSE_U|0.604956|0.718256|0.491655|0.193164|
|DPR_L|0.602582|0.715281|0.489883|0.194935|
|F_FULL|0.600145|0.718077|0.482212|0.202606|
|F_CONV|0.606879|0.719677|0.494081|0.190738|
|STATIC_SOURCE|0.551619|0.418420|0.684819|0.000000|
DPR_U minus RESPONSE_U Final +0.002192; interval [-0.001555,+0.005762]. positive development mean; inspect paired uncertainty.
DPR_U minus DPR_L Final +0.004566; interval [+0.000016,+0.008990]. positive development mean; inspect paired uncertainty.

All three arms update only the same438192 ordinary convolution weights in the14 F_CONV layers. Original source students, frozen readout and LCTX arithmetic are reused. No old model/source/baseline was retrained. Both CIST NO_PRIMARY_ACCURACY_GAIN terminals remain unchanged.
The response query is current-student contrast logits pooled with geometry alone; two independently keyed Rademacher VJPs are detached and do not modify supervised .grad. Normalized J and supervised g use the same pre-update parameters. FP64 correction acts on the actual FP32 Adam increment; original moments/steps are retained. Native current EMA updates only after corrected parameters and supplies no target or response.
Warmup through epoch20 is the genuine native identity branch: no U query and no parameter rewrite. Active U access uses current train_unlabeled images only; DPR_L queries stripped, already-read current L tensors. There are49920 formal U image accesses,95400 L accesses, and0 pseudo-labels. No old-domain training images, history responses, features, prototypes, independent old teacher, or hidden/test GT were used.
Every update retains a corrected pending checkpoint before diagnostic acceptance; resume diagnoses pending state before another optimizer update. The scalar FP64 algebra residual and the actually applied FP32 progress residual are recorded separately. The latter uses a pre-frozen adjacent-FP32 rounding envelope, independent of observed residuals. No relaxed fixed1e-4 residual rule is used.
ACTUAL_STEP_RESPONSE compares raw and corrected finite steps with identical current training tensors at the first epoch40 step and then every500 updates. It reports pooled contrast-logit RMS changes and actual native LCTX loss changes against first-order predictions. Temporary switches restore corrected weights exactly and leave Adam, EMA and RNG unchanged. These measurements never decide acceptance or lambda.
TRAINING_ACCOUNTING contains per-epoch means from every step, including response norms, progress, raw non-descent fraction, condition numbers and degeneracy counts. RESOURCE_ACCOUNTING separates formal optimizer updates, supervised backwards, response VJPs, query forwards, diagnostic forwards, evaluation forwards, qualification, snapshots, Adam, EMA and FP64 workspaces. Equal optimizer steps are not equal compute.
All18 final students were sealed before isolated final evaluation. Deployment is one ordinary student without a response solver or extra module. Final is patient mean within physical domain, then equal old/current, equal orders within seed and equal three seeds. Primary threshold0.005 is not conjoined with all-domain/class/seed superiority; the controls never replace DPR_U.
Patient intervals use2000 paired draws with shared physical-domain weights and analysis seed2026091204, conditional on the trained models and repeatedly exposed development patients. They are not independent-patient confirmation. Three-seed SD and patient intervals describe different uncertainty.
A reduced sampled first-order response does not establish old-domain memory preservation or nonlinear loss decrease. Current-distribution coverage and the two-probe sketch are limited. No SOTA/novelty/clinical guarantee is claimed. The fixed matrix has stopped with no further search.
Only aggregate results, configuration, source and provenance are public. Images/GT, patient identifiers, per-case scores, checkpoints and raw NAS paths remain private.

# CIST Plasticity Add-on V0.2 final report

18/18 target tasks;47700 formal updates. Engineering complete. NO_PRIMARY_ACCURACY_GAIN.
Unique primary CONV_CIST_L minus reused F_CONV Final: -0.006385; development-patient95% interval [-0.013573,+0.000701]. Seed order-averaged effects: [-0.019083139996050102, -0.0007208999423117812, 0.0006475720092717907].

| Arm | Final | Incoming | Old | Absolute Forget |
|---|---:|---:|---:|---:|
|GN_L|0.569932|0.692715|0.447150|0.237669|
|GN_CIST_L|0.576895|0.697060|0.456731|0.228088|
|CONV_CIST_L|0.600493|0.723969|0.477018|0.207801|
|F_FULL|0.600145|0.718077|0.482212|0.202606|
|F_CONV|0.606879|0.719677|0.494081|0.190738|
|ISO_COND_L|0.563747|0.609688|0.517806|0.167012|
|ISO_COND_LU|0.563604|0.610021|0.517187|0.167632|
|STATIC_SOURCE|0.551619|0.418420|0.684819|0.000000|

GN_CIST_L minus GN_L Final: +0.006963; interval [-0.000473,+0.014147]. This control finding does not replace the pre-specified primary.

GN_L learns1408 GN affine scalars. GN_CIST_L learns4328 scalars. CONV_CIST_L learns the exact F_CONV438192 convolution weights plus2920 controller scalars. Other core parameters and source-derived bases stayed fixed. The full gradient flows through current features, context and fixed readout; there is no global no_grad on the training feature path. These are not equal-parameter comparisons.
All18 final students were sealed before any new patient evaluation. Only current L images and own train_labeled GT were used in training. U image access, source training-image access, EMA and teachers are zero. Original complementary LCTX donor/source collection and CE/Dice arithmetic were reused. Epoch20 is an unevaluated recovery/diagnostic checkpoint; epoch100 is the only deployment evaluated.
Every dynamic GN/conv/controller parameter, Adam, RNG, counters and position was checkpointed; pending diagnosis precedes another update. Deployment includes updated core parameters and the controller where present. Frozen subsets, rather than whole source-core equality, are enforced. Parameter drift and isometry describe current H versus its same-forward transport, not source functional invariance.
TRAINING_ACCOUNTING explicitly divides epoch sums by steps. PARAMETER_CHANGES, MECHANISMS and MEMORY_AND_COMPUTE disclose parameter changes, controller behavior, within-image distortion, real updates, memory and runtime. Temporary current-step recovery matrices are removed after commit; no historical patient Q/b library is maintained.
The new primary mean threshold0.005 is not conjoined with seed/domain/class requirements. No individual class loss automatically vetoes it. Reported2000 patient-bootstrap intervals use shared physical-domain weights and analysis_seed2026091202, conditional on repeatedly exposed development patients and trained models. They do not establish independent-patient confirmation or clinical safety.
Any main gain supports the entire module added to the F_CONV update space; old frozen-core ablations do not isolate geometry, conditioning or basis contributions here. GN updating is not a novelty claim. Old CIST is not a complete GOLD reproduction, and its negative terminal remains unchanged. The old global-isometry U invariance is an analytic limitation of that comparison; no GLOBAL rerun was performed.
The fixed experiment has stopped. No new seed, rank, layer, U objective or source-weight search was started. Public outputs contain aggregate metrics and cost/provenance only. Images/GT, patient identifiers, per-case scores, weights and raw NAS paths remain private.

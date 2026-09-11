# CIST V0.1 final report

36/36 new target tasks, 95400 optimizer updates; source/baseline retraining0. NO_PRIMARY_ACCURACY_GAIN.

Primary ISO_COND_LU minus F_CONV Final: -0.043275; paired-patient95% CI [-0.064430, -0.022930]. Order-averaged seed deltas: [-0.032478711341214606, -0.03389668764323539, -0.06344880544446585].
The primary decision uses the pre-specified mean Final signal. Secondary domain/class/seed costs and component ablations do not form conjunctive acceptance gates. A zero-crossing CI means uncertainty; a negative mean is not an accuracy gain. Old RETENTION_VALUE_NOT_ESTABLISHED is unchanged.

| Arm | Final | Incoming | Old | Absolute Forget |
|---|---:|---:|---:|---:|
|SCALE_LU|0.558869|0.567995|0.549744|0.135075|
|AFFINE_LU|0.560332|0.623181|0.497483|0.187336|
|ISO_GLOBAL_LU|0.556148|0.601194|0.511101|0.173718|
|ISO_COND_L|0.563747|0.609688|0.517806|0.167012|
|ISO_COND_LU|0.563604|0.610021|0.517187|0.167632|
|ISO_RANDOM_LU|0.550523|0.595787|0.505258|0.179561|
|F_FULL|0.600145|0.718077|0.482212|0.202606|
|F_CONV|0.606879|0.719677|0.494081|0.190738|
|STATIC_SOURCE|0.551619|0.418420|0.684819|0.000000|

| Full method minus ablation | Final | Positive mean increment |
|---|---:|---|
| ISO_COND_L | -0.000143 | False |
| ISO_GLOBAL_LU | +0.007456 | True |
| AFFINE_LU | +0.003272 | True |
| ISO_RANDOM_LU | +0.013081 | True |
| SCALE_LU | +0.004735 | True |

Source core, readout and normalizations stayed frozen; original 3x3 padding0 and output interpolation align_corners=True were retained. CIST uses one shared2920-scalar controller, FP64 differentiable8x8 Cayley solve and an explicit FP32 feature application. SCALE has16 trainable scalars. AFFINE and GLOBAL match raw MLP size but not effective degrees of freedom; GLOBAL input weights have zero gradients.

The source-derived V/N basis is a16-channel convolution-contrast eigenspace, not a flattened rank2/3 row space. N is less readout-sensitive, not a proven style/null subspace. Random geometry is a combined coordinate/context/view-loss control. SCALE is GOLD-inspired only: no source prototypes, AGOP, normalization adaptation or complete GOLD reproduction. No novelty or SOTA claim is established.

Only current labeled training images/GT and, for LU arms after epoch20, current unlabeled images were accessed. U views retain coordinates and use a scalar gamma plus independent per-image RGB gains and Gaussian RGB offsets as frozen in PROTOCOL.json. U supplies feature consistency, not pseudo labels. No EMA, teacher, replay, patient controller/Q/b cache or historical image access occurs in training. Source/baseline evaluation scores were reused.

Deployment contains one frozen source core plus the trained small controller and fixed bases. It requires one original image, one core and one controller; it cannot generally merge into a fixed convolution. Evaluation adds one readout-only source diagnostic, explicitly counted, and does not ensemble views. Per-image Q/b are not archived. MECHANISMS reports aggregate conditional variation, distance distortion, class fractions and confidence; TRAINING_ACCOUNTING reports view alignment before/after.

Real-arithmetic isometry concerns within-image feature distances only. It does not guarantee old predictions, cross-image geometry, class margins, calibration, boundaries or collapse avoidance. Shared-controller forgetting remains possible.

All36 final weights were sealed before complete-matrix analysis. Two orders are averaged within each of three optimization seeds;2000 paired bootstrap draws share patient weights within each physical domain across every arm/seed/order. Intervals condition on trained models and repeated development patients, not independent-patient confirmation. Full class costs, worst seeds, patient-delta tails, Pareto positions and real compute/memory are separate published tables.

The fixed matrix has stopped. No new seed, third domain, rank sweep, dense backbone update or follow-up experiment is admitted. Weights, RNG, raw images/GT and per-case outputs remain private on NAS.

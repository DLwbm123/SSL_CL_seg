# DPR Finite Response V0.2 final report

12/12 new target tasks,31,800 formal optimizer updates. NO_PRIMARY_ACCURACY_GAIN.
Unique primary DPR_FINITE_U minus reused F_CONV Final: -0.013513; paired development-patient95% interval [-0.019594,-0.007552]. Seed effects: [-0.016258432606590045, -0.003597348216494878, -0.020682590453114158].

| Arm | Final | Incoming | Old | Absolute Forget |
|---|---:|---:|---:|---:|
|DPR_FINITE_U|0.593366|0.708729|0.478003|0.206816|
|DPR_FINITE_SIGN_U|0.611610|0.717091|0.506130|0.178689|
|DPR_U|0.607148|0.715738|0.498558|0.186261|
|RESPONSE_U|0.604956|0.718256|0.491655|0.193164|
|DPR_L|0.602582|0.715281|0.489883|0.194935|
|F_FULL|0.600145|0.718077|0.482212|0.202606|
|F_CONV|0.606879|0.719677|0.494081|0.190738|
|STATIC_SOURCE|0.551619|0.418420|0.684819|0.000000|

Original sources and historical baselines are reused without training. DPR V0.1 SMALL_POSITIVE_PRIMARY_SIGNAL and both CIST negative terminals remain immutable.
One native Adam proposal and one finite candidate per nondegenerate active step. Detached probes use the realized before-to-raw response difference; both J and b share Rnorm. VJPs are at raw parameters and do not contaminate supervised gradients. The single2x2 solve preserves first-order supervised progress within separate FP64 and independently frozen FP32 rounding budgets.
A candidate whose full pooled response increases is rejected in favor of bitwise original Adam weights. Adam moments advance once and EMA follows only final adopted weights. Rejection is normal learning, not task failure. Nonfinite values are engineering failures. No lambda search, interpolation or tuning.
ACTUAL_STEP_RESPONSE separates sampled linear, same-detached-probe finite, full pooled field and native-resolution response for candidates and applied steps. Full-pooled nonincrease is imposed by the guard and is not independent evidence of historical retention. These estimands cannot be divided into a V0.1-to-V0.2 conversion rate. The comparison with DPR_U changes the entire update algorithm; it does not isolate a single mechanism.
DOMAIN_CLASS_COSTS, PAIRED_EFFECTS and PATIENT_DELTA_DISTRIBUTIONS disclose new/old-domain, class, seed and patient-tail tradeoffs. They do not add all-direction-wins gates to the pre-specified equal Final mean. DPR_FINITE_SIGN_U never replaces the primary.
All12 final weights were sealed before isolated validation. Deployment is an ordinary single student; current EMA is never queried. No replay, old teacher, historical responses, pseudo-labels, old-domain training images or test/hidden GT. U supplies only current response queries.
Bootstrap2000 draws with shared physical-domain patient weights use analysis seed2026091206. This is repeatedly exposed development evidence conditional on trained models, not independent patient confirmation. Seed SD is a separate uncertainty source.
RESOURCE_ACCOUNTING separates actual optimizer updates, supervised backwards, response VJPs, response/diagnostic/evaluation forwards, image accesses, qualification and memory. Matching optimizer steps does not match compute. Private patient-level records, weights, raw paths and data remain on NAS. No further experiments.

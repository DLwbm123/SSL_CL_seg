# Native qualification fixture repair after R2

The original approval applies to commit 2070845b679da42ea7a4df4268f492f2943deb6e only. It is retained outside the runtime checkout unchanged. This repair is authorized by the subsequent user instruction to fix the problem and continue training; it does not claim a new independent external review.

The first native CUDA attempt failed before any optimizer invocation because the native classifier has no bias parameter. The synthetic-only foreground fixture now gives rim kernels positive signs and background/cup kernels negative signs, using existing weights and the nonnegative native ReLU features. No parameter or model schema is added. The frozen PLAN phrase “foreground-biased readout bias” denotes foreground preference implemented through kernels, not a Conv2d bias parameter. Formal source models and scientific options are unchanged.

Admission accepts a distinct user repair receipt bound to the new commit/code hash, while checking the unchanged authentic external approval against its original five bindings. PLAN, source reuse and parent binding must remain identical; a fixed file allowlist restricts the patch to qualification, admission, its regressions and CPU/report metadata. It does not re-label the old approval as approving new code.

All prior CPU attempts and the failed CUDA session are retained. CPU regression and native CUDA/source/smoke results will be recorded separately. No additional scientific configurations, source training, real-data calls or monitoring are authorized.

CPU regression: 13/13 groups PASS, including a bias-free native-head schema/prediction regression. Historical cumulative calls 30/40; attempts 6/8. Environment unchanged: Python 3.10.6 and Torch 2.2.1+cu121. Native CUDA qualification remains pending until separately executed.

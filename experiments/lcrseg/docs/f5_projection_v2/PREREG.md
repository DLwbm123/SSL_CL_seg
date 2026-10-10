# F5 projection V2 — one finite follow-up

User authority: “如果结果出来不满意，你可以自己分析并提升，直至成功”. V1 completed and its complete negative/mixed result was delivered at `c624e171f14c9301487cfa52b14e7622e4171db7`, with anonymous access verified. Its costs, code and stops are preserved.

## Hypothesis and single intervention

V1 F5 had negative L-versus-total-U gradient cosine at10/32 predefined samples, five per seed. The linear adapter did not remove sampled conflicts and failed the success gate. These samples motivate a falsifiable optimization intervention; they do not establish persistent conflict, curvature or the cause of B2 superiority.

Keep the original F5_C02 model, A/B and R optimization, current EMA/PAS, LCTX CE+Dice, U KL+0.2 SWD, feature basis, rank ratio .25, lambda_U .25, .2 warmup/ramp, initialization, data, learning rates and schedule. Do not include boundary, OT triplet or added adapter branches. Only replace the merged gradient on existing R:

`g_R = gL + gU - min(0, dot(gL,gU))/||gL||² * gL`.

If gL is zero, keep gU; if no U loss, leave the original L update. Non-conflicting gradients are exactly unchanged. Dot products use FP64 accumulation, and every projected gradient is checked against a relative1e-6 raw-gradient half-space tolerance. U remains forbidden from updating original A/B. Existing split gradients supply both vectors, so projection adds no forward, backward, VJP, loss, learned coefficient, parameter or inference cost.

This borrows the conflict projection in [Gradient Surgery for Multi-Task Learning](https://arxiv.org/pdf/2001.06782), §2.3. The original projects each conflicting task gradient; this is explicitly a **one-sided supervised-priority adaptation**, not symmetric PCGrad. It is an optimization ablation, not a claim of a novel gradient-surgery algorithm. The half-space property concerns raw gradients: Adam momentum, adaptive scaling, decay and nonlinear losses mean it guarantees neither a non-increasing L loss nor old-domain preservation.

## Fixed budget and sequence

Reuse S163/S164 source checkpoints, source training0. Four trajectories: two seeds ×two orders,3200 RIM and2100 Drishti steps each, total21200 physical formal calls and8 target students. First run S163/O1 alone as the finite native diagnostic; only successful engineering completion admits the remaining3 trajectories. Do not inspect validation or select coefficients at this intermediate point. No failed formal retry, search, extra seed, combination or optional continuation.

Qualification caps: CPU optimizer96 (planned32), generated CUDA32 (planned10, including one deliberate after-optimizer failure), real current-L smoke4 (planned2). Every failure stays in cumulative ledgers. Native generated source forwards2, actual source patient forwards0. New root and code snapshot are create-only on canonical NAS; use the storage wrapper and GPUs4–7 with at least4GiB available. Do not kill other work.

Qualify conflict/zero/aligned/tiny-gradient math and two-case optimization, disabled-projection archived-V1 state parity, actual native backward/teacher freeze/U permissions, checkpoint continuation including projection counters, own-prefix stage transition, deployment equality and uncommitted-failure accounting. Reuse the V1 matched F5 control only after this parity passes. The unchanged teacher/data/parent semantics remain covered by V1 qualification.

Record every active-U projection/conflict count without extra queries; reuse four L/U diagnostic points per stage,64 extra formal VJPs total. No new image or gradient probes beyond the frozen qualification/diagnostic budget.

## Readout and decision

Seal all8 target students before20 seen-domain evaluations. Reuse V1's freshly matched F5 scores and historical B0/B2 aggregates with zero new control readouts; retain their provenance. Apply the existing practical gate unchanged against F5 and B2: both seed-mean Final gains positive, mean at least0.005, order-mean Old/Incoming at least−0.005, every individual rim/cup delta at least−0.05. Report all paired and negative outcomes. Do not relax the gate or promote a favorable single seed.

V1 formal84800 plus V2 formal21200 gives a cumulative ceiling106000 for these two rounds; qualification ledgers are added separately, never reset. Hourly monitoring remains active. After completion deliver code, protocol, anonymous results, diagnostic summaries and report through the required GitHub proxy. Continue only in another separately frozen, evidence-based round if this fails.

Development patients are reused; this is not independent patient confirmation, SOTA, clinical efficacy or proof of why F5 trails B2. No RL, test-guided tuning, replay, historical image access or reconstruction of the base method.

## Runtime admission and qualification

The host's installed NVML/libcuda580.178.04 did not match its loaded580.173.02 kernel, and CUDA initialization failed. No model training was attempted in that state. A matching `libnvidia-compute-580`580.173.02 package was fetched from the [official Ubuntu security archive](https://security.ubuntu.com/ubuntu/pool/restricted/n/nvidia-graphics-drivers-580/libnvidia-compute-580_580.173.02-0ubuntu0.24.04.1_amd64.deb) and extracted only into the new NAS run root. Only this round's process environment prepends its compute-library directory. System drivers, packages, service state and other processes were unchanged; no restart was performed. The [NVIDIA component documentation](https://download.nvidia.com/XFree86/Linux-x86_64/580.173.02/README/installedcomponents.html) identifies the corresponding driver libraries.

Isolated NVML queries and CUDA tensor initialization passed. Full native qualification then passed at execution commit `07b439773c3fe59b5eb8f97b8f2f7b0cc80692e2`, using32 CPU,10 generated CUDA and2 real-L smoke calls. These include one deliberate post-optimizer failure; the counters are preserved. Disabled-projection state parity passed against archived V1 and admits reuse of its matched F5 control. Combined V1+V2 qualification costs are416 CPU,48 generated CUDA and18 real-L smoke calls.

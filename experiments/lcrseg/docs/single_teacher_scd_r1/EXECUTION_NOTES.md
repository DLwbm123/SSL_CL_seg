# Execution and accounting notes

- All formal R1 stages use f94be07a8a3db1c062035a9bc2e542af9d4b156b. Later branch commits add public evidence only; the server's detached execution checkout is not advanced during training.
- The R1 adapter reuses V0.1's single stage loop and two-model lifecycle. Checkpoints retain the legacy structural arm D/E plus an independently checked r1_arm. Public tables use the R1 identity, and stage2 requires its own arm's stage1 path and r1_arm.
- L05/L10 never load U images. The inherited checkpoint schema retains an inert zero 195-byte prototype/support placeholder; there is no feature extraction, prototype estimate, PAS call or U loss in these arms. Those placeholder bytes remain reported.
- U0 computes and records raw Lu, retains weak/strong U forwards, PAS and U-KD, and sets actual_lambda_u to zero even after warm-up. nominal_lambda_u is separately logged.
- Per-five-epoch numeric trajectories retain batch quantiles. They are not represented as pooled epoch pixel quantiles. Pseudo-label precision is NOT_EVALUATED; coverage is not precision.
- E's raw class iterations field records the fixed rule (64). Actual per-analytic-pixel bisections are 64; boundary and unchanged branches perform zero bisections. Total pixel-iterations can be derived as 64 times analytical_brackets. Target/diagnostic timing includes scalar transfers and is not an isolated GPU kernel benchmark.
- Target byte counts are logical named tensor sizes and can include aliases. CUDA allocated/reserved and CPU RSS are measured separately. Epoch time sums are distinct from parallel parent wall time.
- First-step branch gradient probes are unweighted norms. Actual loss coefficients are separate fields; GAS is always extracted solely from supervised CE. No additional per-step whole-parameter diagnostic backward is introduced.
- Every old attempt, old terminal and old lock is retained. New formal updates and R1 qualification/replay/development updates are separate ledgers.

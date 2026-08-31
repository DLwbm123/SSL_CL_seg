# Prospective B0 v3 clean regeneration

Registered before formal B0 training. This document does not authorize execution; a separate committed/pushed execution authorization must bind its SHA and passing runtime evidence. Exact implementation: `90b45990b70d67ce325d638a223a1e003ca06003`, based on the requested handoff branch, with unchanged repaired Gate0 source hashes recorded in JSON.

Run only B0 (lambda 0.5), seed0 first. After seed0 completes all engineering, independent checkpoint/matrix/leakage, truthful parent-exit and verified private-archive gates, seed1 and seed2 may run on separate authorized shared physical GPUs. Only GPUs 5, 6 and 7 are permitted; do not kill jobs or reserve GPUs.

Each seed follows REFUGE → RIM_ONE_r3 → Drishti_GS, 100 epochs/domain and 5295 logged global steps. The complete frozen YAML is embedded in JSON; only data and official-reference location resolve to the new destination. Losses, thresholds 0.7/0.7, optimizer, scheduler, augmentations, EMA/GAS, stochastic training and deterministic formal evaluation are unchanged.

The adapter observes the original epoch-index-25 prototype forwards and records their actual returned bank, class counts and validity mask with model/RNG/data/code provenance. It adds zero model forwards. A best selected before this event retains its selected model/EMA/optimizer/scheduler/GAS/RNG/metric/sampler values and receives this genuinely captured bank once, at stage completion. Capture and selection chronology is explicit. There is no reconstruction. Sealed best checkpoints are for independent loading/diagnostics; resume uses the original latest-state path, never a sealed early best.

Hidden and test GT never enter training, optimizer updates, selection logic beyond the original validation role, or reliability fitting. The original frozen test policy is `evaluator_only`: the baseline six-cell stage-by-domain matrices necessarily read test GT in that isolated evaluator. This fact is explicitly disclosed and is not recorded as zero evaluator test reads. Gate1C itself remains test-GT-free.

All stage bests must independently load student/EMA, complete classifier, optimizer/scheduler and GAS, and match the original matrices. PAS fields and all required checkpoint groups have internal hashes plus outer-file SHA. Nonfinite values, missing rows, foreign domains, mutation or archive mismatch block admission. The server-local parent writes the actual exit; SSH exit never substitutes. Every finished seed must be byte-verified locally and atomically promoted to a private archive, with the remote copy retained.

At least 10 GiB must remain free before each seed; subsequent cache/formal phases require their own storage calculation. Existing environment only; no installs. Runtime tests use real UNet/CUDA with explicitly synthetic HDF5, plus read-only frozen-data checks. The duplicate legacy `/root/LCRSeg` TinySegNet test is a disclosed host-path skip, not a claimed pass.

New B0 is compared only descriptively with the old public B0. Flag `REGEN_BASELINE_DRIFT_REVIEW` if any same-seed final foreground Dice differs by more than 5 percentage points, or the new three-seed mean differs by more than 3. Never tune toward historical numbers. K2 and Gate1C remain separate later gates; no C0, method training, transport, Gate2, other dataset or sweep is authorized here.

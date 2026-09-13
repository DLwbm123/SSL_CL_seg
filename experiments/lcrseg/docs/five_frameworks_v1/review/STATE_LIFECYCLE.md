# State lifecycle and failure semantics

Stage entry creates only current A/B adaptation and current Q/R; F_prev is frozen. Q eigensolve and F2/F4 scale/direction probes run once. Then the current-stage EMA is copied from the initialized student. R-EMA uses the same Q as the student and is linearly equivalent to EMA of effective G. F_prev is shared read-only rather than duplicated. Current-L prototype initialization does not consult history.

A successful step performs one optimizer update, one native constraint application, one EMA update, then commits pending prototypes, scheduler, scientific step and sampler cursor. A skip commits none of these. Synthetic skip tests exercise that transaction; real AMP/GradScaler overflow behavior is still unqualified. No old teacher, current-L histogram queue or persistent SWD sample bank is retained.

Checkpoints contain complete student and EMA state (including F/Q/R/A/B), optimizer/scheduler/scaler, prototype values/support, torch and Python RNG, stateless stream scheme, provider read counters, sampler cursor, epoch/successful step, probes/scales/fallbacks, latest diagnostics and operator counters. The provider used here is deterministic by cursor/namespace; a real provider must restore its sampler/worker state as part of binding.

Save writes a temporary file, fsyncs, atomically renames it, fsyncs the containing directory and writes a receipt. A failed pre-rename checkpoint leaves the previous committed state authoritative. The model state file itself is authoritative if receipt writing is interrupted after rename. No hash of historical model tensors is required. Synthetic tensor loading is explicitly marked; it is not a safe interface for untrusted pickle files or an authorized real checkpoint loader.

An append-only physical ledger records each executed optimizer operation. Recovery replays every uncheckpointed tail from the last committed model state; it does not advance because a log line exists. Separate counters expose physical operations, committed scientific steps and retry/uncommitted overhead. The tests inject before/after optimizer, before/after EMA and during checkpoint writing and compare recovered full states with uninterrupted execution.

Stage 2 must consume its own family/config/seed/order stage-1 checkpoint identity. The common source may be reused only by proven source seed/provenance. The controller checks lineage; it cannot select another arm's warmup, EMA or optimizer. It seals final student state, not EMA or best epoch, and disposes of the previous trainer before creating the next EMA.

Deployment is one parent student plus a single d² frozen transform. Independent-process synthetic evaluation loads this student state only; no Q, prototype, optimizer or repair loop is required. Parent adapter history follows its own contract and is not claimed constant-sized for an unknown real parent.

The current controller is finite and synchronous CPU synthetic code. There is no remote job submission, background monitor or automatic phase E expansion.

## R2 replacement/addendum: resume and numeric transactions

Stage resume uses Model.for_resume (no eigensolve) and StageTrainer.for_resume (no stage_entry, probes or prototype initialization). The parent restores its legal parameter whitelist and modes independently of new-adapter initialization. The checkpoint loads current nonzero A/B/R and saved EMA, probes and Q; it never resets them to stage entry. Independent-process stage-2 tests now cover all ten arms, not only first-stage recovery.

The checkpoint now carries resolved semantic options/digest, actual provider/manifest/split/stream/cursor convention, entry-source tensor identity, parent constraint metadata and named optimizer groups. Restore verifies actual constructed objects before state changes. output_path/log_every are the only nonsemantic option exceptions. Precision/device migration has no automatic exception.

The successful transaction order is optimizer invocation → native constraints → scheduler candidate → finite validation of student/effective weights/optimizer/scheduler/scaler/candidate EMA/prototypes → EMA/prototype commit → scientific step/cursor. A failed invocation may have partially changed the student or optimizer; it is physically recorded, scientific state remains uncommitted and a fresh checkpoint restore is required. The ledger's physical count means invoked optimizer operations, not validated scientific updates. Exact candidate EMA arithmetic is checked one parameter at a time to avoid constructing a third full model. The skip fixture models an AMP skip transaction; actual CUDA AMP is still NOT_RUN.

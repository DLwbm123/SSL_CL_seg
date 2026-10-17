# Siemens matched L-only control

Authorized 2026-10-16: physical GPU0, seed163 only, at most43,600 new optimizer calls.
The initial round retains its frozen configuration. No additional seed or independent-test access.

A uses the existing MAIN_STYLE_SSL_V1 Siemens source setup with lambda_U=0.
The shared loss engine returns after the labeled objective when this coefficient is
zero. LCTX still starts at step8720; all L sampling, image geometry/style, hierarchy,
network initialization, supervised objective, Adam, weight decay and poly schedule
stay unchanged. EMA updates and the initial labeled prototype read remain.
The only setup override is source_labeled_only=True; it is rejected for target stages.
No new source-specific architecture or loss is introduced. This is the matched
L-only version of the method, not the independent_mnms_v1 baseline.

B is the already completed V1 seed163 source (43,600 steps), with validation Dice
0.6581757651162471. Its sealed checkpoint and existing validation rows are reused.
No B training or new B validation inference is authorized by this protocol.

Exactly one final A train-L evaluation, one final A validation evaluation, and one
B train-L evaluation are scheduled. Both use whole stored384×384 volumes, batch2,
no augmentation, student deployment, class Dice, ED/ES then patient averaging.
Only the original10 training-L and23 validation patients are admitted. Class and
phase summaries, paired patient differences and train/validation gaps are reported.
Primary effect is validation(B−A) in percentage points. The +/-1pp descriptive band
is not a statistical significance threshold. No best-checkpoint selection.

Synthetic CPU checks and one real initial-loss/finite-backward preflight consume
zero optimizer calls. The latter must match the historical first supervised loss
2.063466310157642 within1e-5 before formal work. A fresh trainer is then constructed,
so engineering counters/gradients do not enter training. A hard optimizer pre-hook
and physical ledger enforce43,600 calls. No two-case overfit or engineering optimizer
updates are added to the authorized budget. Checkpoints are saved every2000 calls
and at the fixed final; failed execution stops and records an incomplete outcome.

Historical B was trained across different GPU/software environments before migration;
GPU0 A runs on the authorized current host. Thus identical algorithm configuration
and first objective are checked, but bitwise trajectory equivalence is not claimed.
The single selected seed and repeatedly viewed validation set support development
diagnosis only. They do not establish a general or independent-test SSL benefit.

Run/log/checkpoint/cache paths remain on the verified NFS mount. A neutral entrypoint
and environment-provided settings keep project/identity details out of process argv.
The user subsequently requested monitoring every30 minutes and engineering recovery.
The heartbeat may repair engineering failures within the authorized cumulative budget;
new scientific rounds, additional seeds and tuning require an explicit new scope.
Attempt1 failed after1 physical optimizer call because deterministic CuBLAS lacked
CUBLAS_WORKSPACE_CONFIG. Its log, ledger and initial checkpoint are retained. The
neutral bootstrap now sets :4096:8 before importing torch. Recovery budget handling
passed three synthetic CPU checks.

User amendment on2026-10-17: allow one replay call, so this round has a cumulative
physical cap43,601 and a scientific endpoint43,600. Restore the original step0
checkpoint into a create-only second attempt; charge the failed call to the budget.
The user also authorizes subsequent result-guided controls using seed163 only, with
130,800 total physical optimizer calls across this round and all follow-ups, including
failed and replayed calls. After this round consumes43,601, at most87,199 remain.
Freeze each follow-up's rationale, configuration and physical allocation before launch;
retain the same data boundaries and physical GPU0. This does not revive old V2 runs.
On actual completion, publish only sanitized code/aggregate results/report
to the project repository; private patient rows, logs, paths and weights stay private.

# CIST V0.1 execution status

RUNNING_VERIFIED, observed 2026-09-11T19:27:39.914301+08:00. This is a launch delivery, not a completed experiment or a scientific conclusion.

- Branch: codex/cist-v0-1
- Frozen execution source: 76afab45389d217196942ccdff3efcbac86a9c10
- Immutable starting result: f54e793ce3a699ab48a21ce2c2d244437fec67fb
- Run: cist_v0_1_20260911_01
- NAS root: /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/cist_v0_1_20260911_01
- Detached parent PID: 2302171; four actual GPU workers on4/5/6/7.
- At this snapshot: 1317 committed formal updates across four running tasks; no immediate failures. This snapshot is not ongoing monitoring.
- Authorized fixed queue: six arms, seeds61/62/63, two domain orders,36 targets and95400 formal updates. No new source or baseline training.

Local CPU and server CUDA each passed17 regression checks on the exact execution source,26 synthetic updates per backend. The original test attempt used identical feature contexts and failed a conditional-variation assertion after9 synthetic updates; it remains in launch_evidence/qualification_attempt_local_01. Only the test input was corrected. Total synthetic qualification61/128; real smoke24/24; all smoke controllers discarded. Real smoke peak allocated 340.90MiB, peak reserved 450.00MiB, frozen minimum free memory 1536MiB. Actual startup GPU footprint678MiB per worker. No existing process was stopped or changed.

Six original SRC_CE receipt/student identities and their cached V/N bases, plus18 existing baseline evaluations, are bound in SOURCE_BINDING and BASELINE_BINDING. Those steps opened no image or GT assets. Real smoke used only the permitted current labeled/unlabeled training interfaces and no val GT. The existing negative LCTX terminal is preserved.

The runner is experiments.lcrseg.cist_v0_1.execute.execute. It invokes one common training loop and then the fixed final evaluator per task. Entrypoints and arguments are passed through R_ENTRY environment payloads via the existing NAS wrapper; visible parent and worker argv were checked to contain no method/project names. Runtime is independent of this conversation and the SSH connection.

Logs: parent.log and logs/{task}_train.log / logs/{task}_eval.log. Exact resumable controller/Adam/RNG states, current uncommitted solve evidence and final deploy students live under tasks/. Immutable full source weights are referenced by hash. No image-specific controller library is maintained.

On completion, the fixed runner seals36 students and emits public_results/FINAL_REPORT.md, TERMINAL.json, paired effects, class costs, patient-delta distributions, bootstrap intervals, memory/compute, mechanisms and NAS archive evidence, then stops. The final primary decision is mean ISO_COND_LU minus F_CONV Final, with no historical conjunctive safety veto. No added seed, rank, method or domain is triggered by the scores. Public completion delivery follows verification when completion is next checked; no automatic conversation monitoring was created.

Public delivery here includes source, fixed protocol and aggregate qualification/startup evidence. Source weights, images/GT, raw per-case scores, RNG/checkpoints and runtime logs remain private on NAS. No completed accuracy result is claimed.

# Startup and recovery status

Recovery attempt2 started on2026-10-17, physical GPU0, seed163.
The original failed attempt remains intact and consumed1 physical optimizer call.
The recovery restored the original step0 checkpoint with unchanged science settings.
CUBLAS_WORKSPACE_CONFIG=:4096:8 is set before torch import.

The startup check confirmed the recorded process birth, neutral full command line,
GPU process presence, passing zero-update preflight, and33 physical updates in the
new attempt with no failure receipt. This passed the former step2 failure point.
The first labeled loss was2.063466310157642, exactly matching the historical value.
This is startup verification, not a completed experiment or a comparison result.

This round:43,600 scientific steps, at most43,601 cumulative physical calls.
Entire authorized study:130,800 physical calls including failures and replays,
seed163 only; at most87,199 remain after a43,601-call first round.
A heartbeat checks every30 minutes and autonomously handles engineering recovery,
result analysis, bounded subsequent controls and sanitized GitHub delivery.
Each subsequent control records its hypothesis, configuration and allocation before
launch. The current round retains all settings except lambda_U=0.

Private paths, PID receipts, logs, patient rows and checkpoints remain outside Git.
Historical B used different hardware/software during training; the comparison
cannot establish a hardware-matched or independent-test effect.

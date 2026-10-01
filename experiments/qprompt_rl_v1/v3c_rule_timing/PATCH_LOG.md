# Pre-update engineering correction

Attempt1, commit bc1929fa0b740a845fd6ef134daca6f87e24ff5d, stopped during initial teacher construction because the launcher omitted CUBLAS_WORKSPACE_CONFIG. PHYSICAL_LEDGER did not exist and operation counts were empty: zero optimizer attempts, zero successful student updates. No source or target checkpoint exists to resume.

The guardian now supplies CUBLAS_WORKSPACE_CONFIG=:4096:8 before worker import, retaining native deterministic algorithms; it also fixes PYTHONHASHSEED=0. No training equation, method, seed, phase, horizon or budget changes. Preserve attempt1 source, configuration, process exit, logs, operation counters and RUN_LOCK. Attempt2 reuses the original SESSION/deadlines; no learned state is discarded because no training update occurred. A repeated occurrence of this same error stops further recovery.

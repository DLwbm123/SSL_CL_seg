# Readout engineering repairs

## 2026-10-07: compare seal order within one clock domain

The first readout invocation from commit `99ab8db` failed before creating its results directory or making any optimizer call. Its guard compared NAS file modification times against the training host's serialized `time.time()` value. The lock file's NAS mtime was 8.92473793 seconds ahead of its recorded writer time. Thus the final training receipt appeared 8.82373810 seconds after the writer timestamp, although it was 0.10099983 seconds before the lock file on the same NAS clock.

All 20 training receipts and final checkpoint files precede the lock's NAS mtime; the development result file follows that lock. The fix compares file order using NAS timestamps throughout, while comparing the original decision and lock's recorded timestamps within their common writer clock. The original single-worker code also seals all endpoints before starting evaluation. No checkpoint, label, reward, result, or scientific threshold changed.

Recovery preserves the failed script and log, uses a separately named patched script, and keeps the original run's artifacts intact. Failed invocation cost: 0 student updates, 0 actor updates. The stdlib self-check passes. The specific 20-endpoint order check was re-evaluated against the common filesystem clock; the source campaign is not rerun.

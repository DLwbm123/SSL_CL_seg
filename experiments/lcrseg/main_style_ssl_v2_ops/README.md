# Saved-artifact early-stop audit

`early_stop_audit.py` reads an already stopped study through `RUN_ROOT` and writes create-only JSON to `AUDIT_OUTPUT`. It requires the original private receipts/checkpoints and PyTorch; it does not start training, data inference or a recovery run. The executed audit used CPU only and one thread. Assertions check sequential canonical ledgers, checkpoint receipts/readability, terminal accounting and engineering costs.

The published anonymous report is an allowlisted summary of this private audit, with terminal operator classifications and recorded missing costs. The audit JSON public-delivery PENDING field records the pre-publication snapshot; actual remote commit/access verification is stored in the study's terminal delivery receipt after push. Raw worker FAILED/KeyboardInterrupt receipts remain distinct from the USER_TERMINATED operator classification.

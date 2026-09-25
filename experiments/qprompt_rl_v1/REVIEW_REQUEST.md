# R0 review entry

Status: **STOP_AWAITING_EXTERNAL_CODE_REVIEW**. R0 code, exact M1 byte migration, target environment, and bounded CPU acceptance are complete. No external code review has been claimed.

Please review the paper-to-code choices, especially U-Net query head, ViT last-block placement, Eq.13 selected surrogate, true background versus no-object, and matched-query loss. The supplied PDF SHA differs from the plan and was explicitly accepted by the user for this R0. The source-only branch is `codex/qprompt-rl-v1-r0`; `CODE_MANIFEST.json` binds the current file bytes. The exact commit and private migration-manifest digest are reported in the final handoff.

Review the known limits: per-user quota ceiling was not independently available; the venv inherits package files from a shared read-only Python installation; top-level HDF5 key checks and byte hashes do not prove medical semantics; no CUDA model test or real-data training occurred. The migration manifest and source/target paths are private. `R1_PLAN.json` and its proposed budgets are not execution approval. R1, R2, and R3 remain disabled until the user separately approves an exact commit and phase after external review.

# R0 review entry

Status: **BLOCKED_MISSING_TARGET** until the new server SSH alias and remote-home absolute path are confirmed. R0 is still in preparation; no external code review has been claimed.

Please review the paper-to-code choices, especially U-Net query head, ViT last-block placement, Eq.13 selected surrogate, true background versus no-object, and matched-query loss. The supplied PDF SHA differs from the plan and was explicitly accepted by the user for this R0. The source-only branch is `codex/qprompt-rl-v1-r0`; `CODE_MANIFEST.json` binds the current file bytes. The exact local commit is reported with the handoff and may change as blocked R0 checks are completed.

Remaining R0 checks: target identity/mount/quota/inodes and GPU queue policy; official weight acquisition/hash; offline import closure; exact M1 manifest and source payload SHA; SSH dry-run/transfer/target SHA receipt; environment lock; bounded CPU model suite and deployment/prefix checks. Until then `R1_PLAN.json` and its proposed budgets are not execution approval.

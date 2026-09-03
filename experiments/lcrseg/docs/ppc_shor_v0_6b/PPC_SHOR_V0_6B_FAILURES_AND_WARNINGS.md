# Failures and warnings

- Scientific status: `FAIL_PPC_SHOR_CURRENT_SAFETY`.
- Failed gate: current safety. All five registered safety limits were exceeded; exact values and
  thresholds are recorded in `PPC_SHOR_V0_6B_STATUS.json`.
- A pre-reservation engineering run stopped before model forward because the isolated worktree
  lacked the separately maintained official JASCL reference checkout. It had zero reservation,
  forward, domain, and GT reads. The environment dependency was then linked read-only without a
  source change; the only scientific attempt is the completed reserved evaluation.
- All Fundus rows are development-consumed; no external claim is permitted.

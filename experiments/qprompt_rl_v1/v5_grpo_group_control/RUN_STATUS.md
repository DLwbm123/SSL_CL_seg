# V5 startup delivery

Verified: 2026-10-04 18:24:24 CST.

- Status: RUNNING — P1 calibration, after engineering qualification PASS.
- Run: `v5_group_rl_20261004T102000Z`.
- Execution source: `77cf65db6802ebbeaf2cd6b7c097fc11f0ebc2cc`; baseline `90eacbacca00d3c03e1296d5c7e6ac22a047df3b`.
- Physical GPUs 5/6/7 each have an active common-initialization calibration worker; observed student steps 50/100/75. These are startup observations, not a live dashboard.
- Native qualification: 690 updates. Synthetic qualification: 65,900 optimizer calls, separately charged.
- Pilot budget: 890,400 student updates plus P0. Fully promoted main budget: 2,758,900 plus P0. Conditional Clip-Higher: at most 143,100 extra.
- Primary remains GRPO_FS. Controller seeds402/403 and five-source confirmation can run only after the registered pilot gate passes. Audit never gates promotion.
- No deadline, automatic retry, resampling, test access, hidden-U label access or monitoring automation. Existing GPU jobs untouched.
- A failed zero-student-update setup is preserved and its synthetic cost retained; corrected launch continues the original budget. See QUALIFICATION_REPORT.md.
- Pilot, reward scale and confirmation results are pending. Full completion and final public result delivery are not claimed.

Private checkpoints, role identities, state panels, raw logs and case scores remain on NAS. This public snapshot contains source, frozen design, budget, anonymous qualification metrics and startup receipt only.

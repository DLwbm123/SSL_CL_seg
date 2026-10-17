# GPU2 source control startup

Seven CPU tests passed, covering empty-mask behavior, class balancing, detached pseudo
targets, finite gradients, and unchanged supervised loss/gradients across warmup.
The native first-batch loss was2.063466310157642, matching the historical value;
backward and initial checkpoint restoration passed with zero optimizer updates.
Startup health confirmed the PID birth, GPU2 process and159 actual optimizer updates
without a failure receipt. U starts at scientific step8,720 as originally configured.

Storage mount and write/read probe passed on the authorized data volume. All code,
data, environment, cache and run outputs are on that volume. A full historical copy
is in progress; existing data/runtime are reused for training. Source copies remain.

The initial interpreter argv used a project-named runtime symlink, violating the
neutral-command convention. The active process is retained rather than restarted
solely for display; subsequent launches must use the resolved neutral environment
path. No claim of compliant initial full argv is made.

Candidate cap43,600; reserve43,601 for the concurrent L-only round; total cap130,800.
No final metrics or cross-paper comparability has been established at startup.

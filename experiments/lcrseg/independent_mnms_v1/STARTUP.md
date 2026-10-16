# Background startup, 2026-10-16

Formal Siemens independent training started at 13:19:22 UTC on physical GPU0.
Seeds163 and164 run sequentially,200epochs each,7800 optimizer updates per seed.
The first startup check confirmed seed163 epoch10 /390updates, finite training
loss0.962547 and a neutral visible process command. This is startup evidence,
not completed training or validation performance.

Five CPU tests passed. The first GPU engineering attempt consumed64updates
and failed the withdrawn arbitrary short-fit threshold. The subsequent
minimal finite-update/exact-checkpoint-replay check passed with3updates;
actual GPU engineering total67, separate from the15600formal budget.
No retries, extra seeds, extra cohorts, independent-test access, or new
recurring monitor were started. Private logs and checkpoints remain on NFS.

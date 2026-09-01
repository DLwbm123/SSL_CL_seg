# PRES-JASCL failures and warnings

The single authorized formal attempt is `BLOCKED_PROTOCOL_OR_LEAKAGE`; it is not an adjudicated feasibility pass or scientific failure. D5 is false because the pinned official JASCL import changed `torch.backends.cudnn.benchmark` to `True` after the registered backend configuration.

All registered compute cells completed, the child exited zero, all 12 model/checkpoint guards passed, model and checkpoint bytes stayed unchanged, and all optimizer/autograd/backward/parameter-gradient/training counters remained zero. The raw D1-D4 values are retained only as non-adjudicable diagnostics.

Commit `0c983666da4458d96450ef8121427a823cbaa3b4` contains a minimal PRES-local guard and a passing zero-forward reproduction. It has not been used for a second formal attempt, and no second attempt is authorized.

Case IDs, paths, raw descriptors, predictions, per-case scores, prototypes, checkpoints, and private manifests are intentionally excluded from the public branch.

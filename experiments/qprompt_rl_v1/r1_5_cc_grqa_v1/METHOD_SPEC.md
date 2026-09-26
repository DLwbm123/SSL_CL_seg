# CC-GRQA V1 frozen implementation

Only detached query assignment changes. A count-state dynamic program maximizes summed FP64-converted detached cosine scores, retaining the lexicographically smallest class-index tuple in query order on exact score ties. No tolerance-based score ties. Each image is solved independently; supported-class bounds are 3–6 / 4–8 / 12 for 3 / 2 / 1 supported classes. Unsupported classes are excluded; zero support returns graph-connected zero. Small-K exhaustive enumeration checks global optimality and tie order.

All R1 GRQA mathematics, including student-selected reference probabilities, detached reward/advantage, population standard deviation, clipped ratio and selected Eq.13, remain unchanged. Original-routing values and gradients are tested exactly against R1. A0 is Q1; A1 is Q3; A2 ramps original GRQA; A3 uses CC; A4 uses CC plus the ramp. Only GRQA weight ramps: 5 min(1,s/300), s=1..1000.

Reuse the R1 model, augmentation, optimizer, bank initialization, atomic checkpoint, and evaluation functions without editing the old directory. Development independently restores each old frozen prefix and its optimizer/scheduler/RNG, then initializes phase-local banks and references. New seeds use identical named RNG construction with 261 replaced by 262 or 263. BF16, TF32-off and deterministic warn-only behavior remain as R1.

Gradient diagnostics use autograd.grad on a separate diagnostic forward, then a fresh training forward/backward; autograd.grad does not populate .grad. Preserve RNG around diagnostic computation. Forensic leaves model/bank/reference fixed and performs no optimizer calls.

The prompt leaves “not clearly negative” class integrity numerically unspecified. Before results, resolve conservatively to nonnegative equal-domain rim and cup A4−A0 for each backbone, in addition to the explicit per-cell/class floor −0.01. No tuning based on forensic or validation. Confirmation is enabled only after the complete development matrix meets every frozen gate.

Shared server GPUs are allowed when measured/qualified memory plus margin fits. Raw logs, private paths, per-case schedules and weights stay on the verified NFS run root. Public reports bind actual training commits; historical R1 artifacts are read-only.

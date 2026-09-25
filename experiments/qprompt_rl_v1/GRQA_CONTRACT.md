# GRQA contract

Bank P contains three per-class normalized vectors. Initialize/reset it from a complete deterministic current-L pass at each phase start; never carry it across domains. For each step, form differentiable current-image class prototypes from normalized pixels and GT, compute loss against a detached bank snapshot, then update bank only after successful optimizer step. The reference is an independent frozen EMA network; update it only after successful step.

For each image, cosine query×P gives top-1 class and reward. Group queries by top-1 class within that image; detach class, reward, advantage, and reference. Population std with epsilon1e-6 gives zero advantage to singleton/constant groups. Current query probabilities keep gradient. Main regularizer is the paper-selected Eq.13 surrogate `expm1(log_ref-log_cur)-(log_ref-log_cur)`; full categorical KL is diagnostic. No supported prototype yields graph-connected zero. Nonfinite values stop.

R1 arm weights and scheduling are frozen in `R1_PLAN.json`; R0 does not launch training. Deployment keeps the student backbone and query head, and removes P/reference. Equality before/after removal requires a real model test before deployment can be reported.

# Implementation delta

New V5 package reuses V4 native source/Trainer/Provider/feature equations/advance/export and independent evaluator. Adds complete fork bookkeeping, pure-softmax actor, reward-free initialization softening, four-trajectory grouped updates, separate critic optimizer, fixed calibration scale, cached same-entry U0 references, global val barriers and preregistered conditional stages. No baseline engine edits.

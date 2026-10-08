# V100 frozen policy expectation and argmax

V99 failed matched deployment comparisons despite higher training returns. Use its six already frozen actors with no new fit. Extract their probabilities at ENTRY100 and every first-action PREFIX200; seal all probabilities and deterministic argmax choices before score lookup. Compare exact T1 expected two-action outcomes, a fixed lowest-ID argmax rule, and the original V99 categorical sample. Every seed and control is retained.

Expected outcomes integrate144conditional trajectories per context; they do not assume independent action marginals or apply a retention hinge after averaging. The original72selected actor probability vectors and all36sampled choices must replay exactly. No student update, actor update, image query, new data or temperature search.

Primary is fixed-argmax expanded RL against matched fixed-argmax CE/initialization, training-global/time, uniform expectation and historical continuing CE/pooled RL. All continuous differences and paired seeds are reported; no independent confirmation is claimed.

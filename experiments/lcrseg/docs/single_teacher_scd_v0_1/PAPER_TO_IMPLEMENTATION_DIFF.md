# Source and implementation distinctions

The referenced [Pandey et al. paper](https://arxiv.org/html/2605.20538v1) was inspected at sections 3.1, 3.2 and Appendix H. It describes gradient-scaled classifier noise and case-based PAS, with scenario-dependent freezing and prototype replay. A here is a budget-matched GAS/PAS Mean-Teacher component reference with current-only prototypes and no prototype replay. Paper performance numbers are not thresholds here.

The official classifier is imported from JASCL commit `3c93ca70784fc3a1d2a887f8d7dce5af6bc75f53`. Its functional convolution omits padding; the existing adapter interpolates logits with align_corners=True. Constants, weight normalization, temperature10 and stochastic noise are unchanged. Unused sigma remains as upstream. GAS is a `.data` state; we explicitly disable its requires_grad flag and copy supervised-only squared gradients after Adam. A uses the frozen helper to average GAS along with the other parameters.

This protocol uses fixed-class Fundus, new common supervised stage0, all originally learnable body weights, 100 epochs, fresh stage Adam and current-only prototypes. B–E use a single weak-to-strong student view, not complete UniMatch (no dual strong, feature perturbation or CutMix). The SCD projection, schedules, augmentation, normalization and practical gates are this protocol's experimental choices, not paper claims.

FP64 teacher probabilities are normalized before common eps smoothing to remove representational sum error. Every C/D/E uses the same operation. Target construction detaches p from the exact logits used in the training KL. The local constraint is about the logit inner product with CE direction, not parameter gradients, Adam, Dice or old-domain risk.

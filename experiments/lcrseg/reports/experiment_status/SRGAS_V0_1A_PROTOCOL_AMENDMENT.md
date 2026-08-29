# SR-GAS V0.1a protocol amendment

**Status:** `SRGAS_V0_1A_PROTOCOL_AMENDMENT_FROZEN`  
**Hypothesis ID:** `SRGAS_V0_1A_H1_RELATION_TO_CLASSIFIER_CLASS_SPACE_BRIDGE`

The original `HARD_STOP_NO_SHARED_SEMANTIC_FEATURE` was correct and remains
permanent evidence. No SR-GAS training has previously run. V0.1a does not add a
64-to-16 mapping, move either head, or change the U-Net architecture.

The sole amendment is A5 sensitivity in the shared, fixed Fundus class space:

```text
frozen previous model + historical anchors -> q_old_relation
current clean segmentation logits -> bilinear downsample -> p_current
L_R2C = KL(stopgrad(q_old_relation) || p_current)
S_R2C = grad(L_R2C, classifier_weight)^2
```

`L_R2C` is a sensitivity proxy only. It is not added to the training objective,
does not update the old model, historical anchors, or projection head, and
introduces no trainable parameter. A5 combines unit-mean normalized supervised
and R2C classifier sensitivities at the frozen `0.5/0.5` weights, then uses the
unchanged inverse-minmax rule and noise variance `0.1`.

All unrelated V0.1 settings, budgets, run names, gates, and conditional order
remain frozen. The original model-path audit and blocker files must not be
modified or deleted.

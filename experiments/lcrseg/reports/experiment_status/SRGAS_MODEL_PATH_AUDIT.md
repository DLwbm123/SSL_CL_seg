# SR-GAS V0.1 model-path audit

**Status:** `HARD_STOP_NO_SHARED_SEMANTIC_FEATURE`

## Located paths

- Segmentation classifier: `model.segmentation_head`, a Conv2d with parameters
  `segmentation_head.weight` and `segmentation_head.bias`.
- Relation projection: `model.projection_head`, two Conv2d layers with GroupNorm
  and ReLU.
- Anchor-update feature source: normalized `SegModelOutput.relation_features`
  returned by `model.projection_head(dec3)`.

The frozen forward is:

```text
dec3 [B,64,H/4,W/4] ──> projection_head ──> relation_features [B,128,H/4,W/4]
          │
          └─> dec2 [B,32,H/2,W/2] ──> dec1 [B,16,H,W] ──> segmentation_head
```

## Runtime and gradient evidence

For `[1,3,384,384]`, forward pre-hooks showed:

- projection-head input is exactly the `dec3` tensor `[1,64,96,96]`;
- segmentation-head input is exactly the `dec1` tensor `[1,16,384,384]`;
- the two head inputs are different tensors with different storage.

`L_sup` produced finite nonzero gradients through `dec1`, `dec2`, and `dec3`.
`L_rel` produced a finite nonzero gradient through `dec3`, but no gradient
through `dec2` or `dec1`.

## Gate decision

`dec3` is the last common computational ancestor, but it has 64 channels while
the classifier weight is `[C,16,1,1]`. The registered A5 formula requires
`S_R[d]` to broadcast directly to classifier input channel `d`; a 64-channel
vector cannot be broadcast to the 16-channel classifier weight. `dec1` has the
correct 16 channels but is not on the relation-head path. Projected relation
features have 128 channels and are not on the classifier path.

Therefore no existing tensor satisfies both the shared-path and channel-space
contracts. Selecting a different layer, changing the projection head, or adding
a 64-to-16 mapping would be an unregistered architectural choice. The protocol
requires a hard stop before implementation, tests, or training.

# Architecture and coordinates

`ParentBridge` specifies features after the final nonlinearity, native affine readout, native supervised reduction, A/B groups, constraints, stage entry/exit, coordinate maps and optimizer groups. The synthetic implementation has three ReLU-separated 1x1 adapters, delta W = B A, a frozen stem and frozen 3x3 readout with bias, padding 0, then bilinear output resize with align_corners=True. It is explicitly unconstrained and is not an inferred KI implementation.

F1/F3/F4/F5 insert `F_prev (I + Q R Q^T) h` immediately before that native readout. Q has shape d by k and R has shape k by k. Q is obtained once per stage by CPU FP64 eigendecomposition of sum_delta `(C_delta F_prev)^T Pi (C_delta F_prev)`, with class centering. Ratios clamp to 1..d-1; d<2 fails. The zero spectrum uses the first k identity columns with `proxy_degenerate=true`. Energy and cut gap are diagnostics, not performance gates. Repeated eigenvalues are compared by subspace projection.

This is a channel sensitivity proxy, not the complete spatial patch row-space, and supplies no zero-forgetting or optimal-generalization guarantee. A full readout row-space projection can leave its input gradient unchanged; that identity is tested and is not presented as a method.

R starts at zero. Stage sealing composes `F_next = F_prev G`, not the opposite order. The native parent adapter merges take place according to the synthetic parent contract. One frozen d by d transform remains in Deployment; Q/R, the EMA, prototype cache and optimizer are absent. F2 and B0/B1/B2 use identity F; B3/B4 use dense G with Q=I as explicit capacity controls. No framework inherits another framework's transform.

The synthetic feature grid is in normalized full-image coordinates. Output-to-feature maps use nearest interpolation for labels/masks and align_corners=True bilinear interpolation for continuous fields. The 3x3 valid convolution and native output resize remain unchanged. These are explicit synthetic coordinate semantics, **not verified real-network alignment**. Binding the actual receptive-field/crop/resize mapping is required before real data can be read.

F3 first maps the two source uncertainty maps through the complementary source mask into each mixed image, then through the bridge onto its feature grid; D acts exactly once on residual coefficients R Q^T h. L is never filtered. Neighboring receptive fields still mix source information, so this is not an exact output-pixel Jacobian projection.

F4 repairs teacher features in the coordinate before F_prev using d of shape B by k by 64 by 64, fixed bilinear interpolation with align_corners=False, and the same frozen native readout. The per-image trust radius uses feature RMS; each low-resolution coordinate vector is projected into that ball. Convex-combination interpolation preserves this norm bound. This is an explicit scalar-radius choice within the V1 trust-fraction contract.

F5 projects both clean-U student and current-L teacher `G h` by the same Q, before F_prev, normalizes vectors with epsilon 1e-6, and compares current-batch class-conditional empirical distributions. No feature queue or history prototype bank exists.

## R2 corrections

One complete channel eigensolve supplies both the selected basis and diagnostics. Model.for_resume creates legal shape placeholders without an eigensolve; the saved stage Q/spectrum replace them at restore. Explicit student/teacher/eval mode reaches both native features and readout. New NativeParentBridge delegates verified native hooks and a full named-parameter partition rather than manufacturing a real parent. Its synthetic conformance is not real-network qualification.

# Loss and gradient contract

The sole supervised authority is `ParentBridge.supervised`. Toy CE averages valid pixels; toy foreground Dice averages active images and rim/cup classes with smooth 1e-6. The real definition must be bound, not inferred from this fixture. Before warmup ends, L uses one native forward. After ceil(0.2 * stage_updates), LCTX uses two complementary forwards of different-patient, reversed-batch contexts and collects log-softmax by source. Each anchor pixel is supervised once. F2 adds CWMI to restored source probabilities. Native parent constraint loss is added once; its update projection runs once after the optimizer.

Active U uses complementary UL views, original/weak teacher U, and source-restored student log probabilities. L context in UL contributes no extra GT loss. Repeated physical U IDs are deduplicated before the forwards. B0/B3 never call the U provider. Ramp duration is ceil(0.2 * stage_updates), reaching full weight in that many successful active-U steps.

| Arm | L direct gradients | U objective | U direct gradients |
|---|---|---|---|
| F1 | parent allowed + R | PAS KL + lambda_JML JML1 | R only |
| F2 | parent allowed, plus scaled CWMI | PAS KL | B only |
| F3 | parent allowed + R | geometry-only KL, custom residual backward | R only |
| F4 | parent allowed + R | original PAS KL to repaired teacher | R only |
| F5 | parent allowed + R | PAS KL + lambda_SWD class SWD | R only |
| B0 | parent allowed | none | none |
| B1 | parent allowed | confidence-only KL | parent allowed |
| B2 | parent allowed | PAS KL | parent allowed |
| B3 | parent allowed + dense R | none | none |
| B4 | parent allowed + dense R | PAS KL | parent allowed + dense R |

KL sums classes and averages accepted pixels; teacher/mask are detached. Empty support returns connected zero. JML1 averages per-image/per-foreground-class 2*d/(s+d), excluding entirely invalid images and defining double-empty class zero. No softDice substitute or rejected-as-background labeling is used.

U features detach from parent for R-only frameworks, while frozen readout operations remain differentiable with respect to R. F2 uses only output-factor gradients for U. L and U autograd results are merged once by allowed parameter identity; an illegal U component is None even when an Adam/decay/L update changes that parameter overall. Actual parameter update norms are recorded separately from permitted gradients.

PAS uses teacher confidence strictly greater than threshold and teacher-feature cosine strictly greater than threshold. Missing current-L prototype falls back to confidence-only and is logged. Student confidence is absent. Current-L teacher statistics are calculated before the update; U uses the old cache; new estimates commit only on success. Features for prototypes are native readout inputs after F_prev G, with explicit bridge geometry maps.

F2 stage entry precedes optimizer/EMA initialization: eight L scale batches, two h-VJPs each; eight clean-L direction batches, one joint effective-weight VJP each. Probe objective is lambda_structure * scale * CWMI structure, without duplicate CE. The positive lambda changes objective/gradient magnitude; exact SVD directions can be invariant to positive scalar scaling and no distinct-direction claim is made. Only new B=0 adapters can receive A initialization with original Frobenius norm. Right-multiply an explicitly supplied hard free projector; a soft parent must retain its own soft rule. Rank insufficiency preserves the original legal A and reports fallback.

F3 D = 1/(1+kappa*u/(normalized_s+1e-6)); u is entropy/prototype uncertainty, missing prototype uses entropy. D is stop-gradient, identity in forward, and multiplies residual-position gradients once in backward. It is a custom optimization rule, not the derivative of the identity function or a scalar loss. Adam may offset gradient scaling.

F4 uses only d gradients, three iterations, step 0.05 after per-image gradient RMS normalization, L2 0.1. Original q/mask stay fixed. Shape_weight=0 or steps=0 exactly returns original q. Readout-only forwards and d-VJPs are counted, and nonfinite objective/gradient raises rather than silently dropping a sample.

F5 retains an extra clean-U student feature forward when lambda_SWD=0. It samples min(64,nL,nU), excludes classes below 8, uses 32 detached unit directions and empirical sorted squared W2; valid classes average equally, with connected zero for no valid classes. Only rim/cup are considered. Teacher directions/features are detached and the SWD stream cannot consume the shared augmentation stream.

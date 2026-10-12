# V18 REDistill reference screen: no native advancement

The frozen zero-training mathematical screen completed with MATH_EXIT=0 and exact-gradient checks passing. The proposed literal fixed-power reference did **not** satisfy its preregistered wrong-teacher attenuation criterion, so no native qualification or formal training is launched. This is a negative result for this reference-only adaptation, not a segmentation efficacy result or a general refutation of REDistill.

Execution source: ce31c21289ec7253248d9872aa6d1c2d68472532. Protocol and all cases are in [PREREG](PREREG.md) and [MATH_CHECKS](delivery/MATH_CHECKS.json); [exit receipt](delivery/MATH_EXIT.json).

| Fixed generated case | Power loss | Forward KL | Gradient norm / KL |
|---|---:|---:|---:|
| Wrong soft teacher | 29.372919 | 5.208943 | 20.334185 |
| Wrong one-hot teacher | 29.879567 | 5.298317 | 20.519711 |
| Aligned soft teacher | 0.032863 | 0.036690 | 0.865841 |

For teacher p, student q and the original fixed lambda=2/3, the adapted divergence is 0.9(sum p^(5/3) q^(-2/3)-1). Its student-logit derivative is 0.6(q sum(w)-w), w=p^(5/3)q^(-2/3). With a one-hot teacher k this multiplies the KL gradient by 0.6 q_k^(-2/3). The definitions follow [the original paper, Eq8–9 and Appendix Eq18–29](https://arxiv.org/html/2602.04677v1); the logit derivative and adaptation diagnosis are our calculations. Probability-space Newton influence is distinct from this gradient under the existing optimizer.

A global reference scaling alone would not alter a halfspace. Therefore the fixed mixed case places wrong and aligned pixels in the same teacher class: their gradient-norm ratio becomes267.112477 versus11.373800 for KL, changing the aggregate direction (cosine0.996469). The generated true-gradient dot after projection is0.015014154 versus0.015037500 for KL. These small synthetic differences do not establish patient performance. They show that the transplant can emphasize a confidently wrong teacher relative to an aligned pixel, contrary to the selected hypothesis.

Identity, empty support, detached teacher and five invalid/overflow cases pass. Actual cost:10 successful CPU kernel calls,10 CPU autograd VJPs,5 rejected inputs; zero optimizer updates, model forwards, GPU calls or patient access. No parameter/fixture search or retry occurred. Formal campaign cost stays403473; qualification optimizer totals stay960 CPU/237 generated CUDA/48 current-L. Unused native/formal allowances are not spent or transferred to retries.

The already authorized independent queue advances to CaRE-KD after verified public delivery. No lambda/sign/direction swap, additional REDistill probe or formal efficacy claim is made.

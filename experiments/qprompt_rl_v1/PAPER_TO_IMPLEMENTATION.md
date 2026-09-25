# Paper to implementation, R0

The cited pages are from the user-confirmed PDF in `SOURCE_PROVENANCE.md`. The table distinguishes paper definitions from research choices. CPU dynamic tests and official-weight integration are pending in `CPU_REPORT.md`.

| Paper definition | R0 choice / code | Divergence or missing specification | Check |
|---|---|---|---|
| p4 Fig.2, Eq.2–3: queries enter only final ViT block | `DINOv2Adapter.forward`, `qprompt/models.py` | ViT-S/14 rather than principal ViT-L; K=12 chosen here. CLS/image positional embeddings come from upstream `prepare_tokens_with_masks`; queries receive none. | Mock geometry test; official model pending |
| p4: two transposed-convolution upsamplers | `DINOv2Adapter.upsample` | 384 input reflect-padded right/bottom to392, then logits cropped to384. Label padding is ignore255 for image prototypes. | Mock geometry test |
| p5 Eq.4–5: normalized pixel embeddings, GT mean, EMA bank, image loss | `image_prototypes`, `PrototypeBank`, `image_alignment_loss` | EMA=0.9, reset per phase, current L only; no prototype for absent class. | Synthetic support/EMA test |
| p5 Eq.6–7: cosine similarity/top-1 reward | `grqa_loss` | Similarity temperature=1; missing bank yields graph-connected zero. | Synthetic test |
| p6 Eq.8–9: same-class query groups, population std advantage | `grqa_loss` | Groups remain per-image; epsilon=1e-6; singleton/zero-variance advantage=0. Advantage detached. | Synthetic test |
| p6 Eq.10–12,14: current/reference softmax, clipped ratio | `grqa_loss` | Independent EMA reference=0.99; selected class detached; clip=0.1. | Gradient/detach test |
| p6 Eq.13: selected k3 surrogate | `paper_selected_k3_surrogate` in `grqa_loss` | Log-space `expm1(delta)-delta`; full categorical KL is diagnosis only. Nonfinite loss stops. | Formula/distinction test |
| p7 Eq.15, 2/3 + 1/3 | R1 plan only | R0 has no scientific training launcher. λimg=10, λgrqa=5 from p10 Table 5. | R1 review gate |
| p5: Hungarian supervised query matching | `supervised_query_loss` | K=12, C=3 including real background, class 3 is no-object; class/BCE/Dice weights 1/5/5 and no-object 0.1 are chosen here. | Query semantics/matching test |
| No paper CNN query architecture | `UNetQuery` | Adds a single cross-attention/FFN query head over existing U-Net body; query head remains at inference. | Geometry test |
| No paper RL action controller | `RL_CONTROL_DESIGN.md` only | Separate R3 hypothesis, default DENY. | No R3 implementation/run |

The existing stochastic JASCL classifier is not reused. `UNetDense` uses a deterministic valid 3×3 readout plus legacy-style interpolation; its initialized weights and behavior are **not** claimed equivalent to the old stochastic head.

# SARD source and adaptation boundary

Read on2026-10-12: [CVF main paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Liu_Structure-Aware_Representation_Distillation_for_Tiny-Dense_Object_Segmentation_CVPR_2026_paper.pdf), sections3.1–3.3, Algorithm1 and section4.1. Official title: Structure-Aware Representation Distillation for Tiny-Dense Object Segmentation, CVPR2026, Xuesong Liu, Anke Xu, Wenbo Cao and Emmett Ientilucci. The official PDF was extracted successfully; earlier web-parser403 did not prevent retrieval.

Equations6–9 define a teacher-feature Sobel structure tensor J=sum_c grad(F_c)grad(F_c)^T, edge E=lambda1-lambda2, corner C=sqrt(lambda1*lambda2), and neighborhood dispersion D=mean_j ||F(j)-mean(F)||_2. Section4.1 fixes beta_e/beta_c/beta_d=2/1/1 and radius7, hence15x15. Equation5 normalizes spatial scores. Equations10–13 additionally use projected feature matching and a denoising objective; those are excluded here.

The [author repository](https://github.com/liuuuuuuxuesong/SARD/tree/627231ace0af873af22a761a6b5df495eb295138) at the inspected revision contains only README, LICENSE and .gitignore. Its README describes directories and commands that are not yet present in that tree. No author implementation was available for code parity.

Explicit local adaptation choices: use the same-capacity own-entry post-prefix final features, replicate padding and unscaled3x3 Sobel, single scale, class-wise score normalization to preserve V13 class mass, and a uniform fallback for a zero-score class. Apply weights to reference KL only. No learned projection, feature MSE, noise head, compression claim or full-paper reproduction claim. These choices are frozen before optimization, not inferred as undocumented author defaults.

---
title: "Labeled-Context Mixing and Source Scoring in Two-Domain Continual Segmentation"
author: "Working manuscript — authors to be supplied"
date: "13 September 2026"
---

# Abstract

We study how labeled-image context mixing and the choice of supervised source regions affect two-domain continual segmentation. Earlier development experiments found improved average final performance for an anchor-supervised context-mixing recipe, LCTX, relative to clean supervised training. However, those results used a different optimization space and do not isolate the value of source scoring against fully supervised mixing. We therefore fix a matched comparison using six existing source students, three optimization seeds, two domain orders, and the same fourteen trainable convolution tensors. A full-source mixing control uses exactly the same complementary inputs as LCTX while supervising all valid mixed-source labels. A second block halves only the target labeled set while keeping the source models and update budgets fixed. The package adds thirty target trainings and 79,500 optimizer updates, with no response vector–Jacobian products. **The new matrix is pending; this draft makes no claim of a positive source-scoring contribution.** We report historical findings separately, specify all paired comparisons regardless of direction, and identify the limits imposed by reused development patients and absent independent method comparisons.

# 1. Introduction

Sequential segmentation adaptation must balance learning the current domain with retaining performance on a previously learned domain. When only a small labeled target set is available, the apparent benefit of a proposed training mechanism can be confounded with the strength of its supervised augmentation recipe. A useful experimental question is therefore whether an apparent retention benefit persists when initialization, trainable parameters, current-domain inputs, and optimization budgets are matched.

Our starting point is an existing labeled-context mixing control, LCTX. Each labeled image is inserted into two complementary contexts constructed with another patient from the same labeled batch. Predictions are then collected back into the original source image before supervised scoring. This construction does not access unlabeled images and does not itself establish an additional semi-supervised learning benefit. Historical full-parameter experiments favored LCTX over clean CE and CE+Dice on average, but they did not provide a matched full-source mixing baseline in the restricted convolution space used here.

We address that gap with a fixed comparison between anchor-source scoring and supervision of all mixed-source regions. The latter is called FULLMIX in this paper. It is a CutMix-style segmentation control matched to this project's inputs, rather than an official reproduction of CutMix or Bidirectional Copy-Paste. CutMix was introduced as a regional mixing strategy for classification [1]. Bidirectional Copy-Paste mixes labeled and unlabeled examples and combines true and pseudo supervision [2]; our present target experiments use only labeled examples. Neither paper's reported performance is used as a numerical baseline here.

The planned contribution is a controlled empirical attribution study, not a presumption that LCTX is a new successful method. We ask (i) whether source-consistent anchor scoring improves over full-source synthetic-image scoring under the same inputs, (ii) how each mixing recipe compares with clean supervised training in the same parameter space, and (iii) whether the mixed-recipe comparison changes when only target annotation availability is halved. All outcomes remain reportable, including inconsistent or unfavorable ones.

# 2. Problem and protocol

We use split 0 with two fixed orders: RIM_ONE_r3 to Drishti_GS (O1), and the reverse (O2). The source stage is reused from the completed SIGN replication: one SRC_CE student per order and optimization seed 71, 72 or 73. Every new target starts from its own matching source student. Source optimizer state and EMA are not inherited. During target training, only the current labeled domain can be accessed; previous-domain training data, unlabeled images and evaluation labels are excluded.

The standard target labeled pools contain 16 RIM patients or 10 Drishti patients. Low-target-label experiments retain 8 or 5, respectively, chosen once from the original labeled pool by sorting the prescribed SHA256 key on domain and patient identity. Selection does not use scores, class area or image labels. All seeds and both low-label methods use the same domain-specific subset. Patient identities remain private; subset manifest digests are public. The source models remain unchanged, so these are target annotation sensitivity experiments, not fully low-label source-to-target sequences.

The architecture, raw three-class linear head, 384 by 384 inputs and output geometry follow the established implementation. Only the fourteen specified convolution weight tensors are trainable (438,192 scalar parameters); normalization, transposed convolutions and readout remain frozen. All targets use Adam with learning rate 0.001, betas (0.9, 0.999), epsilon 1e-8 and weight decay 4e-5, with polynomial power 0.9. There are 100 epochs with 21 updates per epoch for a Drishti target and 32 for a RIM target. These step budgets do not increase as the labeled set shrinks.

Each update reads two distinct labeled patients. Standard-label batching and random keys are unchanged. For the odd low-label pool of five patients, a shuffled cycle's final singleton is paired with that cycle's first patient. Both low-label recipes share this rule. Actual exposure counts are reported rather than treating nominal epochs as equal patient exposure. Current EMA is updated to preserve inherited training semantics; it supplies no pseudo targets and is not deployed. Deployment always uses the epoch-100 student, without validation-based checkpoint selection.

# 3. Existing LCTX and matched FULLMIX

Let $x_i,y_i$ be an augmented labeled anchor and $x_d,y_d$ the other patient's correspondingly augmented sample. The donor is obtained by reversing the two-element batch. Let $M_i$ be the original keyed rectangular mask and $n(x_d)$ the original donor-noise transformation. Both recipes use

$$x_i^a=M_i x_i+(1-M_i)n(x_d),\qquad x_i^b=(1-M_i)x_i+M_i n(x_d).$$

The rectangle spans two thirds of each image dimension, using the existing location sampler. Both recipes use clean supervised CE+Dice in epochs 1–20 and complementary inputs thereafter. Their actual model inputs and random keys are identical.

For LCTX, let $\ell_i^a$ and $\ell_i^b$ denote log-softmax predictions. The scored source prediction is

$$\ell_i^{\mathrm{source}}=\operatorname{where}(M_i,\ell_i^a,\ell_i^b).$$

The original supervised function applies CE and foreground Dice to $(\ell_i^{\mathrm{source}},y_i)$. Donor images contribute context, while their labels are not used as additional targets. The original source is the unit for Dice aggregation.

FULLMIX instead constructs integer labels

$$y_i^a=\operatorname{where}(M_i,y_i,y_d),\qquad y_i^b=\operatorname{where}(M_i,y_d,y_i).$$

A single call to the unchanged supervised function receives concatenated predictions $(\ell^a,\ell^b)$ and labels $(y^a,y^b)$. Label 255 retains its ignore meaning. The loss contains no extra clean-image or reconstructed-source term. CE divides the summed negative log probabilities by the number of valid pixels across the synthetic batch. Dice uses FP64 spatial sums, a smoothing constant of $10^{-5}$, equal foreground-class weights, and equal synthetic-image weights; wholly ignored images contribute zero under the original reduction.

Each patient acts once as anchor and once as donor. FULLMIX therefore has raw source supervision multiplicity two, compared with LCTX's anchor multiplicity one. With no ignore pixels, each occurrence has one-half the raw coefficient under the doubled CE denominator, rather than adding two unnormalized losses. This does not imply identical gradients or exact per-pixel matching in the presence of ignore pixels. Dice's nonlinear ratios also make source-image and synthetic-image grouping generally nonequivalent. The experiment compares the combined recipe change of donor supervision and Dice grouping; it does not identify either factor in isolation or mathematically imply a direction of benefit.

# 4. Methods compared and estimation

The standard-label block includes new C_CE (native clean CE), C_CED (native clean CE+Dice), and C_FULLMIX, plus C_LCTX, the six existing F_CONV+LCTX targets reused from SIGN replication. C_CE and C_CED share the same source states and trainable parameter set as the mixed controls. Their historical full-parameter analogues are not substituted for these new tasks. The low-target-label block newly trains C_LCTX_LOW and C_FULLMIX_LOW; it has no low-label CE/CED arm.

The predesignated source-scoring comparison is C_LCTX minus C_FULLMIX. We also retain C_LCTX minus C_CE, C_LCTX minus C_CED, C_FULLMIX minus C_CED, and C_LCTX_LOW minus C_FULLMIX_LOW. Standard and low target-label budgets are reported separately. Final equals (Incoming + Old)/2; absolute Forget is the source-stage score minus the final old-domain score and is not clipped. Scores are patient-mean foreground macro Dice, with rim and cup results reported separately.

We first average the two order-specific differences within each optimization seed, then average the three seed differences. Paired seed SD describes variation across these three blocks. For conditional patient intervals we use 2,000 bootstrap draws, with physical-domain patient weights shared across every method, seed, order and source stage. These intervals condition on the fitted models and repeatedly used development patients. They do not represent independent patient confirmation or the entire training-randomness distribution. Model selection can overfit finite evaluation criteria [3]; new optimization seeds alone do not remove cohort reuse.

Only engineering validity conditions can stop a task: incorrect initialization, unauthorized data access, nonfinite state, incomplete-state recovery, uncounted updates, or false artifact/metric claims. A difference below 0.005, an interval crossing zero, or an unfavorable seed/class does not prevent reporting. The historical experiment terminals remain unchanged.

# 5. Results

## 5.1 Historical development evidence

In the earlier full-parameter sequence study (seeds 61–63), T_LCTX minus T_CE and T_CED had average Final differences of +0.051042 and +0.055500. These observations motivated the present question, but arose in a different trainable space and cannot establish the matched source-scoring contribution. The old experiment's terminal rules and class costs remain in its original report.

Old-seed SIGN control exploration gave a paired Final difference of +0.00473157 versus F_CONV, with conditional patient interval [-0.00104338, +0.01051551]. The subsequent fixed new-seed replication yielded -0.00018359, with interval [-0.00650457, +0.00600149]. Neither interval excludes zero. The latter did not reproduce a positive mean estimate; this does not prove a zero effect, equivalence or noninferiority. SIGN's recorded 25,440 response VJPs and 38,160 response forwards also do not establish an efficiency advantage. Historical main-candidate results are not replaced by either control analysis.

## 5.2 Matched standard- and low-target-label comparisons

<!-- NEW_RESULTS_START -->
PENDING: fixed matrix has not completed.
<!-- NEW_RESULTS_END -->

The result table will retain all five frozen contrasts. Per-domain rim/cup effects, absolute forgetting, seed/order differences, conditional intervals, exposure counts and actual resource use accompany the primary table. No estimates are inferred from incomplete runs. The evidence for an incremental source-scoring benefit and any label-budget dependence remain pending.

# 6. Discussion

The matched mixing control is necessary to distinguish a useful augmentation recipe from a source-scoring contribution. If LCTX improves over clean supervision but not FULLMIX, the supported account is the value of a strong mixing recipe; a unique anchor-source advantage would remain unestablished. If it improves over FULLMIX, the evidence would support a local practical benefit for the combined source-scoring design under these exact settings. It would not by itself establish superiority over published continual-learning or semi-supervised methods.

A reversal or inconsistent estimate in the low-label block would require a budget-dependent interpretation. It would not authorize additional label fractions or seeds to rescue the conclusion. If the new comparisons are unfavorable overall, the package still supports an empirical analysis of strong baselines and the limits of local stability proxies, with its negative and uncertain results retained. No universal retention mechanism, causal decomposition between donor labels and Dice grouping, or zero-forgetting guarantee follows from the present design.

# 7. Limitations and conclusion

The study has two domains, three optimization seeds and repeated development-cohort exposure. We have not established an evaluation cohort that was uninvolved in all prior training, unlabeled input, initialization and model selection. A manifest role name alone does not prove that independence. We also lack a genuine third training domain and matched implementations of published continual-learning baselines. Those absences limit claims; they do not block the fixed attribution study or justify invented comparisons. No original KI recovery, new DPR/SIGN/CIST/SVD mechanism, response correction, old teacher or replay is part of this package.

The current draft defines a bounded study of source scoring and target annotation sensitivity. New numerical conclusions await the fixed matrix. Completing this package does not establish journal novelty, broad impact or acceptance readiness: those are separate publication criteria [4]. The evidence map and submission-scope document specify which claims remain unsupported.

# References

1. Sangdoo Yun et al. *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features*. ICCV, 2019. [Author paper](https://arxiv.org/abs/1905.04899).
2. Yunhao Bai et al. *Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation*. CVPR, 2023. [CVF paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Bai_Bidirectional_Copy-Paste_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2023_paper.pdf).
3. Gavin C. Cawley and Nicola L. C. Talbot. *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*. JMLR 11, 2079–2107, 2010. [JMLR](https://www.jmlr.org/beta/papers/v11/cawley10a.html).
4. IEEE TMI. *Key Criteria for Publication*. Accessed 13 September 2026. [Journal criteria](https://ieeetmi.org/key-criteria-for-publication/).

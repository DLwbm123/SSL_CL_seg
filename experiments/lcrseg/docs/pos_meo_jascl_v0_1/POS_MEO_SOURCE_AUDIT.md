# POS/MEO official source audit

**Status:** `BLOCKED_POS_SPECIFICATION_AMBIGUOUS`  
**Engineering classification:** `BLOCKED_SOURCE_AMBIGUITY`  
**Audited at:** `2026-09-01T03:26:42Z`

## Verified primary source

The ICCV 2025 paper is verified on the CVF Open Access landing page:

- Paper: *Two Losses, One Goal: Balancing Conflict Gradients for Semi-supervised Semantic Segmentation*, Rui Sun, Huayu Mai, Wangkai Li, Yujia Chen, and Yuan Wang, ICCV 2025, pages 20357-20367.
- Landing page: `https://openaccess.thecvf.com/content/ICCV2025/html/Sun_Two_Losses_One_Goal_Balancing_Conflict_Gradients_for_Semi-supervised_Semantic_ICCV_2025_paper.html`
- PDF: `https://openaccess.thecvf.com/content/ICCV2025/papers/Sun_Two_Losses_One_Goal_Balancing_Conflict_Gradients_for_Semi-supervised_Semantic_ICCV_2025_paper.pdf`
- PDF SHA-256: `f9b5510351a3ad1c2911ba20c2081a478300f471e665d595b4ae1af76438c2b7` (1,482,036 bytes, 11 pages).
- Landing-page SHA-256: `f2c631c6a048d97b40fa3bec4005e2f8a5bc1c010bf2bd0cd222d5e8d5353887` (6,222 bytes).

The source is frozen by canonical URL, byte count, and SHA-256. The PDF is not copied into Git.

## Supplementary, code, commit, and license

`supplementary_status = NOT_PUBLICLY_VERIFIED`. The CVF landing page exposes only the paper PDF. The conventional CVF `supplemental`, `supp`, and papers-directory supplemental URLs returned HTTP 404 at the audit time.

`source_code_status = NOT_PUBLICLY_VERIFIED`. Exact-title and exact-method GitHub repository/code searches returned no candidate. The public repositories of paper authors Rui Sun (`yuisuen`) and Huayu Mai (`mai556`) contain related segmentation projects but no repository or branch attributable to this paper. Those unrelated repositories are not used as POS/MEO specifications.

Because no official code repository is publicly verified, `official_code_commit = null` and `official_code_license = NOT_PUBLICLY_VERIFIED`. Blogs, third-party implementations, earlier project projections, PCGrad, and independently chosen Pareto formulas were not used.

## What the paper specifies

- `g_sup` and `g_unsup` are gradients of the supervised and unsupervised mini-batch objectives with respect to the student-network parameters.
- Equation 5 chooses nonnegative coefficients summing to one that minimize the squared L2 norm of their combined gradient.
- Equation 6 provides a three-case analytical coefficient expression for ordinary nonzero gradients.
- Equation 13 preserves the POS direction and replaces its magnitude with the norm of the equal-weight gradient.
- Coefficients are incorporated into one weighted loss; optimizer and scheduler remain those of the host SSL baseline. The paper reports SGD for the UniMatch V1 experiments, AdamW with weight decay 0.01 for UniMatch V2, and polynomial learning-rate decay.

The requested JASCL comparison prospectively retains JASCL's existing Adam optimizer, polynomial scheduler, PAS probability-MSE objective, augmentations, stochastic classifier, EMA teacher, and all stage semantics. Those are explicit cross-framework adaptations, not paper defaults.

## Unresolved critical specification

The paper and the available official web material do not define:

1. coefficient branches when exactly one gradient is zero;
2. a deterministic coefficient tie when both gradients are zero or the nonzero gradients are identical;
3. MEO behavior when the POS combination has zero norm, for which Equation 13 divides by zero;
4. the executable parameter inventory used by the authors, including treatment of trainable parameters with `None` gradients;
5. an official synthetic reference output against which implementation parity can be established;
6. numerical boundary tolerances, clamping, or epsilon rules.

These cases are mandatory in the requested engineering tests. Supplying any of them locally would be an unverified method choice, and omitting them would violate the required zero-gradient, deterministic-coefficient, finite-value, complete-inventory, and official-parity gates.

## Hard stop

The source gate is therefore `BLOCKED_POS_SPECIFICATION_AMBIGUOUS`. No POS/MEO implementation, preregistration, execution authorization, private checkpoint tensor read, real model forward, optimizer step, or training run was performed. Unblocking requires an attributable official implementation/supplement or an attributable author specification that resolves the cases above.

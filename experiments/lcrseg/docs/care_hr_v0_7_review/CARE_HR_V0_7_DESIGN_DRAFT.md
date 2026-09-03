# CARe-HR V0.7 design draft

- Status: `BLOCKED_AWAITING_EXTERNAL_CODE_REVIEW`
- Draft state: `DRAFT_NOT_REGISTERED`
- Training authority: `NO_TRAINING_AUTHORITY`
- Evaluation authority: `NO_EVALUATION_AUTHORITY`
- Base: `943ab307cc5f1fded0eeb46392a71abe232523c3`

This package is limited to public-source inspection and synthetic-array checks. It neither reads nor names a runtime data root, and it contains no training or evaluation workflow.

## Proposed method contract

An externally frozen PPC decision supplies the case-level historical expert. The primary policy never performs whole-case replacement. For rim class 1 and cup class 2, deterministic 4-connected add/remove components are generated, filtered at 8 pixels, sorted by the fixed contract, overlap-resolved in that order, and capped at 12.

Accepted regions blend current and historical probabilities with lambda 0.50 or 0.75. Outside accepted regions, the current probability bytes are preserved. Lambda 1.0 belongs only to the evaluator-only C9 concept, which remains locked in this branch.

The exact 20-feature order, two closed-form float64 ridge heads, patient-grouped OOF selection, patient-level conformal residual aggregation, 200 positive Dirichlet patient bootstrap replicates, eight candidate settings, acceptance conditions, budgets, and deterministic ranking are machine-readable in `CARE_HR_V0_7_DESIGN_DRAFT.json`.

## Controls

The pure API exposes C0, C3, C4, C5, C6, C7, and C8. C9 is isolated in the evaluator helper and unconditionally calls the review authorization lock. Ground truth occurs only in target/evaluator helpers; it is absent from proposal, feature, and policy APIs.

## Deferred gates

Safety, value, relative-to-PPC-V0.6B, and stability thresholds are recorded as draft constants only. None was executed, and this document makes no scientific claim.

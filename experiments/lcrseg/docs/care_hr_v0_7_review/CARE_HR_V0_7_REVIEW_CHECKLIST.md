# CARe-HR V0.7 external review checklist

- Status: `BLOCKED_AWAITING_EXTERNAL_CODE_REVIEW`
- Draft state: `DRAFT_NOT_REGISTERED`
- Training authority: `NO_TRAINING_AUTHORITY`
- Evaluation authority: `NO_EVALUATION_AUTHORITY`

## Reviewer checks

- [ ] Confirm proposal add/remove semantics, four-connectivity, ordering, IDs, overlap priority, minimum size, and cap.
- [ ] Confirm the C8 path cannot perform whole-case historical replacement.
- [ ] Confirm unchanged probability bytes outside accepted regions.
- [ ] Confirm the 20 feature names and order exactly match the design draft.
- [ ] Confirm proposal, feature, and policy APIs contain no source identity or outcome fields.
- [ ] Confirm targets are confined to training/evaluator-only helpers.
- [ ] Confirm ridge standardization, unpenalized intercept, patient grouping, and larger-lambda tie break.
- [ ] Confirm conformal residual signs and patient-maximum aggregation.
- [ ] Confirm bootstrap positivity, patient sharing, deterministic seed, and separate validity/eligibility accounting.
- [ ] Confirm acceptance inequalities, budgets, candidate ordering, and no-candidate hard stop.
- [ ] Confirm C9 cannot execute while the review lock is active.
- [ ] Confirm all real-mode arguments stop before application file access or execution-package import.
- [ ] Confirm the static audit and predecessor immutability check.

## Observed execution boundary

- Real data reads: 0
- NAS reads: 0
- Checkpoint loads: 0
- Model forwards: 0
- Training fits on real data: 0
- Ground-truth/source-identity reads: 0
- Main merge: no

External approval must be supplied outside this branch before any broader work package is designed.

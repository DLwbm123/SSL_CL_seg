# PRES-DSR-SF V0.2.1 preregistration

Registration ID: `PRES_DSR_SF_V0_2_1_CALLGRAPH_SCOPE_RECOVERY`.

This is a new prospective validation-only protocol from `ff42db2ec2381aad176139ab788a9925eef9d147`. It is not a second V0.2 attempt and changes no descriptor, router, memory, expert, fusion, bootstrap, backend, E1-E6, threshold, or test-set rule. A separate authorization is required before any private input access or evidence generation.

## Recovery scope

The six ridge routers produce 30 lambda and 24 temperature rows, for 54 ridge-local rows. Clean M1 contributes 24 temperature rows. V0.2.1 writes separate M1 and ridge CSVs, then writes their sorted, exact-key, disjoint 78-row union. Missing, duplicate, extra, wrong-family, wrong-grid, multiple-selection, and nonfinite evidence block execution.

The frozen output counts are 78 combined CV, 915 router-score, 117 confusion, 27 cross-expert, 120 soft-fusion, 90 bootstrap, and nine memory rows, totaling 1,356. Expected keys are declared before the first real forward and are not inferred from observed rows.

## Scientific freeze

The eleven named scientific functions in `core.py` are byte/AST-bound to `09f4600348f8708ca9e865f7d5c925b6472cd013`, with combined AST digest `3acfbf968bbd52d417a13859e9de64d217de59433c962a5f4e2f78ac8d10526b`. Any change blocks as `BLOCKED_SCIENCE_SOURCE_CHANGED`.

The registered grids remain five lambdas `{1e-4,1e-3,1e-2,1e-1,1}` and four temperatures `{0.5,1,2,4}`. Descriptor dimension 102, train-only standardization, memory cap 512, five folds, closed-form ridge with unregularized bias, M1/M2 controls, probability fusion, expert mapping, five bootstraps, validation-only use, control tolerance `1e-6`, and all E1-E6 formulas and priorities remain unchanged.

## Execution and hard stop

After separate authorization, exactly one create-only V0.2.1 `formal_01` may regenerate every descriptor, memory, control, router, expert probability, candidate prediction, validation metric, and bootstrap from frozen inputs. Old V0.2 intermediate artifacts are not inputs.

Each registered phase is sealed in order. Validation segmentation GT is evaluator-only after candidate prediction seal. On any failure, partial evidence and the true durable child exit are preserved with no automatic retry.

After the validation report, execution stops for independent review. Test evaluation, a second V0.2.1 attempt, C0 regeneration, hyperparameter expansion, validation refit, fine-tuning, LoRA, adapters, other benchmarks, sweeps, main merge, and performance training remain prohibited.

# MMPR-GS v0.1 prospective feasibility registration

Registration: `MMPR_GS_V0_1_MASS_MATCHED_R3_RANKING`. This independent study does not reopen DI-DMPA. The separately published closure commit is `ac746fe8f1d335036ee35bb2f2f041aac3aae118` and the base is `ad0e1d47b76165beccd118db41242a3025044504`.

The accompanying JSON is the exact executable registration. It binds frozen inputs, equations, tie serialization, aggregation, all five gates, budget, test scope, failure precedence, archive policy and hard stop. No private checkpoint tensor or cache array, forward, gradient or new scientific result has been accessed/generated for this study before this registration.

## Frozen inputs and independent question

Gate0 B0, K2 PASS, Gate1B transport FAIL and Gate1C identity-history reliability FAIL remain unchanged. Reuse the verified 14,470-file / 17,712,127,650-byte complete v3 private bundle. Full content and manifest hashes must pass before array/tensor reads. Use the existing495 validation caches and exactly the original72 pairs, with one new draw0 integration on the original3 integration pairs. Do not regenerate B0 or validation caches.

## Ranking and exact mass

Use original unrounded R3 solely to rank EMA-active pixels within each image and each teacher-predicted class. The target count is exactly the number of original R1-valid active pixels in that stratum. Sort descending score, then ascending original full SHA256 `H([reliability-tie-v1,seed,stage_index,case_id,y,x])`, then coordinates only after hash equality. Select exactly that count, including zero-score ties if necessary. Null rows keep R1 unchanged. Every image/class must have integer mass difference0. GT and evaluator-ignore masks never enter selection. No cross-case/class allocation or threshold search. Q2 uses R2 as control; Q3 is original soft R3 descriptive only.

## Loss and diagnostic projection

Use original student/teacher probabilities and the original detached teacher draw0 target. `L_u=sum(w*sum_c((p_s-stopgrad(p_t))**2))/(sum(w)+1e-12)`. Zero mass returns graph-connected zero. Only pixel normalization is allowed.

The supervised reference is the fixed labeled-cycle CE batch. All trainable parameters enter `autograd.grad`; retain a complete active/inactive inventory with explicit zero placeholders for None, including sigma/classifier. Six existing blocks remain registered. Use same-Gaussian FP64 shadow VJP, preserve native FP32 scores and masks, and retain native-gradient comparisons.

`g_proj=g_u-min(0,g_s.T@g_u)/(g_s.T@g_s+1e-12)*g_s`. No clipping, rescaling or second correction. Report raw/projected dot, cosine, norm, norm ratio, projection flag, six blocks, class components and stage distributions. Required projected dot is at least `-1e-10`.

## Validation and decision

GT is read only by a separate evaluator after masks are sealed. Remove255 only there. Use original equal-case stratum pixel weights and report weighted selected error/precision, exact full-image and non-ignore mass, selection changes, nulls, ties, and one-pixel four-neighbor valid-GT boundaries/interior. Foreground macro is an equal average of exactly18 seed/domain/predicted-class units. Do not drop undefined units. F2 relative improvement is the relative reduction in macro error, not a mean of unit percentages.

F1: every case/class mass difference0 across495 validation cases and72 pairs, with no GT selection/reallocation. F2: macro error reduction>=0.10, at least12/18 strictly improving units, no precision drop>0.02. F3: raw negative cosines<=43/72, median cosine improvement>=0.05, no stage median worsening>0.05, no required undefined comparisons. F4:72 projected dots>=-1e-10, no zero projected gradient, norm-ratio median>=0.50, linear p10>=0.20, each stage median>=0.40, all values finite. F5: no teacher/bank gradient, models/checkpoints unchanged, labeled/validation GT role isolation, no hidden/test GT and zero updates.

Only all five passing yields `PASS_MMPR_GS_FEASIBILITY`. Engineering failures take priority. Otherwise failure precedence is F2 ranking, F3 raw compatibility, F4 retention; retain every failed gate. Q2/Q3 cannot rescue Q1. Scientific failures do not truncate the registered diagnostics. Engineering failure stops execution with complete preserved evidence.

## Exact execution and tests

The synthetic compiler must exercise the same pair kernel, freeze source hashes and match3 native+2 FP64 forwards,3 native+6 FP64 autograd calls per pair. Three integration pairs cost15 real forwards;72 formal pairs cost360; total375. Per pair:6 case/class mass rows,2 global alignment rows,12 block alignment rows,21 class-component rows,7 retention rows,21 native/FP64 comparison rows and1 model guard. The JSON fixes the full call graph. Mismatch is `BLOCKED_CALL_GRAPH_MISMATCH`, never a silent budget increase. No teacher-noise, posterior or PoE phase.

Run all unchanged reusable Gate1C diagnostic regression modules and new tests covering the requested46 obligations. Explicit exclusions before collection: old `test_real.py` would consume forbidden old private inputs/extra forwards; `test_baseline.py` is B0 training/overfit and invokes optimizers/backward/GAS updates prohibited this round. The old Gate0 training suite and its historical skip are not rerun. No skip is allowed in the new actually collected suite. New3-pair integration is the only real integration. No tests or historical files are changed to evade these boundaries.

## Storage, publication and stop

Use the existing zmic44 Python environment and physical GPU4/5/6/7 with existing jobs left alone. All new large outputs, scratch, logs and archives stay under the new NAS root in JSON, through the NAS wrapper with a real write probe. Each phase is create-only, no retry/resume after real execution. Durable parent receipts must contain actual process exits. Seal and fully SHA-verify a separate content-addressed NAS integrity copy; it is not independent-device backup. Keep raw arrays and checkpoints out of Git.

Publish separate execution authorization next; only after both remote verifications may private inputs or computations begin. Publish the compiled exact source before real forwards. Finally publish reports, verify the branch SHA and stop for independent review. No optimizer/backward/parameter.grad write, teacher/GAS/prototype update, method registration/training, transport, class-balanced loss, Gate2, new dataset, full sweep or main merge is authorized.

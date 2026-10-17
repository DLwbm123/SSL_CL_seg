# Final bounded source control: CONSERVATIVE_U_V1

Frozen before launch, after HARD_U_V1 ended at train Dice 33.6671% and validation
14.2786%. Existing SSL is 65.8176% validation; L-only is 72.2917%. No candidate is
discarded. The hard class-balanced objective damaged labeled fitting as well as
validation. A focused deployment audit found matching final/sealed native tensors,
an identity transform, and no running normalization buffers.

Hypothesis: uniformly weighting classes and adding foreground Dice can amplify a
small number of incorrect pseudo-labels. Test one conservative U objective:
detached EMA argmax cross-entropy, restricted to the original admission mask AND
confidence >0.95, averaged over the entire image grid. No class reweighting, Dice,
confidence taper, tuning sweep, pretrained weights or new input transformations.
This is one bundled objective comparison; it cannot isolate its individual terms.

Keep lambda_U=0.25 and the existing ramp, labeled loss, initialization, LCTX,
hierarchy, source data/order/seed163 and Adam settings. Fresh initialization, no
checkpoint selection. The sole schedule adjustment is 43,599 rather than 43,600
steps to fit the remaining physical budget; polynomial decay uses this endpoint.
Warmup and ramp still round up to 8,720 steps. Hardware is the same as HARD_U_V1.

Actual calls already spent: 43,601 + 43,600 = 87,201. This final control has at most
43,599 new calls; the study total cannot exceed 130,800. Any failed/replayed calls
consume this allocation; no automatic budget extension. CPU tests and the native
initial-loss/backward check use zero optimizer calls.

Evaluate fixed-final train-L and validation once. Reuse prior aggregate and patient
rows for comparisons; no independent test or U ground truth. Compare against both
72.2917% L-only and 65.8176% existing SSL, report classes, ED/ES and training gap.
Improvement over the old SSL alone does not establish a positive U contribution.
Publish the complete negative or positive result. No additional training after the
budget is spent without new user authorization.

Startup verified: nine CPU checks passed, including unchanged labeled gradients,
confidence masking, sparse full-grid normalization and zero empty-mask gradients.
The native initial loss matched 2.063466310157642, and backward/initial-checkpoint
restore used zero optimizer calls. The process passed 100 training calls without
failure; its full interpreter/entry command and GPU process display are neutral.
GPU memory and data-volume mount/space/write-read checks passed before launch.
This confirms healthy startup, not effectiveness.

# V8.1 independent paired-noise diagnostic

Authority: user requested continued analysis and improvement after weak experiments, and explicitly authorized image-level exploration. This is a new round; frozen V8 remains stopped with its negative result.

## Evidence and one change

V8 skipped actor updates in 46/64 and 48/64 groups, although the frozen memory gate averaged 0.9914 and policy distributions changed. The absolute reward noise threshold was 0.00202752. Common-random-number candidate branches are compared within groups; absolute reward shifts between streams include baseline drift that cancels from action contrasts.

Use the median absolute cross-stream difference of each action-minus-native reward contrast, divided by sqrt(2), with minimum 1e-4. Original A has exactly 8 contexts × 8 non-native contrasts; the resulting frozen floor is 0.0007379373167760999. No Q_dev value enters this formula. Aggregation remains global to isolate one change; heterogeneous context noise is a known limitation and may still make the estimate inadequate.

## Fixed experiment

Restart both 601/602 actors from zero logits, and all eight native student context states from the immutable original A entries. Reuse the same 64 groups per actor, four paired branches, 100 steps, fixed old-reference reward, branch-zero retention, full-group GRPO and all native supervision/EMA/optimizer settings. Preserve both final actors. Qualified snapshots, data roles and action semantics remain unchanged.

Repeat the registered 20 development endpoints with NATIVE, FIXED_BEST, UNIFORM_ACTION and both final priors. All training finishes and is sealed before development readout. The development set has already been observed; this is a mechanistic diagnostic and cannot establish independent generalization. Do not change conditions, examples, metrics or stop rules after seeing outcomes.

## Success and stop

At least one fixed prior must exceed both uniform and its matching original prior in mean utility by 0.0005, while new Dice is no worse than uniform by 0.0025 and old Dice no worse by 0.005. Fewer than 50% skipped groups is a secondary mechanism check, never sufficient for success. Otherwise seal STOP_PAIRED_NOISE_NO_PRACTICAL_GAIN and retire this floor-only intervention. C/D do not run in this round; a successful diagnostic still needs separately frozen target implementation and qualification.

## Budget and storage

New caps: 8 student and 8 actor qualification invocations; 51,200 student and 512 actor prior invocations; 400 entry and 4,000 development invocations. Total new student cap 55,608, actor cap 520. No unused quota transfer, retries or seed expansion. Carry all 78,829 V8 student and 144 actor invocations into the cumulative ledger; shared historical source adds 8,000 separately. Models and roles stay on NAS, each round is create-only, GPU 5/6/7 only. Publish anonymous full outcomes and negative results through the required proxy.

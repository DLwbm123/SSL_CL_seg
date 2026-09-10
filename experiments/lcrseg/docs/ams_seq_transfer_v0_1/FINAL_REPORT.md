# AMS Sequential Transfer V0.1

Completed36 tasks /95400 formal optimizer updates. Engineering:ENGINEERING_COMPLETE. Scientific states:SEQ_TRANSFER_VALUE_NOT_ESTABLISHED, CONTEXT_VALUE_WITH_PSEUDO_UNCERTAINTY.

Full-parameter sequential control; not KI, parameter isolation, independent-patient confirmation or clinical safety. Fixed epoch100 students; no source-score selection.

| Comparison | Final | Incoming | Old | Forget |
|---|---:|---:|---:|---:|
|T_AMS - T_CE|+0.039855|+0.021767|+0.057942|-0.057942|
|T_AMS - T_CED|+0.044313|+0.032757|+0.055870|-0.055870|
|T_LCTX - T_CED|+0.055500|+0.031413|+0.079587|-0.079587|
|T_UCTX - T_LCTX|-0.012489|+0.007380|-0.032358|+0.032358|
|T_AMS - T_UCTX|+0.001302|-0.006036|+0.008641|-0.008641|
|T_LCTX - T_CE|+0.051042|+0.020424|+0.081660|-0.081660|
|T_UCTX - T_CE|+0.038552|+0.027803|+0.049301|-0.049301|
|T_UCTX - T_CED|+0.043011|+0.038793|+0.047229|-0.047229|

Pseudo Final increment:+0.001302; paired-patient95% interval [-0.003758,+0.005761]. PRACTICAL_INCREMENT_NOT_MET.
Drishti cup amber groups:3; all retained in DECISIONS.json.

PAIRED_EFFECTS.csv contains every seed/order and order-averaged seed SD. Shared physical-domain patient weights are used for all2000 bootstrap draws; two orders do not create six independent seeds. DeltaForget=-DeltaOld and DeltaFinal=(DeltaIncoming+DeltaOld)/2 verified.

DOMAIN_CLASS_COSTS.csv and DECISIONS.json retain all class costs and each frozen guard. Source-only source score does not filter target work. LCTX donor rows are reuse of current L images, not extra U reads or extra donor GT supervision. Source image/GT access is absent from target training; current EMA and Adam reset.

Operation attempts and successes, synthetic/smoke costs, peak allocation/reservation and single-student deployment are in RESOURCE_ACCOUNTING.json and MEMORY_AND_DEPLOYMENT.json. Patient rows, weights and raw runtime paths stay on NAS. No additional experiment starts.

## Completion and interpretation

The matrix finished at 2026-09-09T23:03:55.328079+08:00 (parent wall time 80.7 minutes). Closeout verified 36 training receipts, 36 evaluation receipts and 72 successful process exits, all on execution source `f174cf2a68212cf04ad19b18d8203a6455c94aee`. All 30 target initializations and six shared warmup groups passed the recorded boundary checks. The detached parent has stopped.

AMS improves average Final, Incoming and Old relative to both supervised baselines, but fails the pre-registered single-seed/order/domain/class guard. Thus VALUE_NOT_ESTABLISHED means the complete acceptance rule was not met; it does not mean the mean effect is zero or negative. The failed class items are listed below. Mean Old improved, so the mean-old-guard plasticity/retention terminal did not trigger.

| Arm | Final | Incoming | Old | Forget |
|---|---:|---:|---:|---:|
|T_CE|0.549103|0.697653|0.400553|0.284266|
|T_CED|0.544644|0.686664|0.402625|0.282194|
|T_LCTX|0.600145|0.718077|0.482212|0.202606|
|T_UCTX|0.587655|0.725457|0.449854|0.234965|
|T_AMS|0.588958|0.719420|0.458495|0.226324|

The table equally averages the two fixed orders and three optimization seeds; scores use the patient-mean rim/cup macro Dice. Final = (Incoming + Old)/2. Forget is source-stage score minus final Old score and is not clipped.

| Arm / baseline | All guards | Failed checks |
|---|---|---|
|T_AMS / T_CE|FAIL|class_single|
|T_AMS / T_CED|FAIL|class_single|
|T_UCTX / T_CE|FAIL|class_single|
|T_UCTX / T_CED|FAIL|class_single|
|T_LCTX / T_CE|PASS|none|
|T_LCTX / T_CED|PASS|none|

| Arm / baseline | Order | Seed | Domain / role | Class | Delta |
|---|---|---:|---|---|---:|
|T_AMS / T_CE|O1|61|Drishti_GS / current|cup_dice|-0.062307|
|T_AMS / T_CE|O2|63|Drishti_GS / old|cup_dice|-0.051805|
|T_AMS / T_CED|O1|61|Drishti_GS / current|cup_dice|-0.053250|
|T_UCTX / T_CE|O2|63|Drishti_GS / old|cup_dice|-0.068132|
|T_UCTX / T_CED|O2|61|Drishti_GS / old|rim_dice|-0.067100|

All these items cross the fixed -0.050 guard; none were removed or used to change the recipe. They are domain/class averages within a single optimization seed, not individual-patient effects. All class costs, including non-failing ones, remain in DOMAIN_CLASS_COSTS.csv.

LCTX is the only mixed control passing all the same guards against both CE and CE+Dice. It is an L/L anchor-supervised mixing control, not ordinary fully supervised CutMix. Adding the U context pool (UCTX minus LCTX) improves Incoming but reduces Old and Final on average. Adding pseudo targets (AMS minus UCTX) slightly improves Final and Old while reducing Incoming; its Final increment is below 0.003 and the conditional patient interval includes zero. The strongest supported interpretation is value in labeled-image context mixing, with no established practical pseudo-target increment. LCTX remains a control and does not replace the pre-specified AMS main candidate.

### Drishti cup amber flags

| Arm / baseline | Order | Role | Mean delta |
|---|---|---|---:|
|T_AMS / T_CE|O1|current|-0.010066|
|T_UCTX / T_LCTX|O2|old|-0.027552|
|T_AMS / T_UCTX|O1|current|-0.010817|

### Resource and evidence scope

Training retained two full models (student and current EMA), and evaluation/deployment used one student. Maximum recorded CUDA allocated/reserved memory across formal tasks: 891.1/1020.0 MiB. Formal telemetry records 95,400 optimizer steps/backwards/EMA updates, 148,425 model forwards, 190,800 labeled sample accesses, 49,920 U-image accesses and 2,145 validation sample accesses. Qualification used 931 synthetic and eight discarded real-smoke updates, excluded from formal totals.

This is a repeated-development-cohort, two-domain full-parameter sequential control. It does not establish key isolation, independent-patient generalization, zero forgetting or a recovered KI implementation. The next separately authorized question should retain LCTX as a strong supervised augmentation control and examine class-specific retention before claiming additional SSL value; this run starts no such experiment.

Delivery mapping: SOURCE_AND_INPUT_LINEAGE and qualification evidence are retained inside QUALIFICATION_RECEIPTS.json; domain-boundary qualification is also covered there and actual-task verification appears in CLOSEOUT_VERIFICATION.json plus SOURCE_INITIALIZATION_LEDGER.csv. Source/configuration freeze files remain unchanged. Private patient identifiers, image/GT arrays, weights, raw paths and logs stay on NAS.

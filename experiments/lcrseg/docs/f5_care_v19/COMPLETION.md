# V19 CaRE reference: completed, both original gates fail

All four trajectories completed with8 sealed stages,4 job EXIT0, controller/process EXIT0 and completion audit PASS. All21200 formal updates and20 planned domain evaluations were executed. Frozen source445af47e95019c56882954445a93810d0f2d69e0. No formal retry or extra patient probe occurred.

## Aggregate results

| Comparison | Final delta (pp) | Old delta (pp) | Incoming delta (pp) | Practical gate |
|---|---:|---:|---:|---|
|F5|+2.4181|+2.9628|+1.3289|False|
|B0 (historical)|+3.0250|+3.9465|+1.1821|True|
|B2 (historical)|-0.0720|-0.1094|+0.0027|False|
|V13|+0.0033|-0.1956|+0.4011|False|
|V15|-0.4394|-0.6391|-0.0401|False|
|V17|-0.6822|-1.2089|+0.3712|False|
|V10|+0.8738|+1.1399|+0.3416|False|

Absolute Final/Old/Incoming/Forget (%): Final=67.8291, Old=66.1316, Incoming=71.2239, Forget=10.7210.

The original matched-F5 gate fails on163/O1 RIM cup (−5.1304pp). The historical-B2 gate fails on mean Final (−0.0720pp versus the required+0.5pp), seed164 order-mean Final (−1.6609pp; seed163+1.5168pp), order2 Old (−0.5845pp), and the following class cells. No threshold was relaxed.

| Reference | Seed/order | Domain | Class | Delta (pp) |
|---|---|---|---|---:|
|F5|163/1|RIM_ONE_r3|cup|-5.1304|
|B2|164/2|REFUGE|rim|-5.6721|
|B2|164/2|Drishti_GS|rim|-7.1956|
|B2|164/2|Drishti_GS|cup|-8.1211|

All positive and negative pair/seed/order/domain/class deltas, including smaller negative cells, are retained in [RESULTS](delivery/RESULTS.json). Per-stage scores are final unified readouts after all seals; no partial performance was used.

## Intended improvement and collateral damage

164/O1 RIM rim reaches59.3553%, versus V13+0.9740pp, SARD+1.3533pp, V15−1.3886pp and B2−4.5911pp. It falls from71.9804% at stage1 to59.3553% at stage2 (−12.6251pp). The targeted damage remains despite a small improvement over V13.

Order2 Incoming improves by1.2647pp over V13 and1.5374pp over SARD; versusB2 its order mean is−0.0906pp, within the original order threshold. This gain accompanies seed164/O2 old-domain damage: Drishti cup52.9058%, versus V13−7.8278pp and SARD−10.3662pp, and a74.2346%→52.9058% stage decline (−21.3288pp). Seed163/O2 Drishti cup55.4660% remains close to V13 (−0.2068pp), above V15+9.6255pp but below SARD−2.3654pp; stage decline−17.1307pp.

V19 mean Final differs from V13 by only+0.0033pp. A near-zero aggregate does not mean equivalent cell behavior: seed163 improves and seed164 regresses, while Incoming gains trade against Old. These observations do not establish a causal confidence mechanism.

## Activation, engineering and costs

All16960 active reference calls contained a reverse-branch pixel; total forward/reverse supported pixels=3379094214/1373778918 (reverse share28.9042%). Teacher log-floor values=0. Raw reference projection occurred8866/16960 times. All50880 scheduled present-class reductions and16960 reference VJPs are recorded. Every active main-U call had nonzero A/B/R routes. Activation and raw directional feasibility do not certify teacher correctness, actual Adam feasibility or old-domain retention.

Actual optimizer costs:21200 formal/source0; qualification32CPU/10generatedCUDA/2currentL, including deliberate failure. Actual model costs:89040 student and50888 teacher full forwards, including16960 entry-anchor forwards;16960 reference VJPs and64 R-diagnostic VJPs. Training autograd_grad calls55184 and readout native forwards1190 are independently retained by operation receipts. No extra CaRE full forward or VJP relative to V13. Controller elapsed5328.8126seconds; trajectory runtimes1722.9449/1884.0830/3502.2647/1877.0419seconds in163O1/163O2/164O1/164O2 order. Shared-GPU runtime varies; no causal speed claim. Peak allocated2210312704bytes; peak reserved at most2311061504bytes.

Campaign formal actual424673; cumulative qualification992CPU/247generatedCUDA/50currentL. All historical failures and V5 rollback duplicates remain charged. GUA and REDistill formal0; unused allowances are not transferred to retries. [CAMPAIGN_COSTS](delivery/CAMPAIGN_COSTS.json), [SEALED_DIAGNOSTICS](delivery/SEALED_DIAGNOSTICS.json), [JOB_EXITS](delivery/JOB_EXITS.json), [COMPLETION_AUDIT](delivery/COMPLETION_AUDIT.json) retain detailed evidence.

The four authorized literature directions are now closed for these fixed adaptations: GUA did not meet native advancement, SARD completed and failed both gates, REDistill did not meet mechanism advancement, CaRE completed and failed both gates. No original-candidate retry is authorized by this closeout. A subsequent independently justified module requires its own frozen finite protocol under the existing autonomous follow-up authority.

This is a completed development experiment and a negative practical result, not independent patient confirmation, SOTA, clinical benefit or proof against the complete original CaRE-KD method. Private weights, patient rows, data, raw logs and CONFIG remain on NAS.

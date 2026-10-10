# V6 completed: increasing R U strength failed the practical gates

| Method | Final % | Old % | Incoming % | Forget pp |
|---|---:|---:|---:|---:|
|U_STRENGTH|64.2484|61.3631|70.0191|15.5195|
|F5|65.4109|63.1689|69.8950|13.6757|
|B2_PARENT_PAS_KL|67.9011|66.2410|71.2212|10.7007|
|B0_PARENT_LCTX|64.8041|62.1852|70.0418|14.3667|

Final=(2Old+Incoming)/3, Forget lower-is-better. F5 matched archived control, B0/B2 historical aggregates, not concurrent retraining. Both fixed gates fail; all seed/order/domain/rim/cup negative cells and B0 are retained in RESULTS.json.

| Seed | Order | Final % | Delta F5 pp | Delta historical B2 pp |
|---|---:|---:|---:|---:|
|163|1|66.2890|-4.2657|-0.5483|
|163|2|62.1029|-0.1841|-2.5530|
|164|1|65.0067|-0.3863|-5.5564|
|164|2|63.5951|+0.1862|-5.9529|

Vs F5 meanFinal−1.1625pp/Old−1.8058pp/Incoming+0.1241pp; both seed-averaged Final changes negative (163−2.2249pp,164−0.1000pp), order1 Old−3.7538pp and multiple class losses below−5pp. Worst F5 class163/O1/REFUGE rim−7.1352pp. Historical B2 meanFinal−3.6527pp/Old−4.8779pp/Incoming−1.2021pp; all four Final pairs negative, both order Old gates and order2 Incoming fail; worst class164/O1/RIM rim−15.4575pp. Historical B0 meanFinal−0.5556pp and all four Final pairs negative. No thresholds/checkpoints were selected after readout.

All16960 active U calls have nonzero R gradients; A/B U prohibition retained. 14/32 predefined R gradient points conflict, 9/32 have U norm above L. These sparse points are not full-step conflict frequencies; no B direction evidence or proof of causal forgetting follows. 64 formal diagnostic VJPs, no extra patient probes. SEALED.evaluation=NOT_READ_YET records sealing-time readout isolation, while RESULTS and completion audit document subsequent full evaluation.

Contextual V6−V5 and V6−V2 means: {"V5": {"Final": -0.1366966584355589, "Old": 0.4678407749616892, "Incoming": -1.345771525230055, "Forget": -0.3599990955406973}, "V2": {"Final": -1.2641915322175235, "Old": -1.9804834942470677, "Incoming": 0.16839239184156252, "Forget": 2.0185727265915867}} pp. These change more than one configuration dimension and cannot isolate strength; only the F5 comparison does.

All four exits0, eight targets sealed before20 seen-domain evaluations, controller0 and existing completion auditPASS, ledger21200 verified once. Source0, campaign191473 physical calls, qualification576CPU/121generatedCUDA/26real-L. Original V5 interruption/replay673 and all earlier scientific negatives/qualification failures retained. No formal retries, new seeds, partial readout or test access. Execution ac02d053a07a75b9e3af1f33b92d12f10f781bd5.

Higher U on original F5 worsened average retention and does not close the B2 gap. V2 one-sided R projection was active8183/16960 calls and yielded only+0.1017pp Final over F5, so conflict geometry is observed but not established as the main gap cause. After verified public delivery, next single candidate is conflict rejection at original F5 lambda_U0.25: retain U when gL_R dot gU_R>=0, reject entire R U when negative. Unlike V2 it discards the orthogonal conflicting component too. Test this hypothesis with disabled-gate F5 parity and a separate finite preregistration; no guarantee of Adam alignment or old-domain retention. This report itself does not claim next-round startup.

These repeatedly reused development results are not independent-patient confirmation, SOTA or clinical benefit. Anonymous aggregates/diagnostics/source/protocol/costs are public; patient-level rows, data, weights, private config and raw logs remain NAS. Hourly monitor stays active and RL paused.

# V25 dense entry feature orientation: completed, B2 gate failed

Execution `7ef2e37ae513251b571f49c24ee47cb22512c973`. All 4 job exits, controller and process exits are zero; 8/8 stages sealed; 21,200 formal updates, zero source updates; 20 uniform seen-domain evaluations after all seals. Existing completion audit PASS, version/plan/config/qualification bindings and physical count checked once. Controller ended. No new model, optimizer, qualification or patient evaluation during delivery.

F5 practical gate PASS; historical B2 gate FAIL. Final/Old/Incoming/Forget means are 67.8328/66.0331/71.4322/10.8879%. B2 remains a historical aggregate comparison; repeated development patients are not independent confirmation.

| Reference | ΔFinal pp | ΔOld pp | ΔIncoming pp | ΔForget pp |
|---|---:|---:|---:|---:|
|F5|+2.4219|+2.8642|+1.5372|-2.7878|
|B0|+3.0287|+3.8479|+1.3904|-3.4789|
|B2|-0.0683|-0.2079|+0.2110|+0.1872|
|V24|+0.4155|-0.0335|+1.3135|+0.0034|
|V23|-0.7786|-1.4979|+0.6600|+1.5973|
|V21|-0.4467|-0.8812|+0.4222|+0.8782|
|V22|+1.5353|+2.0700|+0.4659|-1.5109|
|V13|+0.0070|-0.2941|+0.6093|+0.4647|
|V15|-0.4357|-0.7377|+0.1682|+0.8975|
|V17|-0.6785|-1.3074|+0.5795|+1.4804|
|V19|+0.0037|-0.0985|+0.2083|+0.1668|
|V20|-0.6837|-0.7948|-0.4616|+0.6667|
|V10|+0.8776|+1.0414|+0.5499|-1.0049|

B2 failures: mean Final −0.0683 pp (below required +0.5 pp); seed 163 Final +1.6592 pp but seed 164 −1.7958 pp; order 2 Old −0.8189 pp and Incoming −1.5187 pp. Class failures: seed164/order1 RIM rim −5.9890 pp, seed164/order2 Drishti cup −6.4982 pp. Full positive and negative seed/order/domain/rim/cup matrices for every reference remain in RESULTS.json; no cell subset replaces the frozen gate.

| Seed/order | Final % | Old % | Incoming % | Forget % |
|---|---:|---:|---:|---:|
|163/1|69.6785|67.4549|74.1256|8.1552|
|163/2|65.1331|62.9842|69.4308|13.7808|
|164/1|69.5532|68.3442|71.9712|8.1241|
|164/2|66.9665|65.3491|70.2012|13.4913|

## Stage changes and evidence boundary

Seed163/order1 old RIM_ONE_r3/rim: 72.7117 → 55.8698% (-16.8418 pp).
Seed163/order2 old Drishti_GS/cup: 71.4885 → 50.5329% (-20.9555 pp).
Seed164/order1 old RIM_ONE_r3/rim: 72.1053 → 57.9573% (-14.1479 pp).
Seed164/order2 old Drishti_GS/cup: 75.7898 → 54.5286% (-21.2612 pp).

Dense beats pooled V24 mean Final +0.4155 pp through Incoming +1.3135 pp, while Old −0.0335 pp. Against relation V23: Final −0.7786 pp, Old −1.4979 pp, Incoming +0.6600 pp. Against hierarchy V21: Final −0.4467 pp, Old −0.8812 pp, Incoming +0.4222 pp. Against V13: Final +0.0070 pp, Old −0.2941 pp, Incoming +0.6093 pp. Thus removing the pooling blind spot passed mechanism checks but did not establish retention. Dice does not identify a boundary, topology, feature rotation or causal failure mechanism.

## Activation and actual costs

Dense calls 21,200, supported leaf regions 63,600, eligible parent regions 21,200, valid pixels 6,118,393,141, zero-feature pixels 0, positive calls 21,200. Hierarchy coarse/parts each 21,200 positive; alignment 19,968 positive; 42,400 aligned edges, 38,160 L feature views; 68,784,459 seam pixels excluded. These counters rule out a globally inactive dense objective; they do not show teacher correctness or old-domain coverage.

Original U A/B/R nonzero calls each16,960; reference VJP16,960, original anchor full-forward16,960, present-class reductions50,880, aggregate projections9,294, fixed R diagnostic VJP64. Student full-forwards89,040; teacher full-forwards89,048 including38,160 extra current-L entry forwards. Training autograd calls55,184; readout full-forwards1,190. Exact totals and all 8 sealed module/gradient diagnostics are in ACTIVATION_COSTS and SEALED_DIAGNOSTICS.

Controller4,122.9019 seconds; jobs163O1/163O2/164O1/164O2:1,907.6982/1,923.4069/2,108.8452/1,924.5626 seconds; maximum allocated2,268,100,608 and reserved2,361,393,152 bytes. Shared GPU runtimes do not establish causal speed gains.

Cumulative method formal551,873 + single-setting B1 5,300 =557,173. Qualifications method1,184 CPU/307 generated CUDA/62 real L; includingB1 1,216/317/64. Deliberate qualification failure, historical failures,673 rollback updates and5 unfilled NA remain charged. No released historical budget reused. Transfer brace expansion initially rejected by SFTP with no file transferred; explicit allowlisted paths then succeeded. No experiment retry.

## Interpretation and next review

V25 retains V21/V13 and replaces pooled orientation with dense current-L feature direction. All existing teacher/data/deployment permissions, class-balanced reference, raw aggregate guard, original SWD R-only route and coefficients remain frozen. Dense and prototype methods constrain a current-L proxy; neither delivers evidence that the frozen entry representation is correct for these GT regions or covers previous domains. Review the existing current-U reference route and generic ancestry probability decomposition before choosing a next module. No new run, coefficient search, additional patient diagnosis or training is registered by this report.

Source/protocol/qualification/startup already public. This completion publishes only anonymous aggregate results, full positive/negative comparisons, sealed numeric diagnostics, counters, audit, exits and report. Patient rows/images/labels/weights/private configuration/raw logs remain on NAS.

# V20 complete: selected163/O1 gain; original B2 gate fails

Frozen training source d5f1d01a02e3f33e476b92fc259823050f35549a. Four trajectories/eight stages sealed, four job EXIT0, controller/process EXIT0 and completion audit PASS.21200formal/source0;20planned domain evaluations after all seals. No retry, partial readout or selected checkpoint.

| Metric | V20 % | vs matched F5 pp | vs historical B2 pp | vs V13 pp | vs V17 pp |
|---|---:|---:|---:|---:|---:|
|Final|68.5165|+3.1056|+0.6155|+0.6908|+0.0053|
|Old|66.8279|+3.6590|+0.5869|+0.5007|-0.5126|
|Incoming|71.8939|+1.9988|+0.6726|+1.0710|+1.0411|
|Forget|10.2211|-3.4545|-0.4795|-0.2020|+0.8136|

F5 practical gate passes; B2 gate fails. B2 mean Final+0.6155pp clears the0.5pp mean threshold, but seed163+2.7340pp versus seed164−1.5031pp is inconsistent. Order2 Old−1.1616pp breaches−0.5pp. Seed164/O2 Drishti rim−5.4213pp and cup−15.8466pp breach−5pp. Full positive and negative cells are preserved in RESULTS.json.

User-requested same-setting comparison:163/O1 V20 Final70.3506/Old69.9515/Incoming71.1487/Forget6.1051%; matching B2 Final66.8373/Old64.0572/Incoming72.3975/Forget12.3198. Final+3.5133pp, Old+5.8943pp, Incoming−1.2488pp, Forget−6.2147pp. This selected-setting gain does not replace the full gate.

Seed164/O2 Drishti cup drops75.6128%→45.1802% from stage1 to stage2 (−30.4326pp), versus V13−15.5533pp and B2−15.8466pp at final. Seed164/O1 RIM rim reaches60.1063%, but remaining old-domain damage must not be hidden by the mean.

## Mechanism and actual costs

All16960active calls used the block guard;13161had a violating block under the same-state old-V13 aggregate-projection counterfactual. A/B/R projection calls9352/9102/11209. These are V20-state counterfactuals, not archived V13 training measurements. All16960mainU calls had nonzero A/B/R. Class support50880reductions, entry references16960VJPs, anchors16960fullforwards, fixedR diagnostic64VJPs.89040student/50888teacherfullforwards. Activation verifies execution, not correct references or preservation under actual Adam/effective BA/finite steps.

Controller elapsed4274.7023seconds in shared-GPU conditions; trajectory and peak-memory receipts are in JOB_EXITS.json (maximum reserved2311061504bytes). Qualification32CPU/10generatedCUDA/2currentL, including deliberate failure, was reused without repetition. Method formal actual445873; adding supplementary B1 yields451173. Cumulative qualification1056/267/54. Earlier673rollback calls and5NA rows remain preserved.

The block correction has not solved the seed164/order2 retention problem. No causal attribution to an individual block is established by these comparisons. A subsequent justified method requires its own separate finite registration; no retry or released baseline budget is implied.

Source, frozen protocol, complete anonymous results/sealed diagnostics/operations/costs/exits and this report are public. Data, weights, patient rows, private config and raw logs stay NAS. Reused development patients are not independent confirmation.

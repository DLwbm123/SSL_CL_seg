# F5 V4: completed negative routing ablation

Four trajectories exited0; eight targets sealed before20 evaluations; auditPASS. Formal21200, source0. No retry, tuning or favorable checkpoint selection. Execution5944d1c178bbd3cf37d2ed01a675e4e4c32431da.

| Method | Final Dice % | Old % | Incoming % | Forget pp |
|---|---:|---:|---:|---:|
|F5|65.4109|63.1689|69.8950|13.6757|
|KL_OUTPUT_U|64.5470|61.0829|71.4753|15.6896|
|B0_PARENT_LCTX|64.8041|62.1852|70.0418|14.3667|
|B2_PARENT_PAS_KL|67.9011|66.2410|71.2212|10.7007|

F5 is a matched prior control; B0/B2 are historical aggregates, reused after qualification parity, not concurrent retraining. Two seeds×two orders; Forget lower-is-better.

| Seed | Order | Final % | Delta F5 pp | Delta B2 pp | Delta V3 pp |
|---|---:|---:|---:|---:|---:|
|163|1|63.4631|-7.0916|-3.3742|+0.1809|
|163|2|63.8594|+1.5724|-0.7964|+0.1341|
|164|1|66.8300|+1.4370|-3.7331|-0.4179|
|164|2|64.0356|+0.6267|-5.5125|-0.0732|

## Unchanged gates fail

F5 mean Final−0.8639pp, Old−2.0860pp, Incoming+1.5803pp, Forget+2.0140pp. Seed163 Final−2.7596pp and seed164+1.0318pp. Order1 Old−5.4140pp violates tolerance; five individual domain/class costs are below−5pp. Seed163/O1 Final−7.0916pp and Old−10.8739pp reproduce the severe retention failure; worst F5 class cost RIM rim−15.0337pp.

Historical B2 mean Final−3.3541pp, Old−5.1581pp, Incoming+0.2541pp, Forget+4.9890pp. Both seed Final means negative, both order Old means violate tolerance, order2 Incoming violates tolerance; seven class deltas below−5pp, worst164/O2 Drishti rim−15.5689pp. B0 gate also fails: meanFinal−0.2570pp, seed163 negative, order1 Old below tolerance and four class costs below−5pp. RESULTS.json includes every positive and negative pair/class gate.

## V4 versus V3 and diagnosis

Restricting SWD to R while retaining KL on B/R changes mean Final−0.0440pp, Old−0.1629pp, Incoming+0.1938pp, Forget+0.1728pp relative to V3. Both seed163 pairs improve slightly, both seed164 Final pairs decline. The severe seed163/O1 Old failure remains. This does not support treating B exposure to SWD as the primary explanation of V3 failure in these configurations; it also does not prove KL alone is causal. V4_VS_V3.json retains all reused pair/class comparisons.

B had nonzero KL U gradients in all16960 active-U updates. SWD B exclusion and nonzero SWD R gradients were qualified; A U forbidden was checked every step.32 predefined module/gradient points and64 formal diagnostic VJPs are retained. No extra formal queries were added. The existing architecture is active and deploys successfully, but widening B permissions without a magnitude guard has not passed retention gates.

Next falsifiable intervention: retain V4 permissions and bound the group norm of B U gradients by the same-step B L gradient norm. Use a fixed norm ratio, not a selected coefficient; leave A and R gradients unchanged. This tests a supervised-anchor bound on U magnitude, not a guarantee of Adam step alignment or old-domain preservation. A new independently frozen finite round and related gradient/restore qualification are required before native execution.

## Costs and boundary

V4 qualification32 CPU,18 generated CUDA including one deliberate failed-after-optimizer call,2 discarded real-L calls. Cumulative completed formal148400, source0; cumulative qualification480 CPU,80 generated CUDA,22 real-L. Controller3141.50sec; prior failures retained, no cap reset.

All anonymous results/diagnostics, source, protocol, costs and negative findings are public. Private data, patient rows, weights, raw logs and runtime config remainNAS. This is repeated development evidence, not independent patient generalization, significance, SOTA or clinical validation. Hourly autonomous follow-up remains active because success gates failed.

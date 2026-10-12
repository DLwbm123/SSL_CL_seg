# V15 completed: mean improvement, original dual gate failed

All four trajectories ended with EXIT0; eight target students were sealed before the 20 scheduled domain evaluations. Completion audit PASS; controller and process EXIT0. Execution `65ecc10f566fe5fe8150236e08ea1dd80b7d2cf6`. No training retry or partial readout was used.

| Metric | V15 (%) | V15 minus V13 (pp) | minus matched F5 (pp) | minus historical B2 (pp) |
|---|---:|---:|---:|---:|
| Final | 68.2685 | +0.4427 | +2.8576 | +0.3674 |
| Old | 66.7707 | +0.4435 | +3.6019 | +0.5297 |
| Incoming | 71.2640 | +0.4412 | +1.3690 | +0.0428 |

B0 mean Final improvement is +3.4645 pp. Complete seed/order/domain/class comparisons, including every negative cell and V12/V10 background comparison, are retained in delivery/RESULTS.json. Historical B2 is not a newly matched rerun.

## The requested tradeoff

Seed164/order1 RIM rim improves from V13 58.3813% to 60.7438% (+2.3626 pp); it remains 3.2025 pp below B2. Its stage1-to-stage2 change is 71.5868% to 60.7438% (-10.8430 pp), versus V13 -13.4981 pp. Seed164 Incoming averaged over orders improves by +1.1580 pp versus V13 and +1.7777 pp versus B2; seed164 Old improves +0.9703 pp versus V13 but remains -0.8536 pp versus B2.

The seed163 cup benefit is not preserved: order2 Drishti cup is 45.8405%, versus V13 55.6728% (-9.8322 pp), B2 52.2294% (-6.3889 pp), and V12 42.4088% (+3.4317 pp). Its stage1-to-stage2 drop is -24.5638 pp. Thus the better total mean masks a serious negative cell.

## Original gates, unchanged

Both original gates fail. Against B2, both seed Final increments are positive (163 +0.7114 pp, 164 +0.0235 pp), but the mean +0.3674 pp is below the required +0.5 pp; order2 Incoming -0.5463 pp is below -0.5 pp, and seed163/order2 Drishti cup -6.3889 pp breaches -5 pp. Against F5, seed163/order1 REFUGE cup -6.7712 pp and RIM cup -5.2402 pp breach the cell floor. No threshold is relaxed, and no checkpoint is selected after readout.

## Activation and limitations

16,960 class-cone calls used 50,880 reference VJPs; 14,783 calls projected, 8,552 had multiple active constraints, and 5,260 had a nonnegative aggregate reference but a negative individual-class direction. This establishes the structural conflict on V15 states, not on archived V13 states. A/B/R U gradients were nonzero on all 16,960 active calls. The class constraints were active; activation did not establish correct protection or old-domain generalization. Parameter-gradient feasibility does not constrain Adam's actual displacement, and current U need not cover old domains.

## Actual costs and next authorized action

21,200 formal updates, source 0. Campaign actual formal total 382,273, including the historical 673 reboot rollback/lost updates; copied sealed 5,300 are not counted twice. Qualification cumulative CPU 864 / generated CUDA 211 / real L 44. Current qualification 32/10/2 includes the intentional failure. The separate generated three-class supplement adds 3 reference VJPs and 1 student forward, zero optimization. Formal cost includes 89,040 student forwards, 50,888 teacher forwards (16,960 anchor calls), 64 fixed diagnostic VJPs and 50,880 reference VJPs. See CAMPAIGN_COSTS.json for the complete operation ledger. Historical failures and five missing diagnostic rows remain preserved.

After public delivery verification, proceed to the already authorized GUA direction-alignment candidate on the common V13 base, without stacking the V15 class cone. Mathematical checks have passed but do not replace native Adam/checkpoint/teacher/deployment qualification. No additional formal calls have yet been made for the four-paper queue; the reserved 84,800 is an allowance, not consumption.

These are repeatedly reused development-patient results, not independent confirmation, SOTA, or clinical benefit. Source, protocol and anonymous aggregates are public; patient-level records, weights, private configuration and raw logs remain on NAS.

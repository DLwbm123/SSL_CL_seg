# V23 complete: relation retention improves the mean but fails B2 confirmation

Execution 05347ae9ae4ce989229570e9b9ae269cea84c034. All 4 jobs EXIT0, 8 stages SEALED, controller/process EXIT0 and COMPLETION_AUDIT PASS. Exactly 21200 formal updates, source0 and 20 uniform seen-domain evaluations after every seal. No scientific retry, partial efficacy selection or extra evaluation.

| Metric | V23 % | vs F5 pp | vs historical B2 pp | vs V21 pp | vs V13 pp | vs V22 pp |
|---|---:|---:|---:|---:|---:|---:|
|Final|68.6114|+3.2005|+0.7103|+0.3319|+0.7857|+2.3139|
|Old|67.5310|+4.3622|+1.2900|+0.6167|+1.2038|+3.5679|
|Incoming|70.7722|+0.8772|-0.4490|-0.2378|-0.0507|-0.1941|
|Forget|9.2906|-4.3851|-1.4101|-0.7191|-1.1325|-3.1082|

Original F5 gate PASS; historical B2 gate FAIL. B2 mean Final +0.7103pp exceeds the +0.5pp threshold, but seed164 Final -1.2228pp fails the per-seed positive requirement, order2 Incoming -1.1174pp fails the -0.5pp limit, and 164/order1 RIM rim -5.8850pp fails the -5pp class limit. Seed163 Final +2.6435pp does not compensate for seed164. All positive and negative cells, B0 and V10/V13/V15/V17/V19/V20/V21/V22 comparisons remain in completion/RESULTS.json.

| Seed/order | Final % | Old % | Incoming % | Forget % |
|---|---:|---:|---:|---:|
|163/1|71.1433|69.9601|73.5097|5.6953|
|163/2|65.6369|63.2199|70.4710|14.0383|
|164/1|68.0520|67.5058|69.1444|8.4245|
|164/2|69.6136|69.4385|69.9638|9.0043|

Compared with V21, mean Final/Old/Incoming changes are +0.3319/+0.6167/-0.2378pp. Order1 Final/Old fall -0.5836/-0.7401pp; order2 Final/Old improve +1.2474/+1.9736pp. Both seed164 settings lose Incoming versus V21 (-1.1046/-1.2110pp). Thus mean retention improvement accompanies mixed adaptation outcomes.

163/O1 old RIM_ONE_r3 rim: 72.9089% after stage1 to 60.4297% after stage2 (-12.4792pp).

163/O2 old Drishti_GS cup: 72.8394% after stage1 to 49.5541% after stage2 (-23.2853pp).

164/O1 old RIM_ONE_r3 rim: 71.7485% after stage1 to 58.0614% after stage2 (-13.6871pp).

164/O2 old Drishti_GS cup: 75.0185% after stage1 to 63.3838% after stage2 (-11.6347pp).

## Activation, costs and interpretation

All 21200 relation calls had 84800 supported edges and 21200 triplets; 21192 calls had positive relation loss (8 zero). The 38160 extra current-L entry-teacher forwards were counted: teacher full forwards 89048, student 89040. Original U A/B/R nonzero calls and reference VJPs/anchor forwards each16960, class reductions50880, aggregate projections9253, R diagnostic VJPs64. Hierarchy coarse/part positive21200, alignment20332, edges42400, featureviews38160, eligiblepixels6118393141 and seamexcluded68784459. Training autograd55184, readout full forwards1190. These establish execution/support, not teacher correctness or historical-domain protection.

Controller7809.2444seconds with shared GPUs; each job runtime and peak memory are in JOB_EXITS.json. Maximum reserved2344615936bytes. Extra teacher inference makes V23 a more expensive method; shared-GPU runtime does not establish causal speed differences. Cumulative method formal509473; including completed B1 5300:514773. Qualification method1120CPU/287generatedCUDA/58currentL, including B11152/297/60. Historical673 rollback updates,5NA and all prior failed costs remain counted. Completion added no optimizer/model/evaluation calls.

The added current-L relation term improves several old cup cells and the overall mean relative to V21, but does not meet the frozen confirmation objective. Current-L geometry is still a proxy for historical structure. A further mathematical limitation deserves analysis: independently rotating student representations can preserve all distance/angle relations while changing a fixed readout. V23 qualification intentionally checks that invariance; it is not evidence that such rotations caused any patient failure. Next inspect whether a generic class-balanced teacher feature orientation anchor can address this limitation without adding a teacher, head, bank or historical data. No next run or qualification is registered by this report.

The relation component is an ancestry-specific RKD2019 adaptation, not a full RKD/HiPoSeg reproduction or a verified2026 medical method. The native experiment remains2D fundus segmentation. Historical controls and repeatedly reused development patients do not establish independent generalization, clinical benefit, SOTA or novelty.

Source, frozen protocol, complete anonymous positive/negative results, sealed aggregate diagnostics, operations, costs, exits and report are delivered publicly. Patient rows/images/labels, weights, private configuration and raw logs remain NAS.

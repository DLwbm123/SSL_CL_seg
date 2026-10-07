# V8.3 complete-action reward supervision: final result

**STOP_DENSE_REWARD_NO_PRACTICAL_GAIN.** Engineering complete: 288 unique rewards, both final actors (601/602), and 20 development endpoints sealed before evaluation. The original frozen screen was recomputed unchanged. Both seeds fail it; neither improves on uniform. No GRPO efficacy or independent-generalization claim is supported.

| Method | New Dice | Old Dice | Utility |
|---|---:|---:|---:|
| NATIVE | 0.734741107 | 0.812819198 | -0.073255628 |
| FIXED_BEST (action 8) | 0.737623854 | 0.810925372 | -0.072268197 |
| UNIFORM_ACTION | 0.737752672 | 0.812890666 | -0.070331457 |
| DENSE 601 | 0.736709660 | 0.812735751 | -0.071903184 |
| DENSE 602 | 0.736709660 | 0.812735751 | -0.071903184 |

Both dense actors are −0.001571727 utility, −0.104301 pp new Dice and −0.015491 pp old Dice versus uniform. They sample identical action sequences in these four matched development streams, despite different probabilities. Absolute forgetting against the frozen memory is about 5.04 pp. Preserve both seeds; these are not independent patients or independent student replications.

New physical cost: 34,008 student and 133 actor updates. Cumulative campaign cost: 224,853 student and 837 actor updates; the historical common source adds 8,000 student updates separately. Dense information preparation costs 29,600 student updates. Historical forward-call counters/kernel GPU time/peak VRAM were not recorded; per-job process elapsed time is available and labeled accordingly. Full image-role budget is 16 memory-fit + 8 adaptation-fit + 8 training-query + 8 development-query images, with adaptation supports 2/8 being subsets rather than the total label budget.

[Complete anonymous reward table](../v83_readout/V83_ACTION_ROWS.jsonl), [all endpoints and structure-channel scores](../v83_readout/V83_ENDPOINT_RESULTS.csv), [original and cumulative accounting](../v83_readout/V83_PHYSICAL_ACCOUNTING.json), and [independent readout interpretation](../v83_readout/FINAL_INTERPRETATION.md) are included. Source/ENTRY hashes are in CHECKPOINT_PROVENANCE.json and final actor hashes in SEALED_INPUTS.json in the readout directory. Private weights, images, role mappings and raw state checkpoints remain unpublished. The subsequent readout cannot overturn this original STOP.

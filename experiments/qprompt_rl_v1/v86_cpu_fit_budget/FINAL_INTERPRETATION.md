# V8.6 fixed optimization-budget diagnostic

**READY_FOR_SEPARATELY_PREREGISTERED_DEV_CONFIRMATION**

| Evaluation | Arm | Seed | Raw expected reward | Minus global | Minus time | Target KL |
|---|---|---:|---:|---:|---:|---:|
| LOCO | RAW_64_REUSED | 601 | 0.001257515 | 0.000261105 | -0.000019948 | 0.805498426 |
| LOCO | RAW_64_REUSED | 602 | 0.001255830 | 0.000259420 | -0.000021633 | 0.808663424 |
| LOCO | Z_64_REUSED | 601 | 0.001245077 | 0.000248667 | -0.000032387 | 0.699282411 |
| LOCO | Z_64_REUSED | 602 | 0.001193704 | 0.000197294 | -0.000083759 | 0.777350897 |
| LOCO | RAW_1024 | 601 | 0.002118826 | 0.001122416 | 0.000841363 | 0.335809068 |
| LOCO | RAW_1024 | 602 | 0.002099726 | 0.001103316 | 0.000822263 | 0.346008864 |
| LOCO | Z_1024 | 601 | 0.002450698 | 0.001454288 | 0.001173235 | 0.141137227 |
| LOCO | Z_1024 | 602 | 0.002457365 | 0.001460955 | 0.001179901 | 0.119860742 |
| in_table | RAW_64_REUSED | 601 | 0.001367394 | -0.000448557 | -0.000557557 | 0.684399610 |
| in_table | RAW_64_REUSED | 602 | 0.001368311 | -0.000447640 | -0.000556640 | 0.683312739 |
| in_table | Z_64_REUSED | 601 | 0.001454215 | -0.000361736 | -0.000470736 | 0.535489451 |
| in_table | Z_64_REUSED | 602 | 0.001434865 | -0.000381086 | -0.000490086 | 0.564390331 |
| in_table | RAW_1024 | 601 | 0.002260268 | 0.000444317 | 0.000335316 | 0.176000261 |
| in_table | RAW_1024 | 602 | 0.002231816 | 0.000415865 | 0.000306864 | 0.190078830 |
| in_table | Z_1024 | 601 | 0.002538670 | 0.000722719 | 0.000613719 | 0.002670945 |
| in_table | Z_1024 | 602 | 0.002538047 | 0.000722096 | 0.000613095 | 0.003059859 |

All 36 update64 parameter prefixes match their V8.5 counterparts exactly; optimizers then continue without reset.
Only update1024 models are evaluated; loss logs at earlier updates are descriptive and are not used for selection.
This is adaptive shared-source D1 context holdout, not independent patient/image validation. No new image read or student update occurred.
New cost: 36,866 CPU actor updates. Historical dense information still costs 29,600 student updates and is not free.
V8.3, V8.3-R and V8.5 STOPs remain. V8.4 is NOT_RUN. Any development experiment must be separately preregistered.

## Interpretation and next step

The registered primary gate passes for both seeds. Relative to each fold's time-fixed control, Z_1024 gains 0.001173235/0.001179901 raw reward; its mean gain over Z_64 is 0.001234641. Full-table target KL falls from roughly 0.55 to roughly 0.003. RAW_1024 also improves, while standardization adds a further held-context advantage at the same 1,024-update budget. Exact update64 prefixes rule out a changed initialization or an unnoticed early-trajectory change as the explanation for this comparison.

This supports an optimization-budget limitation of the previous 64-update recipe in this saved-table setting. It does not show better trained-student endpoints, fix deployment-state shift, establish GRPO efficacy, or establish independent generalization. The next separate protocol should evaluate both primary seeds against matched information controls in the original four D1 development contexts, saving the previously missing step200 deployment features and sealing all new endpoints before readout. V8.4 remains unexecuted.

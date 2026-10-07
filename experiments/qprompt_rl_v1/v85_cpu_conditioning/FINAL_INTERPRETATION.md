# V8.5 CPU attribution

**STOP_V85_PRIMARY_MAPPING_GATE_FAILED**

| Evaluation | Arm | Seed | Expected reward | Minus uniform | Minus global | Minus time | Entropy |
|---|---|---:|---:|---:|---:|---:|---:|
| LOCO | RAW_T1 | 601 | 0.001026636 | 0.000364301 | 0.000030226 | -0.000250827 | 2.031450362 |
| LOCO | RAW_T1 | 602 | 0.001027500 | 0.000365165 | 0.000031090 | -0.000249963 | 2.031992923 |
| LOCO | RAW_T025 | 601 | 0.001257515 | 0.000595180 | 0.000261105 | -0.000019948 | 1.718146292 |
| LOCO | RAW_T025 | 602 | 0.001255830 | 0.000593495 | 0.000259420 | -0.000021633 | 1.718857803 |
| LOCO | Z_T1 | 601 | 0.001241357 | 0.000579022 | 0.000244947 | -0.000036106 | 2.049040924 |
| LOCO | Z_T1 | 602 | 0.001194349 | 0.000532014 | 0.000197939 | -0.000083114 | 2.074877558 |
| LOCO | Z_T025 | 601 | 0.001245077 | 0.000582741 | 0.000248667 | -0.000032387 | 2.033940391 |
| LOCO | Z_T025 | 602 | 0.001193704 | 0.000531369 | 0.000197294 | -0.000083759 | 2.075850146 |
| in_table | RAW_T1 | 601 | 0.001114192 | 0.000451857 | -0.000701759 | -0.000810759 | 2.034099256 |
| in_table | RAW_T1 | 602 | 0.001112922 | 0.000450587 | -0.000703029 | -0.000812030 | 2.034393680 |
| in_table | RAW_T025 | 601 | 0.001367394 | 0.000705059 | -0.000448557 | -0.000557557 | 1.718796820 |
| in_table | RAW_T025 | 602 | 0.001368311 | 0.000705976 | -0.000447640 | -0.000556640 | 1.720144538 |
| in_table | Z_T1 | 601 | 0.001368966 | 0.000706631 | -0.000446985 | -0.000555986 | 2.016151597 |
| in_table | Z_T1 | 602 | 0.001363938 | 0.000701603 | -0.000452013 | -0.000561013 | 2.020260351 |
| in_table | Z_T025 | 601 | 0.001454215 | 0.000791880 | -0.000361736 | -0.000470736 | 1.952513109 |
| in_table | Z_T025 | 602 | 0.001434865 | 0.000772530 | -0.000381086 | -0.000490086 | 1.989030744 |

All metrics are expected raw reward on saved tables. No new student endpoints or image queries were run.
The primary combined arm and its gate were fixed before these fits. Secondary arms do not replace it.
This is an adaptive diagnostic with shared D1 images and model sources, not independent confirmation or a claim of GRPO efficacy.
Target temperature affects only supervised preferences; categorical deployment temperature remains 1.
Physical new cost is 3,459 CPU actor updates; the 29,600 student-update information generation cost remains attributable to all four arms.
RAW_T1 is reused from V8.3-R, not an additional repetition. All new arms retain both final seeds and 64 updates per fit.

## Interpretation

The primary combined arm fails both registered conditions: neither seed beats its fold-trained time control, and its mean gain over the reused raw/T1 policy is only 0.000192322 rather than 0.0005. No student deployment follows from this round, and a secondary arm is not substituted.

Sharpness alone improves raw-input LOCO reward by about 0.000230, but still loses to the time control. Standardization alone also improves the baseline, while adding sharpness after standardization yields little additional LOCO improvement at 64 updates. The two effects are not additive. The full-table combined model remains below global and time controls despite stronger exact targets; the fixed short optimization recipe is therefore still a plausible limitation to test, not an established explanation of endpoint failure.

These results support one bounded follow-up that changes only the actor optimization duration, retaining both conditioning levels and the fixed sharpened target. Its definition must be frozen before new fits; this round's STOP and all four arms remain unchanged. No target-temperature sweep or seed search is justified by this result.

# SR-GAS V0.1a Freeze for V0.2

Status: `SRGAS_V0_1A_FROZEN_FOR_V0_2`

This record freezes the V0.1a outcome before any V0.2 implementation or training.

- V0.1a hard stop was correct: `SRGAS_PILOT_GATE_FAILED`.
- No V0.1a full runs exist.
- V0.2 does not relax the original worst-trajectory gate of `0.015` on either REFUGE or RIM-ONE-r3.
- V0.2 changes only sensitivity timing and noise onset.
- There is no architecture change.
- There is no R2C formula change.
- The V0.1 model-path blocker and the V0.1a class-space amendment remain immutable evidence.
- The existing seed-0 REFUGE common parent remains byte-identical with SHA-256 `8f188ba27074ecb09a689377982774e6cf59e8c1c652d3927be54fd7c377bf55`.

Frozen V0.1a trajectory evidence:

| Evaluation site | Worst A5-vs-A1 drop | Original gate | Step-1000 A5-vs-A1 delta |
|---|---:|---:|---:|
| REFUGE | 0.031446 | 0.015 | +0.032163 |
| RIM-ONE-r3 | 0.016931 | 0.015 | -0.002011 |

The step-1000 recovery does not override the failed worst-point gate.

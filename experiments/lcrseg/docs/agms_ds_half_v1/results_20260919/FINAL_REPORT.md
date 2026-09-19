# AGMS DS HALF: completed development experiment

**COMPLETE_DS_HALF_AWAITING_SCIENTIFIC_REVIEW**. Both new stages completed; both actual deployed-model proofs VERIFIED. Execution commit `44c1da8a5a8a75021a0298d42bd8f9088e9b059a` remained clean. External R1 review digest `24f06ae4349f51f5b4491feb573c0d6ca0fe4db9e37e39b8e987705aef02c120` and separate current-user launch receipt are retained outside the execution checkout. This results-only publication is not a new production implementation.

## Result

**DS_PRESSURE_SUPPORTED=false.** Halving DS improves mean Final by only **+0.090623 percentage points** over old A5; it remains **−1.734926 points below B2**. O1 worsens relative to old A5 while O2 improves. This is a scientific negative result with no formal engineering failure.

A5 in the raw CSV means the new DS=.125 model. A5_CONTROL means the old DS=.25 model, A0 means B2. Four endpoints are historical imports; only the two HALF endpoints are new. Seed163 only; O1=REFUGE→RIM→Drishti and O2=REFUGE→Drishti→RIM, each from its own original B2 stage1 prefix. Source/stage1 new training=0.

Values below are raw rim/cup macro Dice, not disc-union Dice. Orders are averaged within the same seed, not treated as independent seeds.

| Method | Final | Old | Incoming | Forget |
|---|---:|---:|---:|---:|
| B2 (historical) | 0.657466 | 0.626902 | 0.718594 | 0.145186 |
| Original A5 DS=.25 (historical) | 0.639210 | 0.599444 | 0.718743 | 0.172644 |
| HALF DS=.125 (new) | 0.640117 | 0.603546 | 0.713257 | 0.168541 |

### Paired differences (percentage points)

| Contrast | Order | Final | Old | Incoming |
|---|---|---:|---:|---:|
| HALF-A0 | 1 | -3.334219 | -5.231633 | +0.460610 |
| HALF-A0 | 2 | -0.135634 | +0.560556 | -1.528013 |
| HALF-A0 | mean | -1.734926 | -2.335539 | -0.533701 |
| HALF-old-A5 | 1 | -0.411543 | -0.181243 | -0.872145 |
| HALF-old-A5 | 2 | +0.592789 | +1.001751 | -0.225134 |
| HALF-old-A5 | mean | +0.090623 | +0.410254 | -0.548640 |

The unchanged registered gate requires versus old A5: mean Final≥+.003, mean Old≥+.003, every-order Final≥−.002 and Incoming≥−.005. Mean Final fails, mean Old passes; O1 Final and Incoming fail; both O2 conditions pass. Full per-condition outcomes are in RELEASE_VALIDATION.json. The independent interpretation tiers from the targeted review also fail: B (near B2 in both orders) is false and C (practical gain over B2) is false; these supplementary interpretations do not modify the registered gate.

### Separate old-domain changes versus B2 (percentage points)

| Order | Role | Domain | rim | cup | disc_union | macro Dice |
|---|---|---|---:|---:|---:|---:|
| 1 | source | REFUGE | -2.099089 | -2.070600 | +0.702429 | -2.084844 |
| 1 | first_target | RIM_ONE_r3 | -11.760687 | -4.996156 | -5.311098 | -8.378421 |
| 2 | source | REFUGE | +8.171107 | -1.439101 | +5.043554 | +3.366003 |
| 2 | first_target | Drishti_GS | +2.587891 | -7.077674 | +2.002234 | -2.244891 |

All incoming-domain values and changes are retained in HALF_DOMAIN_COMPARISONS.csv and DOMAIN_METRICS.csv. Source gains must not hide first-target losses; disc/background quality does not establish correct rim/cup boundaries. Final=(2×Old+Incoming)/3; common-prefix DeltaForget=−DeltaOld is the same observation, not independent confirmation.

## Acceptance and cost

- 2/2 new stages: O1 Drishti2100 + O2 RIM3200. **5300 scientific and physical formal optimizer calls**, zero failed formal sessions or retries.
- Native synthetic CUDA **11**, including **2 prescribed after_optimizer failures** charged to physical cost; all required qualification cases passed. Real B2 prefix acceptance **2/2 VERIFIED**; L-only smoke **8** passed and discarded. Total real optimizer calls **5308**.
- Existing CPU **21** in one attempt retained, including2 prescribed failures; no CPU rerun. Prior AGMS80 and earlier134 retained separately. New P0 images/updates=0; old26-L screening metadata only.
- **8 diagnostic points /32 extra VJPs**. Reports **6 final /18 domain /9 paired**. No fabricated historical diagnostics. Coverage and risk report retained.
- All formal, prefix, qualification and integrity cost sessions closed PASS. Full operation counts, memory peaks and actual-model verification cost retained in COST_AND_COMPLETION.json. Summed formal worker time including endpoint evaluation: **33.18 minutes**; full formal phase wall time about33.39 minutes. Completion about2026-09-19 14:14:05 China time.
- Peak formal CUDA allocation **1.768 GiB**, reservation **1.895 GiB**. These are PyTorch recorded peaks, not exclusive GPU resource use.

The later release check matched retained integrity proofs to current file stat identities, all sessions and ledger counts; no weights were reloaded, no additional optimizer or VJP ran. Public files omit private paths, payloads, patients, per-patient scores, images and labels. The original runtime report and approval history remain unchanged; this expanded report exists only in the independent result location.

## Interpretation and stop

Lowering global DS changes both auxiliary-head and main-network gradients, then may alter EMA/risk/masks. It does not isolate upstream interference and does not ensure Adam updates scale by one half. The result fails to establish useful recovery or superiority to B2, and does not justify another scalar halving scan.

The training-L risk diagnostics are not independent calibration. Four sparse points per trajectory are not a full gradient distribution; same-state U uniform/risk-mask comparison was not measured (NA). This is tuning on already observed development domains, not independent patient generalization, significance, original KI reproduction, SOTA or a complete new continual-learning trajectory.

**No automatic follow-up.** OBS0/ROUTE_HALF, fine downgrade, more DS values, seed164, source/stage1 retraining and P1R/P2/P3 remain unauthorized. Await scientific review; any next study needs separately specified scope and required implementation/review gates.

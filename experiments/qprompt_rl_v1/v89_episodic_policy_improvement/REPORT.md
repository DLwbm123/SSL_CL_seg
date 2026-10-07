# V8.9 episodic policy improvement

**STOP_V89_NO_RL_INCREMENTAL_GAIN**

| Method | New | Old | Utility | Absolute forgetting |
|---|---:|---:|---:|---:|
| NATIVE | 0.734741107 | 0.812819198 | -0.073255628 | 0.050268013 |
| FIXED_BEST | 0.737623854 | 0.810925372 | -0.072268197 | 0.052161839 |
| UNIFORM_ACTION | 0.737752672 | 0.812890666 | -0.070331457 | 0.050196545 |
| TIME_FIXED | 0.737614123 | 0.811875904 | -0.072033168 | 0.051211307 |
| Z_1024_601 | 0.739587104 | 0.812129237 | -0.069718187 | 0.050957974 |
| Z_1024_602 | 0.739216264 | 0.811548013 | -0.070577540 | 0.051539198 |
| LOCAL_CE_601 | 0.739543077 | 0.812147744 | -0.069772382 | 0.050939467 |
| LOCAL_CE_602 | 0.739543077 | 0.812147744 | -0.069772382 | 0.050939467 |
| EPISODIC_CE_601 | 0.737905312 | 0.811192308 | -0.071846252 | 0.051894903 |
| EPISODIC_CE_602 | 0.737706015 | 0.811183345 | -0.072012924 | 0.051903866 |
| EPISODIC_RL_601 | 0.738736102 | 0.811767960 | -0.070548958 | 0.051319251 |
| EPISODIC_RL_602 | 0.738668056 | 0.811847810 | -0.070550735 | 0.051239401 |
| Z_1024 | 0.739401684 | 0.811838625 | -0.070147864 | 0.051248586 |
| LOCAL_CE | 0.739543077 | 0.812147744 | -0.069772382 | 0.050939467 |
| EPISODIC_CE | 0.737805664 | 0.811187826 | -0.071929588 | 0.051899385 |
| EPISODIC_RL | 0.738702079 | 0.811807885 | -0.070549847 | 0.051279326 |

The primary tests incremental RL gain over identical-data episodic distillation and all frozen controls.
Secondary distillation gains do not replace a failed primary. All48 endpoints were sealed before new evaluation.
This is one full-action policy-improvement step on repeatedly observed D1 development, not GRPO or independent generalization.
Private states/weights remain on NAS; V84 remains NOT_RUN. All costs and query counts are in COSTS.json.

## Interpretation and limits

V89 completed normally. The preregistered primary failed: the RL arm beats episodic CE at both actor seeds, but loses to uniform, the frozen V86 actor and LOCAL_CE. Both practical tradeoff gates fail. The secondary episodic CE minus local CE utility is -0.002157206; aligning the credit horizon alone did not improve this development result. Actor seeds share the student stream, initial entry and categorical random draws; these are paired policy fits, not independent student replications. LOCAL_CE's two seeds produce identical endpoint scores, so their equality is not an uncertainty estimate.

Foreground soft Dice (macro over the two foreground channels), not background-inclusive or thresholded hard Dice: RL new 73.870208%, old 81.180788%; uniform new 73.775267%, old 81.289067%; LOCAL_CE new 73.954308%, old 81.214774%; EPISODIC_CE new 73.780566%, old 81.118783%. RL minus uniform is +0.094941 percentage points new, -0.108278 points old, utility -0.000218390. RL minus episodic CE is +0.089642 points new, +0.062006 points old, utility +0.001379741. All per-channel and per-context endpoints are preserved.

RL is a full-action expected-return improvement with KL to its own frozen initial policy, using Monte Carlo continuation values. It is neither GRPO nor an on-policy multi-round algorithm. CE and RL use the same 32 states, nine complete actions per state, 1024 optimizer steps, two actor seeds and deployment protocol; CE uses the registered entropy loss while RL also has its preregistered KL term. The comparison is of these complete objectives, not a proof that RL generally outperforms distillation.

Exploratory observation after the primary readout: training local and episodic forgetting penalties are zero in every one of the 288 rows. The development penalties are positive. This is a concrete reward-signal mismatch to investigate; it does not prove causality or authorize a retroactive change to V89's reward. Any alternative reward needs a separate plan, matched controls and cost ledger.

## Verification, cost and disclosure

All 27 jobs exited 0; 48 endpoints (24 reused, 24 new) were sealed before the four new evaluation jobs started. The primary was recomputed from all endpoints and exactly matched DECISION.json. There were 46,408 successful native student updates and 6,146 successful actor updates, no failed optimizer attempts. Training query image evaluations: 5,056; development: 288. These repeatedly access eight existing training-query and eight existing development-query images, with zero new unique images. The memory-fit budget remains 16 labelled images and support sizes 2/8; overlapping role accounting and identity uncertainty prohibit adding these into a claimed patient-independent cohort size. No hidden U labels were read.

Cumulative campaign cost through V89: 276,869 student updates plus the separately recorded 8,000 common source updates (284,869 including source), and 48,333 actor updates. V88 added zero updates. Earlier failures/stops and V84 NOT_RUN remain unchanged. Physical optimizer ledgers, fit summaries, complete reward rows and job receipts are published; private tensors, weights, image/label data, role mappings and host-specific paths remain on NAS. This is repeatedly observed current-D1 development, with simulated old-domain retention, not independent or real subsequent-domain evidence. No paper-ready positive claim is supported.

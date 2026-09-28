# V3A final interpretation

Completed P0/P1/P2. Primary horizon remains h5. All values below are feedback-quality differences relative to FINE, **not Dice or Dice percentage points**.

## 1. Is there measured action opportunity?

The four-cell mean post-hoc optimistic oracle is 0.000194741064. This is a finite, noisy observed maximum, not a generalization bound or a guarantee of long-term benefit. DINO/Drishti h5 has repeat error 0.000150021 and a resulting numerical dead zone 0.000750106; its oracle 0.000225352 lies below that dead zone. The other three h5 cells have dead zone 0.000001. Opportunity therefore depends on the cell, and the DINO/Drishti apparent opportunity is not clearly separated from measured repeat noise.

## 2. Does online feedback transfer to audit feedback?

Mean online-selected audit value is 5.84830996e-05 with the registered dead-zone rule; raw selection gives 8.65035108e-05. Three h5 cells have positive transfer. DINO/Drishti makes zero dead-zone interventions (all 16 online margins are uncertain), so its transfer is zero. Across the 28 actual h5 interventions, 7 are negative beyond the relevant dead zone. Positive average transfer does not make individual interventions reliable. Per-action rates and values are in HORIZON_TRANSFER.csv.

## 3. Does the observed state predict useful choices?

FI_POLICY and its shuffled-reward control are nearly identical in mean h5 audit value: their difference is -1.37330449e-09. FI_POLICY minus NC_OPT is 1.02712177e-07. These descriptive differences do not establish a useful context contribution. LIN_VALUE averages -3.97742765e-05, worse overall, although it is positive on UNet/Drishti where FI_POLICY is negative. This mixed result does not establish either universal state insufficiency or an RL-specific advantage.

## 4. Can full-information policy optimization learn useful behavior?

All 12 synthetic problem/seed combinations passed their registered behavior checks, using 3,072 optimizer calls. The real-data primary result is FI_POLICY−FINE = -1.87105199e-05, negative. It is also below STATIC_PRIOR by -1.01674252e-05. Synthetic learnability therefore did not translate into positive mean held-out medical-feedback value with this frozen learner, scale and entry state. No hyperparameter or checkpoint selection was performed.

| h5 method | Mean audit value |
|---|---:|
| FINE | 0 |
| STATIC_PRIOR | -8.54309474e-06 |
| NC_OPT | -1.88132321e-05 |
| LIN_VALUE | -3.97742765e-05 |
| FI_POLICY | -1.87105199e-05 |
| FI_POLICY_SHUFFLE | -1.87091466e-05 |

## 5. Is there evidence for a future closed-loop pilot?

The frozen proposal criterion is not met: 0/4 FI_POLICY cells exceed their numerical dead zone, the UNet mean is negative, and FI_POLICY does not outperform both the prior and shuffle control. DINO's positive backbone mean does not reverse this conclusion. No V3B, R2 or R3c was launched. This result is not a claim that all RL is ineffective; it applies to the registered short intervention, feedback, state and learner.

## Coverage, diagnostics and limitations

P0 covers 2,224 effective decisions/contexts from 36 adaptive V2 trajectories and four R3a panels. Fine was observed in 1,825 and all three actions in 1,555. Missing fine contrasts remain missing, and observed-action coverage is behavior-dependent. P0 reports exact-zero uncertainty separately; P1 uses measured numerical dead zones. These are descriptive analyses, without IPS/DR or independent-sample claims.

P1 completed 64 primary scenes, 192 main branches and 16 repeat branches. Each branch used five steps; only the first action varied. Root restoration, isolated CPU Adam storage, local scheduler 1–5, Adam 2001–2005 and per-step teacher updates were verified. Nonfine h1/h5 audit signs changed in 52/128 pairs; coarse/skip ranking changed in 32/64 scenes. Fine's definitional zero was excluded from these counts. Branch diagnostics contain teacher/student differences, KL and actual U gradients. Their action field denotes the first-step intervention; aggregates include the following four FINE steps, so SKIP branches can have nonzero mean U gradients. Coverage is not pseudo-label accuracy.

Each domain supplies 12 or 14 distinct U groups per cell; every repeated U identity remained within one fold. All 64 neural fits used 256 updates. There were also 32 constant analytic solutions, 32 ridge fits and 128 held-out analytic ridge decisions. Grouped OOF is not patient or L-feedback independence; shared feedback images, one prefix per cell and fixed old R3a reward scales limit inference about later training states.

## Engineering event and complete cost

The first DINO synthetic qualification failed an implementation-added repeat-endpoint tolerance, while full root restoration passed. The patch records CUDA repeat variability diagnostically and preserves exact root/state restoration and storage isolation; it does not change student mathematics, precision, rewards or the registered numerical dead-zone rule. Existing completed receipts remain bound to their original code hashes. Ten repeated qualification calls were charged to replay. Original failed qualification calls were retained in the ledger; no budget or deadline was reset.

Total physical calls: 1,070 student (960 main + 80 null + 20 synthetic + 10 replay) and 19,456 controller (16,384 main + 3,072 synthetic), totaling 20,526. Every attempt has a successful optimizer receipt; one qualification acceptance failure is separately recorded. There are zero unresolved optimizer calls, zero retained student updates, zero new medical endpoints and zero real-L smoke calls. All caps were respected. Execution finished at 2026-09-28 23:29:33 Beijing time. Reports are committed locally; GitHub publication awaits explicit current-stage authorization.

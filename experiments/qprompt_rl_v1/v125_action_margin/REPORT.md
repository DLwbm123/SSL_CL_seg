# V125 — no practical policy margin on the frozen V124 table

**NO_PRACTICAL_POLICY_MARGIN_ON_FROZEN_TABLE**. Even an outcome-informed selector over the five already measured whole-trajectory recipes cannot pass the frozen practical and absolute gates on this finite table. This is an executed retrospective bound diagnosis, not learned-policy performance, independent confirmation or a universal impossibility result.

Run`v125_margin_20261009T122929Z`, execution commit`2cd882726d611c05bd29d255faaa727bcf660694`, exit0; source/protocol proxy-published and anonymously verified before applying the analysis. Runtime0.164seconds, no GPU used. Synthetic positive/tie/insufficient-practical/negative-absolute tests PASS. Input is all160 sealed V124 trajectories and its verified completion/publication; no subsets, newly observed outcomes or new model/image calls.

## Optimistic bounds against fixed controls

Each metric is separately maximized per condition/stream, then equally averaged. Different columns may require different choices; these are optimistic coordinatewise bounds, not a single feasible policy. Values are percentage points for Dice changes; reward differences are multiplied by100.

| comparator | maximum weighted-new difference | maximum final-new difference | maximum old difference | maximum reward difference ×100 | potentially positive contexts /16 | necessary conditions |
| --- | --- | --- | --- | --- | --- | --- |
| BASE | +0.07109064 | +0.07114843 | +0.00993693 | +0.04301668 | 5 | False |
| RANDOM_MATCHED | +0.09050755 | +0.09218393 | +0.02106670 | +0.07356337 | 15 | False |
| UNIFORM_MATCHED | +0.07860940 | +0.07948480 | +0.01828265 | +0.05888116 | 15 | False |
| OFF | +0.92478470 | +1.01833880 | +1.47577667 | +2.36255049 | 15 | True |

Against BASE, maximum final-new improvement is0.07114843pp below the0.2pp practical threshold; maximum old improvement is0.00993693pp below0.5pp. Thus neither practical arm is possible, even before requiring simultaneous old/new tradeoffs. Maximum reward difference0.000430166837759 also falls below0.0005; only5/16 contexts have any possible positive reward difference, below12. These are separate failures, not a borderline reward rounding issue. The random and uniform controls also fail the practical necessary condition.

## Actual outcome-informed oracles

These selections use their own evaluation outcomes and cannot be deployed or reported as learned performance. Ties use the predeclared method order. Full96 selections are in ORACLE_ROWS.json.

| optimization objective | weighted new vs ENTRY (pp) | final new vs ENTRY (pp) | final old vs ENTRY (pp) | reward ×100 | selections |
| --- | --- | --- | --- | --- | --- |
| reward | -0.06240244 | -0.08917266 | +0.90518617 | +0.84278373 | {"BASE": 22, "IGNORE_C": 1, "OFF": 3, "RANDOM_MATCHED": 3, "UNIFORM_MATCHED": 3} |
| final_new_gain | -0.03101209 | -0.05700572 | +0.81281217 | +0.78180008 | {"BASE": 17, "IGNORE_C": 8, "OFF": 4, "RANDOM_MATCHED": 1, "UNIFORM_MATCHED": 2} |
| old_gain | -0.08716027 | -0.11380233 | +0.91180671 | +0.82464644 | {"BASE": 20, "IGNORE_C": 1, "OFF": 1, "RANDOM_MATCHED": 9, "UNIFORM_MATCHED": 1} |

The final-new oracle still loses0.05700572pp relative to ENTRY on average. No selection among these measured complete recipes can produce positive pooled final-new learning on this table. The reward oracle keeps BASE22/32 times and selects IGNORE_C only once, yet its new learning remains negative. Oracle selection frequency is descriptive, not a learned action preference.

## Every subgroup's absolute coordinatewise upper bounds

These values again maximize each metric separately before subgroup averaging. They are not jointly attainable metrics from one policy.

| group | weighted new bound (pp) | final new bound (pp) | final old bound (pp) | reward bound ×100 |
| --- | --- | --- | --- | --- |
| fold=0 | +0.38639740 | +0.38787164 | +1.22263669 | +1.59825213 |
| fold=1 | -0.44842159 | -0.50188308 | +0.60097673 | +0.08731533 |
| stream=3 | -0.15504057 | -0.20608543 | +0.86850817 | +0.67967367 |
| stream=4 | +0.09301639 | +0.09207400 | +0.95510525 | +1.00589379 |
| source_step=2000 | -0.01831359 | -0.04951844 | +0.95309974 | +0.91831555 |
| source_step=8000 | -0.04371059 | -0.06449299 | +0.87051368 | +0.76725192 |
| labeled_images=2 | +0.44283225 | +0.48184546 | +1.31435213 | +1.71680251 |
| labeled_images=4 | -0.50485644 | -0.59585689 | +0.50926129 | -0.03123505 |
| condition=brightness | -0.26212629 | -0.30858542 | +0.89333812 | +0.60946420 |
| condition=contrast | +0.20010211 | +0.19457398 | +0.93027530 | +1.07610326 |
| context=0 | -0.09594674 | -0.11166343 | +1.32754222 | +1.23159548 |
| context=1 | +0.74166490 | +0.74893148 | +0.99648871 | +1.73815361 |
| context=2 | +0.21076633 | +0.21370510 | +1.26523152 | +1.47511347 |
| context=3 | +0.57285200 | +0.48008855 | +1.17546245 | +1.74831445 |
| context=4 | +0.91734277 | +0.96508395 | +1.43465176 | +2.34944718 |
| context=5 | +0.46278276 | +0.47779437 | +1.51275508 | +1.97553784 |
| context=6 | +0.15826412 | +0.17191842 | +1.32506490 | +1.48260663 |
| context=7 | +0.12345309 | +0.15711468 | +0.74389689 | +0.78524838 |
| context=8 | +0.25402149 | +0.33905916 | +1.45757757 | +1.71159906 |
| context=9 | +0.71626110 | +0.81315599 | +2.08187662 | +2.67106341 |
| context=10 | -2.10110806 | -2.35172063 | -0.70148036 | -2.80284882 |
| context=11 | -0.44501973 | -0.52770376 | +0.02209917 | -0.42646630 |
| context=12 | +0.36694296 | +0.37154295 | +1.11794136 | +1.48488432 |
| context=13 | +0.17958879 | +0.25085919 | +0.58598369 | +0.57213921 |
| context=14 | -1.80729320 | -2.06660889 | -0.07982403 | -2.05668369 |
| context=15 | -0.75076604 | -0.84364861 | +0.32363981 | -0.45516454 |

Absolute necessary-gate failures:

- fold=1: weighted new upper bound <=0
- fold=1: final new upper bound <=0
- stream=3: weighted new upper bound <=0
- stream=3: final new upper bound <=0

## Consequence and scope

Do not run a full selector/RL campaign whose only decision is choosing one of these five whole-trajectory recipes from ENTRY100. Better prediction cannot exceed the measured table's optimistic maxima. For a random mixture over the same measured recipes, expected scalar utility is a convex combination and cannot exceed the corresponding maximum either.

This does not bound per-update switching: it creates different, unmeasured trajectories. It does not bound other actions, new streams, representations, gradients or the population. This retrospective analysis does not justify inventing a new positive result by reducing gates, selecting a favorable fold, shortening the endpoint, or changing seeds. The ongoing authorized campaign must formulate a genuinely different, observable supervision-usefulness hypothesis before new training, with its own predeclared controls and unchanged evidence boundaries.

No native update, actor update, solve, model forward, image read, query or annotation occurred. Cumulative876309native excluding source8000 /884309 including,181593actor57solves are unchanged; historical1152diagnosticforwards288exposures and512/146CPUactorforwards are preserved. V124's forward-accounting supplement separately records2816 clean readout forwards bypassing module hooks, without any repeated model call. All bounds/oracles/cost/source/protocol/report are anonymous and public; no private artifact was copied. Final delivery requires verified proxy push, remote/public access and NAS FINAL_PUBLICATION receipt.

# V121 prototype increment over EMA and location shuffle — complete

Decision: **NO_RELIABLE_INCREMENTAL_PROTOTYPE_SIGNAL**. All three pre-registered PROTO-versus-control comparisons must pass; individual improvements do not replace that joint requirement. This is development-only target-quality evidence, not model-held-out validation, final Dice gain or RL/campaign success.

Completed 2026-10-09T17:30:51.322541+08:00, elapsed 47.5seconds, root EXIT0 and once-only auditPASS. Run `v121_incremental_20261009T092927Z`; execution/preregistration commit `378963d7ed39d235bc25737957db9433a6be8c88` proxy-pushed and verified remotely/anonymous200 before launch. GPU4/NASwrite-read/neutralargv/no-overlappingEXEC_RUN startup checksPASS. Exactly160forward attempt/success pairs,40prototype-support exclusions,32metric rows and8control summaries audited. All model/EMA/memory/optimizer states unchanged. BASE statistics and fixed candidate repair/harm counts reproduce V120; this is internal measurement consistency, not independent evidence.

## Frozen comparison

The candidate region is the union of V120's original blended-target→prototype-corrected argmax transitions1→0 and2→1, selected from prior development evidence. All8existing ENTRY100 contexts and all40canonical current A_fit image/context exposures are retained. The model has already fitted these images; only prototype support excludes the current target image. Same data/model/seed119/geometry/support ordering is intentionally reused. No condition, seed, threshold or individual transition is chosen after this readout.

BASE keeps the original target. PROTO applies the existing probability exchange on the fixed candidate region. EMA uses the original EMA argmax and identical exchange operation on the exact same region; agreement with the original target gives identity. SHUFFLE moves destination/no-change tokens over the broader selected set within original-target class×EMA class strata, using the single pre-registered per-image generator seed. It preserves destination counts, changed count and class-weight sum, not continuous probability amplitudes. No target truth label enters any control's construction. Every arm keeps each pixel's probability multiset and all unselected targets unchanged.

## All four arms

Rates are equal-context whole-selected effects relative to BASE: positive accuracy and negative true-label NLL delta are better. Counts pool repeated pixel exposures and are not independent samples. These are pseudo-target metrics, not student Dice or native KL(target||student) training losses.

| arm | changed | repaired | harmed | weighted_accuracy_delta | weighted_nll_delta |
| --- | --- | --- | --- | --- | --- |
| BASE | 0 | 0 | 0 | 0.000000000 | 0.000000000 |
| PROTO | 3858 | 3817 | 41 | 0.002034232 | -0.001874445 |
| EMA | 3858 | 3817 | 41 | 0.002034232 | -0.001874445 |
| SHUFFLE | 3858 | 3649 | 209 | 0.001816237 | -0.001870589 |

## All primary comparisons

Each requires mean accuracy>0 and NLL<0, jointly positive in≥6/8contexts, and both directions in every n=2/n=8/source2000/source8000 subgroup. All three must pass. Exact ties fail; empty or zero-effect conditions stay in the denominator. No old practical/absolute/matched-CE/fresh-stream campaign gate changes.

| control | equal_context_weighted_accuracy_delta | equal_context_weighted_nll_delta | joint_positive_contexts | passed |
| --- | --- | --- | --- | --- |
| BASE | 0.002034232 | -0.001874445 | 8 | True |
| EMA | 0.000000000 | 0.000000000 | 0 | False |
| SHUFFLE | 0.000217994 | -0.000003856 | 3 | False |

### PROTO minus BASE

| context | source_step | labeled_images | weighted_accuracy_delta | weighted_nll_delta |
| --- | --- | --- | --- | --- |
| 0 | 2000 | 2 | 0.001187970 | -0.000703386 |
| 1 | 2000 | 2 | 0.003583225 | -0.003399786 |
| 2 | 2000 | 8 | 0.000306428 | -0.000258791 |
| 3 | 2000 | 8 | 0.000454241 | -0.000330516 |
| 4 | 8000 | 2 | 0.002080855 | -0.001578487 |
| 5 | 8000 | 2 | 0.004958948 | -0.005035501 |
| 6 | 8000 | 8 | 0.002317254 | -0.002356940 |
| 7 | 8000 | 8 | 0.001384933 | -0.001332155 |

| group | weighted_accuracy_delta | weighted_nll_delta |
| --- | --- | --- |
| labeled_images=2 | 0.002952749 | -0.002679290 |
| labeled_images=8 | 0.001115714 | -0.001069601 |
| source_step=2000 | 0.001382966 | -0.001173120 |
| source_step=8000 | 0.002685497 | -0.002575771 |

### PROTO minus EMA

| context | source_step | labeled_images | weighted_accuracy_delta | weighted_nll_delta |
| --- | --- | --- | --- | --- |
| 0 | 2000 | 2 | 0.000000000 | 0.000000000 |
| 1 | 2000 | 2 | 0.000000000 | 0.000000000 |
| 2 | 2000 | 8 | 0.000000000 | 0.000000000 |
| 3 | 2000 | 8 | 0.000000000 | 0.000000000 |
| 4 | 8000 | 2 | 0.000000000 | 0.000000000 |
| 5 | 8000 | 2 | 0.000000000 | 0.000000000 |
| 6 | 8000 | 8 | 0.000000000 | 0.000000000 |
| 7 | 8000 | 8 | 0.000000000 | 0.000000000 |

| group | weighted_accuracy_delta | weighted_nll_delta |
| --- | --- | --- |
| labeled_images=2 | 0.000000000 | 0.000000000 |
| labeled_images=8 | 0.000000000 | 0.000000000 |
| source_step=2000 | 0.000000000 | 0.000000000 |
| source_step=8000 | 0.000000000 | 0.000000000 |

### PROTO minus SHUFFLE

| context | source_step | labeled_images | weighted_accuracy_delta | weighted_nll_delta |
| --- | --- | --- | --- | --- |
| 0 | 2000 | 2 | 0.000275779 | 0.000058424 |
| 1 | 2000 | 2 | 0.000105700 | 0.000133760 |
| 2 | 2000 | 8 | 0.000073543 | -0.000026023 |
| 3 | 2000 | 8 | -0.000021825 | 0.000094342 |
| 4 | 8000 | 2 | 0.000167726 | 0.000411484 |
| 5 | 8000 | 2 | 0.000727242 | -0.000535970 |
| 6 | 8000 | 8 | 0.000069894 | 0.000078707 |
| 7 | 8000 | 8 | 0.000345895 | -0.000245573 |

| group | weighted_accuracy_delta | weighted_nll_delta |
| --- | --- | --- |
| labeled_images=2 | 0.000319112 | 0.000016924 |
| labeled_images=8 | 0.000116877 | -0.000024637 |
| source_step=2000 | 0.000108299 | 0.000065126 |
| source_step=8000 | 0.000327689 | -0.000072838 |

## Candidate attribution and shuffle realization

EMA agreement is measured on the fixed prototype candidate region. Every candidate belongs to exactly one of: EMA agrees with prototype, EMA agrees with original, or EMA proposes the third class. This tests the EMA-restoration explanation directly. Actual changed counts can differ for EMA; identical candidate region does not force an artificial equal dose.

| context | candidate | EMA_agrees_prototype | EMA_agrees_original | EMA_other_class | PROTO_changed | EMA_changed | SHUFFLE_changed | shuffle_pool | active_shuffle_strata | fully_occupied_strata | retained_original_changed_locations | shuffle_locations_different |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 112 | 112 | 0 | 0 | 112 | 112 | 112 | 505 | 2 | 0 | 33 | 158 |
| 1 | 339 | 339 | 0 | 0 | 339 | 339 | 339 | 559 | 2 | 0 | 211 | 256 |
| 2 | 231 | 231 | 0 | 0 | 231 | 231 | 231 | 2757 | 5 | 0 | 22 | 418 |
| 3 | 198 | 198 | 0 | 0 | 198 | 198 | 198 | 823 | 6 | 0 | 74 | 248 |
| 4 | 277 | 277 | 0 | 0 | 277 | 277 | 277 | 1463 | 4 | 0 | 81 | 392 |
| 5 | 479 | 479 | 0 | 0 | 479 | 479 | 479 | 1065 | 3 | 0 | 272 | 414 |
| 6 | 1621 | 1621 | 0 | 0 | 1621 | 1621 | 1621 | 3672 | 7 | 0 | 861 | 1520 |
| 7 | 601 | 601 | 0 | 0 | 601 | 601 | 601 | 1930 | 9 | 0 | 286 | 630 |

Pooled candidate agreement counts: 3858 with prototype, 0 with original, 0 with neither; candidate total 3858.

Weighted target L1 change is recorded below to expose the unmatched continuous-amplitude component. It is not a measured training gradient norm. Matching changed counts/destination classes/EMA weights does not ensure the same original confidence at relocated pixels. Thus this control isolates location relative to a coarse class/weight prior, not every aspect of semantic information or confidence.

| context | BASE_weighted_target_L1_change | PROTO_weighted_target_L1_change | EMA_weighted_target_L1_change | SHUFFLE_weighted_target_L1_change |
| --- | --- | --- | --- | --- |
| 0 | 0.000000000 | 63.048585415 | 63.048585415 | 86.813686132 |
| 1 | 0.000000000 | 295.079868674 | 295.079868674 | 310.925748289 |
| 2 | 0.000000000 | 89.513566375 | 89.513566375 | 105.988170862 |
| 3 | 0.000000000 | 123.308808208 | 123.308808208 | 153.650305867 |
| 4 | 0.000000000 | 141.002915144 | 141.002915144 | 188.162970543 |
| 5 | 0.000000000 | 440.847235978 | 440.847235978 | 460.132878184 |
| 6 | 0.000000000 | 802.752856702 | 802.752856702 | 844.290140212 |
| 7 | 0.000000000 | 489.905171067 | 489.905171067 | 538.257330418 |

## Complete per-context metrics

All32rows, including positive/negative/no-effect arms. RESULTS.json/csv retain the complete additive unweighted/weighted statistics and rates.

| context | source_step | labeled_images | scope | pixels | changed | repaired | harmed | weighted_net_accuracy | weighted_nll_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2000 | 2 | BASE | 146094 | 0 | 0 | 0 | 0.000000000 | 0.000000000 |
| 0 | 2000 | 2 | PROTO | 146094 | 112 | 112 | 0 | 0.001187970 | -0.000703386 |
| 0 | 2000 | 2 | EMA | 146094 | 112 | 112 | 0 | 0.001187970 | -0.000703386 |
| 0 | 2000 | 2 | SHUFFLE | 146094 | 112 | 99 | 13 | 0.000912191 | -0.000761810 |
| 1 | 2000 | 2 | BASE | 146160 | 0 | 0 | 0 | 0.000000000 | 0.000000000 |
| 1 | 2000 | 2 | PROTO | 146160 | 339 | 339 | 0 | 0.003583225 | -0.003399786 |
| 1 | 2000 | 2 | EMA | 146160 | 339 | 339 | 0 | 0.003583225 | -0.003399786 |
| 1 | 2000 | 2 | SHUFFLE | 146160 | 339 | 334 | 5 | 0.003477526 | -0.003533546 |
| 2 | 2000 | 8 | BASE | 583423 | 0 | 0 | 0 | 0.000000000 | 0.000000000 |
| 2 | 2000 | 8 | PROTO | 583423 | 231 | 226 | 5 | 0.000306428 | -0.000258791 |
| 2 | 2000 | 8 | EMA | 583423 | 231 | 226 | 5 | 0.000306428 | -0.000258791 |
| 2 | 2000 | 8 | SHUFFLE | 583423 | 231 | 199 | 32 | 0.000232885 | -0.000232769 |
| 3 | 2000 | 8 | BASE | 583570 | 0 | 0 | 0 | 0.000000000 | 0.000000000 |
| 3 | 2000 | 8 | PROTO | 583570 | 198 | 187 | 11 | 0.000454241 | -0.000330516 |
| 3 | 2000 | 8 | EMA | 583570 | 198 | 187 | 11 | 0.000454241 | -0.000330516 |
| 3 | 2000 | 8 | SHUFFLE | 583570 | 198 | 191 | 7 | 0.000476066 | -0.000424858 |
| 4 | 8000 | 2 | BASE | 146979 | 0 | 0 | 0 | 0.000000000 | 0.000000000 |
| 4 | 8000 | 2 | PROTO | 146979 | 277 | 277 | 0 | 0.002080855 | -0.001578487 |
| 4 | 8000 | 2 | EMA | 146979 | 277 | 277 | 0 | 0.002080855 | -0.001578487 |
| 4 | 8000 | 2 | SHUFFLE | 146979 | 277 | 269 | 8 | 0.001913128 | -0.001989972 |
| 5 | 8000 | 2 | BASE | 146984 | 0 | 0 | 0 | 0.000000000 | 0.000000000 |
| 5 | 8000 | 2 | PROTO | 146984 | 479 | 477 | 2 | 0.004958948 | -0.005035501 |
| 5 | 8000 | 2 | EMA | 146984 | 479 | 477 | 2 | 0.004958948 | -0.005035501 |
| 5 | 8000 | 2 | SHUFFLE | 146984 | 479 | 442 | 37 | 0.004231706 | -0.004499531 |
| 6 | 8000 | 8 | BASE | 587956 | 0 | 0 | 0 | 0.000000000 | 0.000000000 |
| 6 | 8000 | 8 | PROTO | 587956 | 1621 | 1621 | 0 | 0.002317254 | -0.002356940 |
| 6 | 8000 | 8 | EMA | 587956 | 1621 | 1621 | 0 | 0.002317254 | -0.002356940 |
| 6 | 8000 | 8 | SHUFFLE | 587956 | 1621 | 1601 | 20 | 0.002247360 | -0.002435647 |
| 7 | 8000 | 8 | BASE | 588011 | 0 | 0 | 0 | 0.000000000 | 0.000000000 |
| 7 | 8000 | 8 | PROTO | 588011 | 601 | 578 | 23 | 0.001384933 | -0.001332155 |
| 7 | 8000 | 8 | EMA | 588011 | 601 | 578 | 23 | 0.001384933 | -0.001332155 |
| 7 | 8000 | 8 | SHUFFLE | 588011 | 601 | 514 | 87 | 0.001039038 | -0.001086582 |

## Interpretation and next evidence boundary

Every candidate receives exactly the same destination from EMA and prototype; with the same swap rule the two interventions coincide on this region. The measured prototype candidate therefore adds no target information beyond the already available EMA decision here. This does not prove that the prototype-derived candidate-location selector can be removed: locating that region still uses prototype suggestions. An EMA-only location selector has not been evaluated.

The joint incremental-information gate failed. Do not promote this exact candidate union to full RL or claim that the prototype mechanism has demonstrated reliable additional target quality. Any next hypothesis must address the failed comparison explicitly and be frozen separately; no threshold/permutation/condition search or automatic full training follows from this result.

V118/V119 whole-mask failures remain unchanged. The current candidate union is sparse and selected using V120 outcomes; repeated image/context exposures, differing support counts(n=2 one other image,n=8 two) and model-seen labels prevent independent-generalization or causal annotation-count claims. This is not an unobserved-U correctness test. Before unseen-image reliability claims, use separately frozen legal training-only image splits that exclude target images from all fitting, without opening sealed Q_dev/test/hidden U labels. Before training-value claims, use identical entry/stream/budget paired training and a mass-matched control if selected mass/U weight changes.

History-free trained-parameter memory is preserved; prototypes remain transient current-domain auxiliary evidence. No new historical bank, additional loss, optimizer update, GRPO/fullRL training or next experiment was performed during this diagnostic closeout.

## Costs and delivery

New40current-labeled image/context reads/evaluations,160model-image forwards(40each EMA/memory/flipped-memory/student),40prototype assignments. Four target arms reuse these same forwards. V119–V121 combined480segmentation forwards/120labeled evaluations, separately charged. Zero native/actor optimizer updates, solves,Q_train/Q_dev,U-image reads,hidden labels or new annotations. Cumulative training841061native excluding8000source(849061including),181593actor,57solves unchanged; all prior failures and separate512/146CPUactor-forward diagnostics retained. Synthetic selfcheck and scalar completion audit add0model/image/optimization calls.

Public delivery includes source/protocol/selfcheck,all32anonymous metric rows,8control summaries,all3comparisons with fullcontext/subgroup outcomes,support/reference/qualification/cost/execution/audit/report. Private image metrics/support indices/features/data/weights/raw logs stay on NAS. Committed-file NAS delivery plus proxy GitHub push/remote branch/anonymous report checks are recorded in FINAL_PUBLICATION.json.


## Requested posthoc interpretation: class suggestion, location and memory conflict

This section adds only published-scalar arithmetic and inspection of the existing target/control source. No model/image/query/fit/optimization was performed. The original NO_RELIABLE_INCREMENTAL_PROTOTYPE_SIGNAL gate remains unchanged. All past costs/results remain intact.

### What the identical EMA intervention does establish

All3858candidate destinations agree with EMA. Because the same original probabilities are exchanged at the same locations, PROTO and EMA are exactly the same target operation here, not merely statistically similar. No additional class suggestion is established on these candidates. This comparison tests which class to propose CONDITIONAL ON the prototype-derived region. It does not test how to find that region without prototypes. In particular, the pre-registered joint gate tests a stronger proposition than mere selector utility: a method can in principle be useful by selecting when to trust EMA without inventing a different destination. Its gate failure must not be paraphrased as proof that all prototype information is useless.

### Code-derived localization of the conflict

The unchanged action11 source in v98_action_capacity/run.py sets target=q when memory gate g=0, and target=.25*q+.75*qm when g=1. The memory gate requires max(qm)>.7 and agreement with flipped-memory argmax. On any valid g=1 pixel, the memory winning class has target probability>.75*.7=.525, while any other class has probability<.25+.75*.3=.475. Thus the blended target argmax necessarily equals memory argmax. If blended argmax differs from EMA, the gate must be on: when off the target equals EMA exactly.

Combining this code property with the observed3858/3858 EMA-prototype agreement shows that every current candidate is a memory-gate-on EMA-versus-memory disagreement, and the exchange restores EMA's winning CLASS. This is a logical consequence of source plus recorded agreement, not an extra measurement of hidden U labels. The exchange does not restore the entire EMA probability vector: it retains and permutes the blended probability values. Nor does this establish that memory should be removed, the entire gate is faulty, or V118's training loss has a uniquely proven cause. Old-task protection and U training are not measured here; current EMA already trained on these labeled targets.

### Most of this local mean benefit survives the coarse conflict prior

SHUFFLE also improves both target metrics versus BASE in8/8contexts. It preserves approximately89.2837% of PROTO's equal-context weighted accuracy effect and99.7943% of its mean NLL improvement. These are ratios of observed mean effects, not causal variance explained or a performance bound. They suggest that the coarse original-class×EMA-class conflict strata and their allocated intervention budgets account for much of the observed benefit under this diagnostic.

Crucially, the shuffle is NOT prototype-free. For each image/stratum, its changed count K and destination/no-change tokens come from PROTO. Thus both same-region EMA and the matched shuffle inherit information from prototypes. V121 has not evaluated an EMA-only selector or an independently specified intervention allocation; it cannot conclude that simple EMA gates reproduce the result without this information. The reported12774shuffle-pool pixels cover only active strata, and the one sampled3858-pixel result cannot be called accuracy on all12774 or all EMA-memory disagreements.

### There is a hard-label location signal, but unstable soft-target increment

PROTO repairs3817/3858changed exposures(98.9373%), SHUFFLE3649/3858(94.5827%). PROTO's weighted accuracy is higher in7/8contexts. This is a real descriptive advantage and should not be erased by the failed joint gate. However, NLL improves over SHUFFLE in only3/8contexts and worsens in5/8. Its equal-context NLL increment is merely−0.0000038561. As an explicitly posthoc sensitivity description, omitting context5 would reverse that mean to+0.0000721602; no context is actually dropped and no original decision is recalculated. The mean is not evidence of broad probabilistic-supervision improvement.

The current data are consistent with useful classification-focused location ranking that does not yet provide stable soft-target benefits, but are insufficient for a general selector claim. A single frozen random permutation gives one comparison, not an expectation or significance test, and the reused V120-selected candidate cannot count as independent confirmation.

### The matched control still has important limits

The shuffle retains1840/3858original changed locations(47.6931%) and replaces2018. It destroys correspondence partially, not completely. Do not subtract this overlap out of scores or invent disjoint-only metrics absent from saved statistics. Counts, destinations and EMA weight strata are matched, but weighted target L1 change is9.9271% larger in SHUFFLE and larger in all8contexts. Continuous confidence/perturbation amplitudes and exact native gradients are not matched, so the difference is not a clean isolation of prototype semantics alone. This limitation does not invalidate the exact same-region EMA equality.

### Implication for the research question

The most concrete current question is whether deployment-visible evidence can identify when frozen-memory class dominance should yield to the current EMA, without damaging retention. Prototype evidence could act as a corroborating selector, even when the destination class already comes from EMA. This is a narrower hypothesis than a new pseudo-label source; it does not call for changing the history-free architecture, removing parameter memory, or returning to a loss-weight sweep.

Before considering a new run, distinguish the untested contributions: EMA-only location and budget selection; prototype's incremental selection value after that control; reliability on genuinely model-unseen training-only images; and translation to new/old-task training gains. Any future comparison requires a separate frozen protocol and must retain all original results/gates. Do not jump to full GRPO, expand the same eight-condition diagnostics indefinitely, or replace a failed gate with an easier after-the-fact success definition. This analysis launches no further experiment.

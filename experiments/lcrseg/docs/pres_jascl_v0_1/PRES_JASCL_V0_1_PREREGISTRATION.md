# PRES-JASCL V0.1 preregistration

Registration ID: `PRES_JASCL_V0_1_SNAPSHOT_DOMAIN_ROUTER`. This is a no-training, parameter-isolated snapshot-expert routing feasibility baseline. It is not a PMGC repair and does not authorize prototype pseudo-labels, feature transport, relation loss, gradient-cone work, or any method training.

## Frozen lineage and barrier

The branch is `codex/pres-jascl-v0-1-routing-feasibility`, created directly from the unique PMGC HEAD `06945fa738cf87a6cef22db949b627490e7847f1`. PMGC closure commit `c003e13cb14ee1b9c14c6b445ff66c364e6c68b7` is its required ancestor. Until this preregistration and a separate execution authorization are pushed and independently remote-verified, checkpoint tensor reads, private HDF5 reads, model forwards, router prototype construction, and formal-result generation all remain zero.

The frozen input is the complete verified Gate1C regenerated-B0 private bundle. Its published bundle-content SHA, bundle-manifest SHA, all nine stage-best checkpoint SHAs, and all 2962 data checksums must be resolved from the frozen Gate1C reports and verified item by item. PMGC virtual-step outputs are forbidden.

## Fixed expert bank and router

For each seed 0/1/2, expert0, expert1, and expert2 are respectively the B0 stage-best snapshots for REFUGE, RIM_ONE_r3, and Drishti_GS. Their inference identity is exactly the student-or-EMA identity and posterior-mean stochastic-classifier procedure used by the frozen regenerated-B0 final evaluator. PRES cannot reselect an expert on validation. Oracle-snapshot is the fixed mapping REFUGE→expert0, RIM_ONE_r3→expert1, Drishti_GS→expert2; Shared-final uses expert2 everywhere.

For each seed the router extractor is the B0 stage0 REFUGE stage-best EMA encoder, fixed across every domain and stage. It runs in eval/no-grad mode, AMP off, stochastic classifier off, with float32 forwards. Only the normalized RGB input, enc1, and enc2 tensors are exposed. No decoder, logits, mask, PAS, path/name token, domain one-hot, GT, or expert-specific encoder may enter the descriptor.

For each block `F`, compute its spatial channel mean and population standard deviation in float64, concatenate mean/std, and unit-normalize when the block norm is greater than `1e-12`; otherwise use a zero block marked invalid. Concatenate the normalized RGB, enc1, and enc2 blocks and unit-normalize again. A final norm at or below `1e-12` is `BLOCKED_NUMERICAL_FAILURE`. Each case produces exactly one descriptor.

## Frozen prototypes, routing, and bootstrap

Only same-domain `train_unlabeled` descriptors build prototypes. M1 is the unit-normalized equal-case mean. M2 directly reuses the repository's CPU float64 weighted spherical K-means with five restarts and no cluster deletion or domain mixing. Formal weights are one per case; bootstrap weights are resampling multiplicities. The two M2 slots must both be active, finite, unit norm, and have occupancy at least 0.10.

The existing stable seed function is retained. M2 restart `r` uses `clustering_seed(seed, domain_index, domain_index, 2, replicate, r)`, with `replicate=-1` for the formal fit. Each domain is fit once under its own domain index, so adding a domain must leave every older prototype and its metadata byte-identical. Domain score is the maximum prototype cosine; routing is argmax over seen domains with exact ties assigned to the lowest domain index. Stage1 exposes only domains 0/1 and Stage2 exposes 0/1/2.

There are five case bootstrap replicates 0–4. Sorted case IDs are sampled uniformly with replacement by NumPy PCG64 using `S(["pres-bootstrap-v1", seed, stage, role, domain, replicate])`; the draws are shared by M1 and M2. Train-unlabeled draws rebuild prototypes, validation draws evaluate routing, M1 stability is cosine to the formal prototype, and M2 stability uses Hungarian slot matching.

## Validation and metrics

All router descriptors are extracted and sealed before domain metadata reaches the routing evaluator. Validation GT is accepted only by an independent segmentation evaluator; router APIs cannot accept it. No test object is constructed. The complete execution comprises Stage1 and Stage2 routing for M1/M2, both confusion matrices, all seed/domain/expert cells of the 3×3 Stage2 segmentation matrix, Shared-final, fixed Oracle-snapshot, Prototype-routed, and all five bootstraps.

Routing reports accuracy, equal-domain macro and per-domain accuracy, integer confusion, true-domain cosine-score margin with p05/p10/median, unit-temperature softmax entropy in nats, prototype separation, bootstrap prototype stability, and bootstrap routing accuracy. Segmentation reuses the exact regenerated-B0 evaluator arithmetic and reports Mean Dice, Foreground Dice, Mean IoU, per-class/per-domain Dice, three-domain and historical averages, current-domain performance, forgetting, oracle-routed gap, and routed-shared gain. Gates use finite unrounded validation values. Each seed/domain is reduced by case mean, then seeds and domains are averaged equally.

## Frozen gates

- D1: Oracle minus Shared Stage2 foreground Dice is at least 0.015 over three domains and 0.020 over REFUGE/RIM; at least two of three seeds have positive three-domain gain; no seed/domain Oracle drop exceeds 0.005.
- D2, for each complete M: Stage1 macro accuracy is at least 0.95 and every domain at least 0.90; Stage2 macro is at least 0.90 and every domain at least 0.85.
- D3, for each complete M: Stage2 routed foreground Dice is within 0.010 of Oracle, gains at least 0.010 over Shared and 0.015 on historical domains, is positive in at least two seeds, and no seed/domain drop versus Shared exceeds 0.010.
- D4: M1 bootstrap prototype cosine median is at least 0.95. M2 occupancy is at least 0.10 for both slots and matched cosine median at least 0.90. For the selected candidate, the linear five-replicate routing-macro p10 is no more than 0.05 below formal macro, and every prototype/score/margin value is finite.
- D5: all model/checkpoint states are bitwise unchanged; optimizer, autograd, backward, and parameter-grad-write counts are zero; no test/hidden GT is used; validation GT is evaluator-only; the router has no trainable parameter; all required artifact hashes are complete.

`passing_M` contains complete candidates satisfying D2–D4, and `selected_M` is the smallest. D1 and D5 must also pass for `PASS_PRES_ROUTING_FEASIBILITY`. M2 cannot rescue incomplete M1 evidence. All gates are reported even when one fails; blockers take precedence over scientific failures.

## Call graph, isolation, publication, and stop

After exact source is committed, pushed, and remote-verified—but before any real forward—the verified manifest must compile and freeze `PRES_JASCL_CALL_GRAPH.json` with exact router forwards, segmentation forwards, bootstrap operations, model guards, and output rows. Any execution mismatch is `BLOCKED_CALL_GRAPH_MISMATCH`.

All output is create-only on a new directory under `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg`, launched through `experiments/lcrseg/scripts/with_nas_storage.sh`; there is no home fallback. The required 46-category PRES suite plus relevant Gate0 evaluator/checkpoint regressions must finish with zero failures, errors, or skips. Model/router optimizer steps and autograd calls remain zero, `method_registered=false`, and `training_launched=false`.

After the validation report and required manifests/CSVs/audits are committed, pushed, and matched to the remote branch SHA, execution hard-stops for independent review. Test evaluation, C0 regeneration, retraining, LoRA/adapters, MILE work, Prostate, MnMS, Gate2/full sweeps, and main merge remain unauthorized.

# PRES-DSR-SF V0.2 preregistration

Registration: `PRES_DSR_SF_V0_2_DESCRIPTOR_MEMORY_SOFT_ROUTER`. This is one new prospective, validation-only protocol. It is not PRES V0.1 attempt 2 and does not adjudicate or repair the frozen V0.1 run.

## Lineage and release barrier

The branch `codex/pres-dsr-sf-v0-2-feasibility` descends directly from the unique verified PRES V0.1 HEAD `ab71694ad6b3134fe1b45bd479658349e619fdc5`. V0.1 closure commit `607a067319a6e8f0bfc1b8d6a305f014cd6ab676` is required. Until this preregistration and a separate execution authorization are committed, pushed, and remote-verified, checkpoint tensors, private HDF5, model forwards, descriptors, router fits, and formal results remain unread or zero.

## Frozen method

Each seed uses the fixed regenerated-B0 student snapshots for expert0/1/2 at REFUGE/RIM_ONE_r3/Drishti_GS stages. Oracle uses that mapping and never validation-best selection; Shared-final uses expert2 everywhere. Expert probabilities use the frozen evaluator's deterministic student posterior mean.

The router extractor is each seed's stage0 REFUGE best EMA encoder, fixed for all stages. From normalized RGB, enc1, and enc2 it computes spatial channel means and `log(population_std + 1e-6)`, concatenated to 102 dimensions. Blocks are not L2-normalized. Seen-domain train memory combines train_labeled and train_unlabeled images without reading segmentation labels, stores at most 512 rows per domain by a frozen salted case-hash order, and contains only float64 descriptors, domain indices, and case hashes. Global per-dimension standardization is fit only on seen-domain train memory; constant dimensions map to zero.

At each seed/stage, a domain-balanced CPU-float64 ridge classifier with an unregularized bias is fit in closed form. Five deterministic domain-stratified case-hash folds choose lambda from `{1e-4,1e-3,1e-2,1e-1,1}` by macro accuracy, NLL, then largest lambda. Pooled selected-lambda OOF logits choose temperature from `{0.5,1,2,4}` by NLL, distance from one, then smaller T. Validation never selects lambda or temperature.

Hard routing is deterministic argmax with lowest-index tie handling. The primary C6 output is probability fusion, `p_mix = sum_d alpha_d p_expert_d`; logits are never averaged. Stage1 exposes experts 0/1 and Stage2 exposes 0/1/2. C0 Shared, C1 Oracle, clean C2 M1-hard, clean C3 M2-hard, C4 M1-soft, C5 ridge-hard, and descriptive C7 uniform cannot rescue C6.

## Backend and evidence isolation

The worker captures the initial backend, imports the pinned JASCL classifier without constructing a model or reading a checkpoint, immediately applies and freezes the registered deterministic state, and only then constructs/loads models. From that point, any change to deterministic algorithms, cuDNN deterministic/benchmark, TF32, or autocast state is `BLOCKED_BACKEND_STATE_MUTATION`. The state is verified after every phase.

All descriptors and router parameters are sealed before domain IDs reach routing evaluation. Validation masks enter only after descriptor, router, lambda/temperature, expert-probability, and candidate-mixture seals. Router fitting cannot accept segmentation GT. Five fixed train-case bootstraps refit ridge and temperature using cached validation descriptors/probabilities and perform no additional model forward.

Clean M1/M2 controls are recomputed from new V0.2 forwards and must match the frozen public V0.1 diagnostic metrics within maximum absolute difference `1e-6`; V0.1 metric rows are not reused as evidence.

## Gates and stop

E1 requires backend/control reproduction, immutability, and finiteness. E2 freezes the snapshot-oracle value thresholds. E3 requires ridge-hard Stage1 macro/domain accuracy at least 0.95/0.90 and Stage2 at least 0.90/0.85. E4 requires primary Stage2 soft fusion to be within 0.020 of Oracle, gain at least 0.130 over Shared, 0.200 historically, and 0.010 over clean M1-hard, with positive gains in all seeds and bounded seed/domain/current-domain drops. E5 freezes bootstrap hard-routing and soft-fusion quantiles. E6 requires zero model optimization/autograd/backward/grad writes, strict GT isolation, memory cap, and hash-complete evidence.

Only all E1-E6 pass yields `PASS_PRES_DSR_SF_FEASIBILITY`; blockers take precedence, then the registered scientific failure states. The single formal attempt must complete all registered controls. After report/archive publication and remote verification, execution hard-stops for independent review. No test evaluation, regeneration, training, fine-tuning, adapter, other benchmark, sweep, Gate2, or main merge is authorized.

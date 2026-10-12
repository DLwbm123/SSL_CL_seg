# V19 CaRE hard-gated reference adaptation

Authority: the explicit independent GUA→SARD→REDistill→CaRE-KD queue and autonomous method follow-up. V17 SARD is fully delivered at dbe08e658430db00b3ba6608ff4fdd5f070d61f1, with both original gates failing. V18 REDistill's fixed mathematical screen is fully delivered at e6c95f3; it fails native advancement and is not trained. V19 uses common V13, with no SARD, GUA or class-cone stacking.

## Source and one selected component

Primary source: [Distilling What Matters: Confidence-Aware Selective Distillation for Large Language Models, v1](https://arxiv.org/html/2609.36734v1), Eq1–3, §3.1 and §8.3. Its hard gate compares normalized entropy confidence c=1-H/log(K), uses forward KL when the teacher is at least as confident as the student, and reverse KL otherwise. The hard gate has no soft-gate selection derivative. The paper also proposes BALD-based Revival; that component is not selected. Its log-probability floor is mentioned without a numerical value in §8.3.

Our explicit segmentation adaptation: teacher p is the frozen own-entry distribution, student q is the current U student's distribution, T=1, K=3. The same original PAS-valid pixels with entry max>=0.7 form support. Per pixel, detached g=1[H(p)<=H(q)]; ties select forward. Loss is g KL(p||q)+(1-g)KL(q||p), with each present entry-argmax class averaged separately, then equal mean over present classes. Empty support yields graph-connected zero. No reference term is added to the training objective; only its one VJP supplies the original V13 aggregate raw-gradient halfspace.

Numerical convention frozen for this adaptation: teacher and student log-probabilities come from log_softmax; reverse KL uses teacher logp floored at log(torch.finfo(dtype).tiny). Teacher probabilities, entropy gate and support are detached. Forward KL uses original p and logp; student q/logq remain differentiable. Count teacher-floor values in supported pixels. Finite input and result validation remains. This machine-range convention is not claimed to be an original-paper constant or tuned threshold.

## Hypothesis and limits

SARD improved aggregate Old and seed163/O2 cup but did not improve the seed164/O1 RIM rim regression or order2 Incoming gate. V15's class cone improved the former but damaged seed163/O2 cup. REDistill's literal fixed divergence amplified wrong-teacher gradients in generated cases. These observations justify independently testing confidence-dependent reference geometry; they do not prove a confidence or boundary cause.

This is the hard CaRE-Divergence reference component, not a full LLM CaRE-KD reproduction. No Revival, MC dropout, quantile history, soft gate, extra teacher, loss coefficient, data permission or inference parameter is introduced. Confidence is not correctness; a confidently wrong teacher remains harmful. Reverse KL does not guarantee recovery. The raw-gradient guard does not constrain actual Adam displacement, finite-step loss or every class. Current U is not historical-domain coverage.

## Frozen route, qualification and budget

Retain V13 linear R; joint main KL→A/B/R; clean SWD→R-only; own-entry/current-EMA conflict rejection; lambdaU0.25, SWD0.2, PAS0.7/0.5, Qk8/D16, optimizer/lr/calendar/prototypes, hard input projection, source and frozen data. Two teachers only; source+identity then own sealed student/prefix. Immutable entry is shared with EMA and discarded at deployment. Actual flag, formula, source digest and entry fingerprint bind checkpoint semantics; branch and reference are recomputed from restored cursor.

Create-only NAS root, exact code commit/config/PLAN/auth before execution. Qualification total caps96 CPU optimizer/32 generated CUDA/4 current-L, no budget reset. First attempt expected32/10/2, including one deliberately failed after-optimizer call. CPU: analytical forward/reverse gradients with unequal class supports, equality/empty/extreme/floor/invalid/detach checks,32-step two-case fit. Generated native: disabled archived V13 complete model/EMA/optimizer/scheduler/cursor/prototype and old loss/update parity; unforced reverse activation and nonzero reference-gradient difference versus V13; A/B/R gradients, actual raw projection, frozen/shared teachers, main rejection, SWD R-only, linear EMA, true checkpoint continuation and changed-flag rejection, deployment/own-prefix next stage. Current-L smoke reads zero U. All must pass before formal; fixed native absence of reverse activation does not authorize more probes or forced formal. No partial patient efficacy selection.

Formal at most once: seeds163/164×orders1/2, four trajectories/eight target stages/21200 optimizer/source0. First163/O1 two seals/EXIT0 then remaining three for engineering, never performance pruning. All8SEALED/4EXIT0 before the original20 final seen-domain evaluations. No formal automatic retry, parameter/PAS/seed/data search or checkpoint selection.

Formal VJP budget16960 entry-reference plus64 original R diagnostics; anchor full forwards16960 and class reductions<=50880, unchanged from V13. No extra formal full model forward/VJP/optimizer. Qualification extras are separately reported:5 enabled reference VJPs,4 extra generated gradient VJPs,2 analytic/empty CPU VJPs,1 generated loss call and2 anchor forwards; ordinary updates retain detailed telemetry. Record branch counts/reverse calls, floor counts, runtime and peakmemory. Activation is not scientific success.

Formal actual before V19=403473, V19 cumulative cap424673. Actual qualification before V19=960CPU/237generatedCUDA/48currentL; planned first attempt would become992/247/50. The broader four-paper maximum467073 is an allowance, not spent cost; GUA/REDistill unused formal budgets cannot fund retries. Keep all historical failures, deliberate failures, V5 rollback duplicates and missing diagnostics.

## Readout, delivery and monitoring

Report all paired cells versus V13, V15, V17 and V10, matched F5 and historical B0/B2. Original gates against both F5/B2: per-seed order-mean Final gain>0, pooled Final>=0.005, each order mean Old/Incoming>=−0.005, each seed/order/domain/rim-cup>=−0.05. Keep all negative findings. Repeated development patients do not establish independent generalization, SOTA or clinical benefit.

Use NAS wrapper and neutral parent/child argv; NFS write/read/capacity and GPU admission before start, no system or unrelated-process edits. One startup check, then existing hourly f5 heartbeat, quiet without meaningful change; RL stays paused. Publish complete source/frozen protocol/anonymous results/diagnostics/costs via proxy, verify remote SHA and anonymous access before any subsequent scientific round. Stop the heartbeat only after both original gates and verified delivery, or explicit user stop.

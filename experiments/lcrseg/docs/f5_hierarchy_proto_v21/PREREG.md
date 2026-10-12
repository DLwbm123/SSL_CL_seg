# V21 whole–part hierarchy prototypes on V13

User authorization (2026-10-12): “那就按通用的整体—部件层级 prototype 来做，基于 V13 先在现有眼底任务上设计最小改动方案，再做资格检查和实验，保留原 F5/B2 双门槛，不要写死眼科逻辑，方便后面扩展到其他医学分割任务。” This resumes this research task after the user's pause. V20/B1 are already complete and publicly delivered at806f7719a732831ce7755dd214ab8b70119fc49e; neither is rerun. RL stays paused.

## Evidence and one change

V13 uses flat class-mean current-stage prototypes for PAS. Its class-balanced entry reference has no whole–part hierarchy. V20 did not satisfy the B2 gate and worsened164/O2 old cup retention. Those results motivate studying representation, but do not prove that missing hierarchy caused the damage.

Retain V13 and add one current-labeled-data hierarchy regularizer. Keep the native three-class output, parent/linear sidecar, joint main KL, clean SWD→R-only, PAS thresholds, one class-balanced entry-reference VJP and its original aggregate raw halfspace. No V20 block guard, SARD, CaRE, GUA or class-cone combination. No new head, projection layer, teacher, prototype optimizer/bank, historical image/label access, source training or deployment parameter.

The generic core accepts feature tensors B×D×spatial, integer leaf labels and an explicit integer leaf-to-parent map. This task supplies [0,1,1]: background belongs to parent0, rim/cup belong to parent1 (disc). Names and geometry do not enter the kernel. A four-leaf alternate mapping and3D tensor-grid check test this contract. The existing native backbone, data and training engine remain2D; this is not a completed3D or other-organ implementation.

## Exact equations and gradient contract

Reuse existing post-transform current-L features h from one warmup view or the two complementary LCTX views. Reassemble the two using the same LCTX mask, resized nearest-neighbor to the feature grid. Map current-L labels nearest-neighbor; exclude the3×3 feature-grid band spanning the LCTX seam. ignore255 is excluded from every prototype and loss. Do not run an additional feature/model forward.

Normalize each valid feature z=h/max(||h||,1e−12). For each supported leaf or parent, average its features separately within each present image, then average those image means equally and normalize. This retains equal-image voting rather than letting a large object/image dominate. Zero-norm or absent prototypes are unsupported.

1. Coarse contrast: score a pixel against supported parent prototypes by cosine/0.1. Use cross-entropy to its configured parent, reduced equally over present supported parent classes. Prototype targets are detached. Fewer than two supported parents produce differentiable zero.
2. Part contrast: within each parent with at least two supported children, classify only that parent's pixels against its supported leaf prototypes by cosine/0.1. Reduce equally over present children, then equally over eligible parents. Prototype targets are detached. Singleton/absent decisions are omitted.
3. Parent–child alignment: for each eligible supported parent/child, penalize max(0,||p_child−stopgrad(p_parent)||−0.25), averaged over supported children and eligible parents. The child mean is a live function of current-L features; the parent is a detached target. These are transient tensors, not optimizer parameters or a stored anchor bank.

L_L = original V13 L_L +0.1×(L_coarse+L_parts+L_alignment). The coefficient0.1, cosine temperature0.1 and normalized-distance margin0.25 are fixed before qualification; no scan or retuning. The0.25 margin follows HiPoSeg's parent–child proximity component; other choices are this bounded adaptation. The regularizer is active from the first current-L update, including the original U warmup; there is no new schedule. The original supervised loss continues to discriminate leaves; hierarchy does not replace segmentation supervision or enforce geometric containment.

All added gradients enter the original current-L A/B/R path and existing merged-gradient guard. U losses, teachers, PAS and reference support stay unchanged. Caches are transient, cleared on every return/failure and never affect EMA, checkpoint state or deployment. Actual enable flag, mapping, weight, temperature, margin and source bind the checkpoint semantic contract.

## Source and replication boundary

HiPoSeg, ICLR2026: [official paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/083d4b31542f68b619d842a4cbb68021-Paper-Conference.pdf). Method Section3.1 Eq2–6 defines the hierarchy and prototypes; Section3.2 Eq7–14 defines contrasts and margin alignment. Original Eq15 uses momentum prototype banks and a staged schedule. Its Table5 also reports that simply combining high/low contrasts without alignment can worsen results; we retain a parent–child alignment component rather than treating extra losses as automatically complementary.

This is a selected component adaptation: no original projection head, six-loss objective, persistent hierarchical banks or new curriculum. Fourier Contour Learning is excluded because its full text was unavailable and the user permitted skipping it. No claims of full HiPoSeg replication, novelty, topology guarantee or clinical benefit.

## Finite qualification and advancement

Create-only NAS root and immutable source/PLAN/USER_AUTHORIZATION/config before optimization. Qualification maximum96CPU/32generatedCUDA/4currentL updates; intended first attempt32/10/2, including deliberate after-optimizer failure. Retain every failed/preparation call. No formal retry or fixture/seed search.

CPU checks: generic mapping and leaf renaming, four-leaf alternate task,2D/3D grids, positive feature scaling, whole-batch replication, absent/singleton/all-ignore/zero norms, zero ignored gradients, detached prototype-target finite difference, ten invalid input rejections, and one32-update two-case joint fit. Three extra CPU gradient VJPs outside the fit are counted.

Native checks: disabled complete model/EMA/Adam/scheduler/cursor/PAS prototypes and loss/update match archived V13; current-L feature capture aligns source votes and has no stale cache; original U A/B/R gradients, main KL rejection, nonzero entry reference and actual aggregate projection, clean SWD R-only, shared frozen entry/EMA, linear EMA, deployment/own-prefix transition, exact checkpoint continuation and refusal of changed actual hierarchy settings, deliberate failure accounting, two current-L zero-U smoke updates.

Exactly two already scheduled unforced generated native active updates (steps2 and3) must have supported coarse and sibling decisions. One fixed extra generated loss/gradient check must establish nonzero added current-L gradients in A/B/R; record all three component losses and alignment activity without demanding a fabricated positive hinge. Qualification adds six generated gradient VJPs, five enabled reference VJPs, one generated loss evaluation and two anchor full forwards outside the optimizer-update loops. No hidden/current-U patient probe or extra model training for advancement. If engineering passes but the stated activation requirement fails, preserve a qualification-only negative and do not extend probes or force formal training.

## Formal experiment and original gates

If qualified and admitted:163/164×orders1/2, four trajectories/eight stages,21200 formal updates/source0. First163/O1 engineering seals/EXIT0, then the other three; no performance pruning. All eight sealed/four EXIT0 before the original20 uniform seen-domain evaluations. Inherit each trajectory's own sealed prefix. No checkpoint selection, changed data/seed/schedule/coefficients or automatic retry.

Compare complete paired results to V13, V15/V17/V19/V20/V10 context, matched F5 and historical B0/B2. The gate independently against F5 and B2 remains: each seed's order-mean Final delta>0; overall mean Final delta≥0.005; each order's mean Old and Incoming delta≥−0.005; every seed/order/domain/rim-or-cup delta≥−0.05. Publish all negative cells. Reused development patients do not establish independent generalization or cross-organ efficacy.

Existing active-U costs stay16960 entry-reference VJPs/16960 anchor full forwards/50880 present-class reductions/64 fixed R-diagnostic VJPs. The hierarchy adds21200 current-L evaluations, five prototype estimates (three leaves/two parents here) and three terms per current-L forward pair; no additional formal optimizer/model forward/VJP calls. Report actual captured feature views, support, positive term activity, runtime, peak memory and all operational counts.

Before V21: method formal actual445873 plus completed supplementary B1 5300 =451173. V21 adds a separately authorized21200 cap: method ceiling467073 and combined ceiling472373, not reuse of cancelled B1 or literature allowances. Qualification prior cumulative1056CPU/267generatedCUDA/54currentL; intended first attempt1088/277/56. Preserve historical673 rollback calls, five missing diagnostics and every previous negative result.

All large files/caches/logs/weights stay NAS; use the storage wrapper, NFS real probe and GPU admission. Parent/worker command lines are neutral; no unrelated jobs are stopped. Publish source, frozen protocol, anonymous results/costs/audits/exits/report via the required GitHub proxy and verify delivery. Private configuration, data, patient rows and weights remain private.

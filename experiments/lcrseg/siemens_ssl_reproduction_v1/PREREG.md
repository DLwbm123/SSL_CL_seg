# FixMatch and CPS: Siemens method reproduction adaptations

The user approved two new runs on physical GPU2, seed163, 43,600 iterations per
method and monitoring every30 minutes. This is separate from the completed
130,800-call source debugging study. FixMatch uses one optimizer:43,600 calls.
CPS uses two optimizers:87,200 calls. New allocation totals130,800 physical calls;
no extra real engineering updates, seeds, sweep or blind replay are allocated.
Initially scheduled sequentially. On 2026-10-17, following the user request to improve GPU utilization, scheduling was amended to run both already-authorized methods concurrently on the same GPU2. FixMatch continued without interruption; only the waiting controller was retired before launching CPS. Scientific settings and per-method budgets are unchanged. Preserve negative results and every failure.

## Author sources audited before implementation

- FixMatch: https://github.com/google-research/fixmatch/tree/d4985a158065947dba803e626ee9a6721709c570
  `fixmatch.py:85-137`: stop-gradient weak prediction, threshold>=0.95, masked CE
  averaged over all U examples, weight1, U:L ratio7, SGD Nesterov momentum0.9,
  kernel decay5e-4, LR0.03 cosine(7*pi*t/(16*T)), EMA0.999 for evaluation.
  https://arxiv.org/abs/2001.07685
- Plain CPS (without CutMix): https://github.com/charlesCXK/TorchSemiSeg/tree/5e0c29c02738728b6848a732817db15620a6e034
  `exp.voc/voc8.res50v3+.CPS/train.py` and `config.py`: two independently initialized
  networks, reciprocal detached argmax CE on BOTH L and U, two labeled CE terms,
  CPS weight1.5, SGD momentum0.9 (not Nesterov), LR0.005 poly power0.9, decay1e-4.
  https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Semi-Supervised_Semantic_Segmentation_With_Cross_Pseudo_Supervision_CVPR_2021_paper.html

## Frozen adaptations and limits of reproduction

Use the existing four-class single-channel native U-Net backbone on the frozen
Siemens split:10 labeled patients/230 slices,872 unlabeled slices,23 validation
patients/46 ED/ES volumes. The same role-bound accessors prohibit U labels/test.
Stored384x384 full frames, normalization and uniform patient->phase->slice sampling
are retained. No GT ROI, pretrained model download, hierarchy, LCTX, PAS, style
mixing, entry anchor, projection, foreground reweighting or supervised Dice.
Supervision is plain CE as in author implementations. This is not our prior U loss
attached to the full method.

Both methods use L batch2. FixMatch U batch14 (ratio7); CPS U batch2. Weak geometry
is the existing flips and 90-degree rotations, shared between each FixMatch pair.
FixMatch strong views use two randomly selected grayscale photometric operations
(identity,autocontrast,equalize,brightness,contrast,sharpness,posterize,solarize),
random magnitude in[0,1], plus half-width Cutout. Independent geometric RandAugment
operations are omitted to preserve pixel-label alignment. This is an explicit
grayscale dense-prediction adaptation, not byte-identical classification augmentation.
The backbone has no BatchNorm; separate weak no-grad forwards therefore do not
change batch statistics. EMA supplies evaluation only, not FixMatch pseudo labels.
CPS uses the same weak L/U images in both branches and no confidence threshold.

Single experiment seed163; CPS branch2 uses a deterministic child initialization
stream derived from(seed163,CPS,branch2), not a second experiment seed. Network1
is the predeclared CPS primary; network2 validation is secondary, never selected
by score and no ensemble is substituted. FixMatch primary is EMA. No SSL warmup
or coefficient ramp. Both use FP32 with TF32 disabled and fixed final evaluation.
Weight decay applies to multidimensional trainable tensors, not biases/norm vectors.

The original papers used different datasets/backbones (classification WRN and
pretrained DeepLab/ResNet), batch sizes, augmentation and training lengths. These
runs reproduce the algorithms under the requested Siemens protocol; they do not
reproduce original paper tables. The existing72.2917% L-only/65.8176% SSL results
are contextual references: their labeled objective/optimizer differ, so beating
them would not by itself isolate U's causal contribution. No third supervised run
is implicitly authorized.

## Execution and delivery

Synthetic CPU checks and native finite-backward/checkpoint-load preflight use zero
optimizer calls. Save initial/latest/final states, all optimizer states, EMA and RNG.
Count each actual optimizer call with a post-hook and enforce the physical cap
before each call. A partial CPS iteration remains charged. Failure stops that arm;
the other independently authorized arm may still run. Do not automatically replay.

Evaluate fixed-final train-L and validation once per primary; CPS branch2 gets one
secondary validation. Use the shared volume Dice, ED/ES then patient averaging.
No intermediate validation, test access or best-checkpoint selection. Publish
source, adaptations, aggregate metrics, paired comparisons and a Chinese report
through the required proxy, verifying anonymous access; keep patient rows, raw
logs, settings and weights private on the data volume.

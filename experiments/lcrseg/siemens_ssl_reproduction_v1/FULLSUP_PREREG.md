# Matched fully supervised Siemens baseline, 2026-10-18

The user explicitly authorized exactly one seed163 run with 43,600 optimizer
updates and fixed Siemens splits. This authorization opens labels only for the
existing 38 training-U patients (76 ED/ES volumes) in this isolated run. Frozen
manifests, HDF5, historical outputs and other SSL accessors remain unchanged.
Validation remains 23 patients; independent test and other vendors are excluded.

Reuse the existing FixMatch engine and native U-Net. Match initial weights to its
saved step-zero state. Keep original L2/U14 patient/phase/slice streams, paired
geometry, grayscale strong augmentation including Cutout, CE group means weighted
1+1, SGD Nesterov momentum .9, decay 5e-4 (same parameter groups), initial LR .03
and cosine(7*pi*t/(16*43600)), EMA .999, FP32 and TF32 disabled. On former U,
replace pseudo labels with true labels and remove confidence rejection; no weak
prediction forward is necessary. Cutout retains the underlying semantic labels.
No Dice loss, head/ROI change, pretrained model, warmup, tuning, extra seeds,
intermediate validation or best-checkpoint selection. Same sampling and update
budget does not mean equal FLOPs. This is a matched full-supervision reference,
not uniformly mixed 48-patient training and not an absolute performance upper bound.

Training-label overlay is create-only, containing exactly the authorized 76
labels and their provenance from the existing preprocessing manifest. Bind case,
patient, phase, image record and preprocessing record against the frozen training
manifest. Reuse internal file-integrity checks. Keep images unchanged. Native
preflight verifies finite backward and checkpoint restoration with zero optimizer
updates; synthetic tests verify objective, access bounds and existing algorithms.
No separate overfit training or extra engineering updates are authorized.

At fixed final EMA, evaluate original 10-patient training group, former 38-patient
training group, and validation separately. Reuse existing FixMatch patient rows,
verify their aggregate is 80.3674% (rounded), and compute paired full-minus-FixMatch
mean/median, improvement counts, LV/MYO/RV and ED/ES summaries. One selected seed
supports descriptive validation conclusions only.

Run detached on the established data volume through the existing storage wrapper,
with neutral visible entry points. Save initial/latest/final states, update ledger,
status, exit receipts and a Chinese report. Fail closed; no automatic replay or
budget extension. Do not automatically create monitoring. On a subsequent status
check finding completion, validate results and publish source, aggregate results
and report to the project GitHub through its required proxy. Keep weights, patient
rows, raw data, machine paths and raw logs private.

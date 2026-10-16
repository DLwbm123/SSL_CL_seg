# Independent Siemens training, fixed protocol v1

Authorized on 2026-10-16: complete independent training; stop the existing M&Ms
V2 experiment and use GPU0. This is a separate supervised source baseline, not
a restart or follow-up round of V2. Two seeds (163,164) run serially on GPU0.
No extra vendor, seed, ablation, test, retry or automatic parameter search.

## Evidence and resulting choice

- Campello et al., TMI 2021, DOI 10.1109/TMI.2021.3090082,
  [M&Ms challenge report](https://www.ub.edu/mnms/): participants generally used
  basic preprocessing and data augmentation. The challenge's 150 labeled
  training cases and vendor split are not our 10-patient source protocol.
- Full et al., [challenge winner study](https://arxiv.org/abs/2011.07592),
  Table 3: no-augmentation and basic spatial augmentation substantially differ
  even within vendor. This motivates retracting the earlier no-augmentation
  proposal, not copying its larger nnU-Net ensemble or comparing its score as
  an acceptance target. Fixed lightweight flips, rotations and gamma are
  implementation choices, not a claim of exact winner reproduction.
- DC2T, TMI 2025, DOI 10.1109/TMI.2024.3469528: Adam 5e-4, batch6,
  200 epochs, CE+Dice, heart-region crop and 384-square input. Its IndividualTrain
  unlabeled objective and complete architecture are not specified. This is a
  supervised independent baseline with those principal hyperparameters, not
  a claimed exact reproduction of its 87.10% Siemens result.
- [CineMA preprocessing](https://github.com/mathpluscode/CineMA/blob/main/cinema/data/mnms/preprocess.py)
  obtains its 192mm crop centre from ED LV labels. That per-validation-patient
  ground-truth localization is deliberately not used here.

## Frozen geometry and data boundary

Use only current split0 canonical320 Siemens training-L (10 patients,20 ED/ES
volumes,230 slices). No hidden U labels, U image training, independent test,
extra cohort, external pretrained model or learned locator. Validation is the
same 23 patients/46 volumes used by existing source comparisons.

The 10 training patients' ED/ES-union LV/MYO/RV bounding-box centres have median
(y,x)=(214.25,203.25) in the existing 384-square,320mm canvas. The maximum required
half-width about that centre is99.375mm. Add20mm on each side and round the side
up to16mm:240mm. Round crop origin once to(y,x)=(70,59), take288x288 pixels and
resize to384x384. Actual centre=(214,203), a0.25pixel discrepancy per axis from
the ideal centre. This is a training-calibrated fixed spatial prior, not a
patient-specific detector. Train and val use exactly the same crop without
consulting each patient's labels. No Z removal. Existing normalization stays.

At evaluation, resize softmax probabilities back to288x288, take argmax and
paste into the complete original384x384 canvas with background outside the ROI.
Compute volume Dice before ED/ES averaging per patient, then patient/class
means. Never evaluate only cropped truth. Validation ROI retention is reported
descriptively and cannot change the crop or trigger a retuned run.

## Training and accounting

The executable frozen settings are in protocol.py. Shared UNet2D retains
16/32/64/128 channels and GroupNorm; remove the unused learned projection head
and use its1x1 four-class segmentation head. Reuse SupervisedMethod and Trainer.
The shared loss is CE plus batch foreground squared-denominator Soft Dice,
equal weights; exact Dice formula is an explicit implementation convention.

Adam defaults beta=(.9,.999),eps1e-8,weight_decay0; constant learning rate5e-4,
no scheduler decay/warmup, FP32/noTF32/noAMP. Per-image augmentation: each-axis
flip p=.5; rotation p=.2 angle uniform[-30,30]degrees; gamma p=.3 uniform[.7,1.5]
on the [0,1] mapped intensity, then return to[-1,1]. Image interpolation bilinear,
label nearest. No CutMix, hierarchy, EMA, pseudo labels, projection/protection,
reconstruction, TTA, ensemble or connected-component postprocessing.

Shuffle all230 slices once per epoch with no replacement, retain the final
batch of2. No foreground oversampling. 39updates/epoch *200=7800updates/seed;
formal total15600; 46000 labeled slice exposures/seed. Stop at fixed epoch200,
not a performance threshold. Validate every10epochs,20passes/seed. Final is
primary; best-validation checkpoint is secondary descriptive, never used to
change training. Save separate physical ledgers, component loss curves,
class/domain metrics, image exposures, failures, time and peak memory.

Engineering is separate. Five CPU contract/replay tests passed. The first
GPU two-patient check used64updates but failed an assistant-imposed loss-halving
criterion; initial/final loss was not persisted in that original failure.
Preserve it as a failed engineering attempt, not a scientific result. The user
then authorized one additional at-most130-update diagnostic (cumulative194),
and explicitly rejected useless admission gates. Remove the arbitrary
short-fit loss threshold. Use only3additional GPUupdates (one step + two
checkpoint-replay steps,30image-forward exposures) to validate finite gradients
and restore correctness; do not consume the unused128-step fitting budget.
Expected total real engineering consumption67, below the194authorized ceiling.
No validation in qualification. No convergence/Dice threshold prevents the
fixed200-epoch formal run; final effect is assessed from actual validation.

Existing V2 is user-interrupted, not a completed six-target scientific study.
Its incomplete results remain NA, and its optimizer use is separate. All
checkpoints/raw/private metrics stay on approved storage. Publish anonymous
source/results/report after actual terminal audit; background startup is not
scientific completion. No new recurring monitor is authorized by this request.

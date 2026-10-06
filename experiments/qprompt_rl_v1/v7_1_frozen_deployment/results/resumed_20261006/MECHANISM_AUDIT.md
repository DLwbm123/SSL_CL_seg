# Mechanism audit

Zero student/controller updates and no virtual steps. Features and all restored native snapshot fields were checked exactly. Physical operation counts are outside restored model state.

## Policy distribution

The policy is set-conditioned; each greedy-conditioned selection position records legal count, entropy, uniform entropy, KL, finite-logit SD, maximum probability and top-two gap. Eight private-RNG sampled sets per existing state supply pairwise Jaccards. States: original entry, greedy endpoint, last-group entry and branch 0. No intermediate checkpoint availability is assumed.

- RIM_ONE_r3 / 601: mean KL to uniform 0.000881948; mean max probability 0.0194865; policy gradient norm 0.0150737, weighted entropy gradient norm 4.98223e-05, angle 148.06705340583687.
- RIM_ONE_r3 / 602: mean KL to uniform 1.06177e-05; mean max probability 0.0182897; policy gradient norm 0.0352332, weighted entropy gradient norm 1.18625e-05, angle 10.257546184063765.
- Drishti_GS / 601: mean KL to uniform 0.00139388; mean max probability 0.0305637; policy gradient norm 0.0178797, weighted entropy gradient norm 8.5994e-05, angle 159.32557399317767.
- Drishti_GS / 602: mean KL to uniform 1.58551e-05; mean max probability 0.0282044; policy gradient norm 0.0111283, weighted entropy gradient norm 1.03445e-05, angle 169.40509426384125.

Gradient diagnostics average the four saved group samples at the saved behavior policy, before any Adam step or clipping. They do not reconstruct later policy-update epochs or justify changing entropy regularization. High entropy alone is not evidence of excessive entropy regularization; greedy repetition alone is not stochastic distribution collapse.

## Exposure

Historical global counts are actual saved training exposures. Historical per-window counts are reconstructed from the unchanged loader indices and checked against those saved totals. New windows record actual exposure deltas. Anonymous sorted distributions preserve every count without publishing image mappings. Coverage does not imply within-window balance.

## Reward

Only the last saved group branch 0 has matched before/after states. Its total delta was reproduced within 1e-6, using the saved original anchors, coefficient 0.1, threshold >0.7 and all-pixel denominator. Quality is negative balanced NLL plus rim/cup soft Dice loss. Source KL channel terms may individually be negative; their sum is KL. Other branches/earlier groups are unavailable for decomposition; no training was replayed. Source predictions on target U do not guarantee REFUGE retention.

## Numerical summary

See [MECHANISM_SUMMARY.md](MECHANISM_SUMMARY.md) for the four-policy, historical-exposure and reward-component tables.

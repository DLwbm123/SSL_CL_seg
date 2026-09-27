# Frozen reward contract

Quality is negative per-image class-balanced NLL plus mean rim/cup soft Dice loss, with log floor 1e-8 and smoothing 1. Feedback is clean, FP32, eval/inference_mode, preserving RNG and model mode. Supported GT classes alone enter class-balanced NLL; missing foreground remains in Dice. All-ignore images are excluded. Empty feedback is invalid, never fabricated.

Each candidate reward subtracts an independent same-state L-only anchor. The 32 predetermined coarse/fine R3a rewards per backbone/domain determine RMS scale with floor 1e-4. No mean centering, clipping, or future validation information enters advantage. Online fold alone updates controllers; audit fold is diagnostic only. These folds were seen by the historical prefix and are not independent validation.

RL and NC_RL use the actual 0.9*softmax+0.1/3 mixture for behavior ratios and exact categorical reference KL. Four updates at most per valid decision; exact-zero advantages may skip RL calls. Policy reference remains fixed throughout those updates and is EMA-updated once afterward. REG uses sampled-action MSE with the same mixed action distribution, without policy-gradient or KL. Current rewards only change future actions.

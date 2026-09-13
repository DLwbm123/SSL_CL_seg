# Post-hoc SIGN analysis

Old seeds61/62/63, never the old primary. Intervals are computed directly from saved paired patient scores with2000 shared physical-domain draws (2026091301), not by subtracting CI endpoints. No training, model forwards or image/GT arrays.

Probe absolute coefficients match at fixed delta, but q_sign has RMS/q_align=1/sqrt(n_eff). R and Rnorm also change, so this does not prove eta is smaller. SIGN is not a strictly amplitude-matched direction ablation. See per-task b/Rnorm/eta and acceptance. accepted_algebra_eta_over_d0 is computed jointly from steps; the unrecorded exact FP32 applied norm is NOT_RECORDED, not imputed. Guard nonincrease is imposed, not independent memory evidence.

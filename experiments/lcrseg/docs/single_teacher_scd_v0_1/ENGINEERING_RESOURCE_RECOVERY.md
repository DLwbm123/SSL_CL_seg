# Hardware admission recovery; scientific design unchanged

The initial parent required 8,192 MiB free on each assigned GPU. While common stage0 was running, other jobs reduced free memory on GPU5/7 to about 4,182/4,186 MiB. The parent declined A/C/E before any of their model updates. S/B continued. All initial parent receipts remain preserved.

The user explicitly authorized GPU4/5/6/7 when sufficient memory remains even alongside other jobs. One additional all-synthetic 2x3x384x384 E step, on the exact frozen training source 057ce07, measured 985,934,848 allocated bytes and 1,098,907,648 reserved bytes. Adding 1 GiB headroom gives a 2,072 MiB hardware admission floor. This measured requirement fits the available GPUs. A/C/E were then admitted on their originally assigned lanes: GPU5 A then E; GPU7 C. This is a hardware admission override of our initial conservative floor, not a change to any scientific threshold, learning rate, epoch budget, objective, input cohort or seed.

The probe used no real image/GT, made exactly one separately accounted synthetic optimizer update, and its weights were discarded. No formal update was replayed. All formal stages still use the same source, configuration and 39,800-update matrix. The recovery driver refuses to overwrite any started stage, waits for sufficient free memory, and records every real child exit. No other user's process was killed.

At closeout, the initial parent admission outcome and recovery receipt will be retained separately. A combined reporting view may be built only after checking all 13 real stage receipts, unchanged source identities, last-step evaluations and final deployment checks. Scientific adjudication remains a single pass after the complete matrix.

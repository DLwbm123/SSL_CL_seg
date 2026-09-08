# Anchored mix stability is running

Execution source: `800cabe5e5dd69612738aaba45ad35121d63c0f2`.

GPU 3/4/5, one formal task per GPU. Parent PID 181437 is detached with PPID1. First startup check observed 781 actual optimizer updates across three live tasks, with no failures. This is a startup snapshot, not a current or final result.

P1: all 24 tasks, 63600 formal updates, seeds31/32/33 and both domains. P2: automatically admitted only by the frozen selection rule, then all fixed seed41/42 tasks run; maximum total formal updates106000. No CL, new hyperparameters or additional seeds.

Qualification: four recorded attempts (one zero-update development assertion failure preserved), 426 successful synthetic optimizer updates; exact source CPU/CUDA passed. Real labeled-only smoke8 updates, four actual labeled sample opens, zero real U images opened; initialization discarded.

NAS root: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/ssl_anchored_mix_v0_1_20260908_01`. Parent log `runner.log`; individual logs and step records under `P1/seed*/<domain>/`. Qualification and source checks are complete; final metrics and scientific terminal are pending.

# MMPR-GS exact-source and call-graph freeze

Exact execution code: `bda0af8e25db492785ff09315b2722042e0174e0`, pushed and remotely verified before private-input access. The existing server environment passed326 tests (246 unchanged Gate1C diagnostic regressions plus80 new tests),0 failures,0 errors,0 skips. The durable server parent recorded actual child exit0. The separately disclosed old B0-training and old-private-input integration modules remain excluded as registered; no old test was modified.

The same pair kernel compiled on synthetic inputs uses3 native FP32 forwards and2 same-Gaussian FP64 forwards,3 native autograd calls and6 FP64 autograd calls. Per-pair output coverage and full parameter/None inventories are bound in the JSON. Three integration pairs cost15 real forwards;72 formal pairs cost360. Total375. Source files are frozen by both the commit and SHA256 manifest.

No new real forward, private checkpoint tensor read, validation-cache array read or scientific result has occurred at this gate. Next: full private bundle SHA verification, then the authorized read-only evaluator and fixed draw0 phases. All large outputs stay on NAS.

# R0 CPU synthetic acceptance — passed

The results below apply to the original R0 snapshot. External review subsequently found F1/F2 boundary defects. After explicit user approval of a separate budget, their repairs passed all three targeted checks on the first of at most two runs with PyTorch 2.6.0+cu124 on CPU. Optimizer calls, real-data reads, and CUDA calls were all zero. FP32/FP16/BF16 inputs and CPU BF16 autocast were checked for normalized finite outputs and finite gradients; this is not CUDA/AMP qualification. `review/REVIEW_FIX_RESULTS.json` records the tested commit and hashes. `run_review_fix.py` uses an independent named ledger; original R0 ledgers remain unchanged at 2 suite attempts and 5 optimizer calls.

- `python3 -m compileall` passed for R0 Python files.
- Attached migration fixture suite: **16/16 passed**, zero optimizer calls. One added check covers the observed separate metadata/payload roots.
- On the target, suite attempt 1 passed **6/6** checks. Suite attempt 2 passed **8/8** checks, including exact selected Eq.13 value, singleton/group edge case, charged failure, and full prefix-state replay. The two-run ledger records **2 started / 2 successful** suite attempts and **5 attempted / 5 successful** physical synthetic segmentation optimizer calls (cap64). The fake failure accounting check used its own temporary ledger and did not call a real optimizer.
- Official DINOv2 source imported offline, the SHA-bound weight loaded strictly into ViT-S/14, and both U-Net and ViT query inference produced identical CPU outputs before and after removing the separate prototype bank and EMA reference. No implicit download occurred during model construction. The upstream xFormers fallback emitted warnings only.
- Real-data optimizer updates, CUDA model tests, L-only smoke, and formal R1: **0**.

Target M1 role construction returned RIM 16 L / 40 val and Drishti 10 L / 25 val; U was denied. The check did not read payload arrays. These checks do not establish HDF5 pixel/medical semantics, GPU readiness, training convergence, or scientific benefit.

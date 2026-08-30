# Gate 0 resume equivalence report

Date: 2026-08-29
Benchmark/seed: Fundus seed 0
Model: LCRSeg UNet2D with official JASCL 3x3 stochastic classifier
Device: NVIDIA GeForce RTX 3090, logical GPU 0
Tolerance: `atol=1e-6`, `rtol=1e-6`

Compared trajectories:

1. uninterrupted from initialization through global step 6;
2. interrupted after global step 3, restored from the complete checkpoint, and
   continued through global step 6.

Result: `PASS`.

| State group | Within tolerance | Maximum absolute difference |
|---|---:|---:|
| student | yes | 0.0 |
| EMA teacher | yes | 0.0 |
| optimizer | yes | 0.0 |
| scheduler | yes | 0.0 |
| GAS `grad_update` | yes | 0.0 |
| RNG state | exact | 0.0 |
| stage state | exact | 0.0 |
| sampler phase/offset | exact | 0.0 |

The former DeepLab comparison is invalidated and excluded. This UNet comparison
restored student, EMA teacher, optimizer, scheduler, stage/epoch/global step,
GAS, Python/NumPy/CPU/CUDA RNG, sampler state, PAS prototype slot, config hash,
and evaluation matrices. Machine-readable evidence is at
`/root/LCRSeg/runs/gate0_repaired_unet_resume_equivalence.json`.

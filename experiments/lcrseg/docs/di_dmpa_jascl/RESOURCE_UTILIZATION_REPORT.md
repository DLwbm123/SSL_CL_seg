# Resource scheduling audit

Date: 2026-08-30. This change responds to the user's request to maximize use
of both GPUs without changing the frozen Gate 0 experiment.

## Diagnosis

The container shows 112 host CPUs but cgroup v1 CPU quota is
`1600000 / 100000 = 16` cores. Both seed-0 Python processes created about
56 active CPU compute threads each, using about 800% CPU each; the cgroup
reported CPU throttling. Each RTX 3090 used about 1.4 GiB of 24 GiB memory.
This was CPU oversubscription, not exhausted GPU memory.

Six `nvidia-smi dmon -s pum -d 2 -c 6` samples during late seed-0 training:

| GPU | SM utilization samples (%) |
|---|---|
| 0 | 6, 12, 1, 11, 8, 5 |
| 1 | 5, 0, 6, 10, 6, 6 |

A separate `pmon` snapshot recorded 31% / 26%. Utilization is phase-dependent;
these are observations, not an isolated controlled throughput benchmark.

## Change and numerical validation

No running seed-0 process was interrupted or altered. After seed 0 completed
and passed its compiler/checkpoint gate, seeds 1/2 use:

```text
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
GPU 0: C0 seed 1, C0 seed 2
GPU 1: B0 seed 1, B0 seed 2
```

No training-source, batch-size, precision, optimizer, schedule, thresholds,
loss weights, data order, or RNG-seed change was made. The two-process-per-GPU
scheduling is only for the already authorized independent runs.

The resource validation suite passed 52/52 tests with no skips. The four
synthetic actual-model reference trajectories match the original-thread
reference trajectories within `atol=rtol=1e-6`. The same fixed real batch
in each Fundus domain reproduced all recorded PAS counts, losses, and
gradient norms exactly (difference 0). See
`resource_audit_threads1/RESOURCE_UTILIZATION_AUDIT.json` and its source
test/resume/gradient reports, JUnit, and transcript. The 213.20-second resource
test ran while seed-0 jobs occupied the same CPU/GPU resources; its wall time
must not be compared as an isolated speed benchmark with the original suite.

## Observed utilization after launch

Ten `nvidia-smi dmon -s pum -d 2 -c 10` samples:

| GPU | SM utilization samples (%) | Average | Peak |
|---|---|---:|---:|
| 0 | 66, 90, 92, 87, 88, 94, 69, 64, 73, 64 | 78.7% | 94% |
| 1 | 39, 28, 69, 88, 91, 89, 91, 92, 83, 52 | 72.2% | 92% |

Each of the four Python processes used approximately 100% CPU; per-GPU
memory in that supervised-phase sample was 2107 MiB. During this sample both
cards had two assigned runs. This is a substantial observed
improvement, not a claim of permanent 100% utilization or a hardware-global
optimum. Validation, checkpoint I/O, and stage transitions can lower it.

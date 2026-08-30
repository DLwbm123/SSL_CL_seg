# Gate 0 exact commands

The archived reports in this directory were produced from JASCL Gate 0 source
commit `342f44308dd0b1492ce5d549421e52eb876e4819`. In the merged `SSL_CL_seg`
layout, run from `experiments/lcrseg` and install the pinned official JASCL
checkout once as a local, ignored provenance dependency:

```bash
cd /root/SSL_CL_seg/experiments/lcrseg
git clone https://github.com/prinshul/JASCL.git third_party/JASCL_REFERENCE
git -C third_party/JASCL_REFERENCE checkout --detach \
  3c93ca70784fc3a1d2a887f8d7dce5af6bc75f53
test "$(git -C third_party/JASCL_REFERENCE rev-parse HEAD)" = \
  3c93ca70784fc3a1d2a887f8d7dce5af6bc75f53
```

The runner refuses to start if this checkout has a different origin, commit,
missing source subtree, or tracked edits in the audited subtree.

## Test

```bash
cd /root/SSL_CL_seg/experiments/lcrseg
LD_LIBRARY_PATH=/lib/x86_64-linux-gnu PYTHONPATH=. CUDA_VISIBLE_DEVICES= \
  /root/.venvs/lcrseg-py310/bin/python -m pytest -q tests/gate0
```

`LD_LIBRARY_PATH` is required because the image's CUDA 13 compatibility
`libcuda` otherwise shadows the host driver library and produces CUDA error
804 on RTX 3090.

## Formal Fundus Gate 0 runs

```bash
cd /root/SSL_CL_seg/experiments/lcrseg
mkdir -p /root/LCRSeg/runs/gate0_repaired_unet_fundus_seed{0,1,2}
nohup scripts/run_gate0_job.sh 0 0 \
  /root/LCRSeg/runs/gate0_repaired_unet_fundus_seed0 \
  > /root/LCRSeg/runs/gate0_repaired_unet_fundus_seed0/stdout.log \
  2> /root/LCRSeg/runs/gate0_repaired_unet_fundus_seed0/stderr.log &

nohup scripts/run_gate0_job.sh 1 1 \
  /root/LCRSeg/runs/gate0_repaired_unet_fundus_seed1 \
  > /root/LCRSeg/runs/gate0_repaired_unet_fundus_seed1/stdout.log \
  2> /root/LCRSeg/runs/gate0_repaired_unet_fundus_seed1/stderr.log &

# Run after GPU 0 is free.
nohup scripts/run_gate0_job.sh 2 0 \
  /root/LCRSeg/runs/gate0_repaired_unet_fundus_seed2 \
  > /root/LCRSeg/runs/gate0_repaired_unet_fundus_seed2/stdout.log \
  2> /root/LCRSeg/runs/gate0_repaired_unet_fundus_seed2/stderr.log &
```

## Resume a formal run

```bash
cd /root/SSL_CL_seg/experiments/lcrseg
LD_LIBRARY_PATH=/lib/x86_64-linux-gnu PYTHONPATH=. CUDA_VISIBLE_DEVICES=GPU_INDEX \
  /root/.venvs/lcrseg-py310/bin/python scripts/run_gate0_repaired.py \
  --config configs/gate0_repaired/fundus.yaml \
  --seed SEED \
  --output-dir /root/LCRSeg/runs/gate0_repaired_unet_fundus_seedSEED \
  --device cuda \
  --resume /root/LCRSeg/runs/gate0_repaired_unet_fundus_seedSEED/last.pt
```

## Verify interrupted/resumed trajectory

```bash
cd /root/SSL_CL_seg/experiments/lcrseg
LD_LIBRARY_PATH=/lib/x86_64-linux-gnu PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
  /root/.venvs/lcrseg-py310/bin/python scripts/verify_resume_equivalence.py \
  --reference /root/LCRSeg/runs/gate0_repaired_unet_resume_reference/last.pt \
  --candidate /root/LCRSeg/runs/gate0_repaired_unet_resume_candidate/last.pt \
  --output /root/LCRSeg/runs/gate0_repaired_unet_resume_equivalence.json \
  --atol 1e-6 --rtol 1e-6
```

## Compile Gate 0 reports

```bash
cd /root/SSL_CL_seg/experiments/lcrseg
PYTHONPATH=. /root/.venvs/lcrseg-py310/bin/python \
  scripts/compile_gate0_reports.py \
  --runs-root /root/LCRSeg/runs \
  --output-dir docs/di_dmpa_jascl
```

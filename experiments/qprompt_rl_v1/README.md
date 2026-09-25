# SSLCL_QPROMPT_RL_V1 — R0 review package

This branch prepares two query segmentation backbones and GRQA primitives. It has no training launcher and cannot start R1, R2, or R3. `R1_PLAN.json` keeps every scientific phase disabled.

The study adapts the supplied 16-page QPrompt-R1 paper to canonical fundus segmentation. The [official QPrompt-R1 repository](https://github.com/straybird2333/QPrompt-R1) had only a README on 2026-09-25; this code is an independent reimplementation. The ViT source is a pinned minimal subset of official DINOv2; its pretrained weight is still absent.

Local source entry points: `qprompt/models.py`, `qprompt/losses.py`, `qprompt/data.py`, `qprompt/state.py`. `run_cpu_tests.py` is the bounded CPU synthetic suite; `migration/minimal_manifest.py` builds an exact M1 byte manifest after source roots and a clean code commit are bound. `RUNBOOK.md` gives the transfer protocol. Importing the package never downloads weights.

Current R0 status is in `CPU_REPORT.md`, `COUNTS.md`, and `REVIEW_REQUEST.md`. No real training, CUDA model run, real smoke, or old experiment recovery has occurred.

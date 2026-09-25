# Environment lock — pending target binding

New server SSH alias, remote-home realpath, owner, mount/quota/inodes, GPU inventory, scheduling policy, driver, CUDA runtime, Python/PyTorch build, and DINOv2 weight hash are **UNBOUND**. No environment has been installed. The old server's Python 3.10.6 / torch 2.2.1+cu121 is an inventory fact and is not selected as this study's runtime.

Required target layout: confirmed `${REMOTE_HOME_ABS}/sslcl_qprompt_rl_v1`; `HF_HOME`, `TORCH_HOME`, `PIP_CACHE_DIR`, and `TMPDIR` must point inside it. No old NAS wrapper or implicit model download. Pin actual versions after target hardware/driver inspection, without changing system drivers or existing environments.

Runtime package needs: Python≥3.10, PyTorch with a target-compatible build, NumPy (official DINOv2 source), h5py for real canonical HDF5, and standard library. The included official DINOv2 ViT-S/14 subset imports without xFormers using its upstream fallback. This import closure still requires offline dynamic verification on the target.

# External CPU source dependency setup

The review does not mirror repositories without declared redistribution licenses. `DEPENDENCY_LOCK.json` gives exact immutable file URLs and SHA256 fingerprints. Source retrieval is optional and explicit; nothing is downloaded by training, planning or import.

From the repository root, this retrieves the four small author source files needed for CPU verification into a local directory:

```sh
python -m experiments.lcrseg.five_frameworks_v1.fetch_dependencies --destination /your/local/source_dependencies
export SSLCL5_DEP_ROOT=/your/local/source_dependencies
python -m experiments.lcrseg.five_frameworks_v1.cli qualify --synthetic --device cpu
```

The destination is not placed in the repository. Existing matching files are reused. A conflicting file raises instead of overwriting it. Only `.py` source files named in the frozen lock are retrieved, never datasets or model tensors. This verifies source bytes because the current review explicitly requires source fingerprints.

CWMI uses installed torch, torchvision, numpy, matplotlib and Pillow due to the author's imports. Version evidence is in the lock. No environment creation or torch reinstallation is required. F2 explicitly refuses CUDA until a separately reviewed device-local backend and formal optimizer qualification are supplied. Other kernels have CPU evidence only; absence of a CUDA test is never treated as a CUDA pass.

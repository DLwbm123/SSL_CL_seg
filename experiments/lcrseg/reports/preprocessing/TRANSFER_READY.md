# LCR-Seg transfer readiness

Local preprocessing and local checksum verification have passed.

- HDF5 files: 2932 (1843390563 bytes)
- HDF5 pairs: 1466
- Checksum: checksums/checksums.sha256 (pass)
- Transfer bundle: pass
- Configured direct endpoint: jiangsuiyang@10.12.208.180:22 via local source address 10.75.81.150.

Before any transfer, provide exactly these two explicit values (do not guess them):

```bash
export LCRSEG_REMOTE_ROOT=/absolute/existing/writable/LCRSeg/path
export LCRSEG_REMOTE_PYTHON=/absolute/path/to/python-with-h5py-and-numpy
```

Then run the repository synchronizer first without `--execute` for its rsync dry run, review the itemized output, and rerun with `--execute`.

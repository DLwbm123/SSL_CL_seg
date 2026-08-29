# LCR-Seg data workspace

This workspace implements the data-only phase of the LCR-Seg plan. It never
modifies source data under `/Volumes/DataP`.

The local preprocessing phase is complete and frozen at
`/Volumes/DataP/LCRSeg/h5/v1`. The derived bundle contains 1,466 image/label
pairs, fixed training/diagnostic manifests for seeds `0`, `1`, and `2`, and a
verified `checksums/checksums.sha256` list. See `STATUS.md` and
`reports/preprocessing/PREPROCESSING_COMPLETION.md` for the acceptance record.

Run commands use the existing `/opt/miniconda3/bin/python` runtime; no Python
environment is created or modified by this workspace.

The data transfer script requires an explicit, existing absolute remote root
and an explicit remote Python interpreter. No transfer has been performed.
For example, after local acceptance:

```bash
export LCRSEG_REMOTE_ROOT=/absolute/existing/path/LCRSeg
bash scripts/sync_to_jiangsuiyang.sh --execute \
  --local-root /Volumes/DataP/LCRSeg \
  --remote-host 10.12.208.180 \
  --ssh-user jiangsuiyang \
  --ssh-port 22 \
  --bind-address 10.75.81.150 \
  --remote-root "$LCRSEG_REMOTE_ROOT"
```

It prompts for SSH authentication rather than storing a password, never uploads
raw source images, uses no `--delete`, and verifies the remote inventory
separately.

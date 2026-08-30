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

## JASCL Gate 0 repaired baseline

The fixed-class Fundus Gate 0 semantic repair is implemented in
`di_dmpa_jascl/`. It uses the frozen medical UNet2D body and the official JASCL
3x3 stochastic classifier, with `method_registered=false`; no DI-DMPA method
training is included. Protocol, repair ledger, reports, three-seed matrices,
and reproduction commands are under `docs/di_dmpa_jascl/`. V1's overall PASS
is withdrawn because its unlabeled gradient was zero. V2 remains blocked
until every revised audit and C0/B0 three-seed gate passes; consult the latest
`GATE0_STATUS.json`, not the historical archived PASS.

The official JASCL checkout is intentionally not vendored. Follow
`docs/di_dmpa_jascl/EXACT_COMMANDS.md` to install the pinned reference at
`third_party/JASCL_REFERENCE` before running the Gate 0 tests or runner.

# LCR-Seg local preprocessing completion

Generated: 2026-08-19T06:28:10Z

## Completed HDF5 cohorts

- Fundus: 660/660 accepted pairs; failed=0; label values={0,1,2}; minimum crop retention=1.0
- Prostate: 116/116 accepted pairs; failed=0; manual review=0; label values={0,1}.
- Prostate geometry: 31 mismatches, distribution={"index_geometry_repair": 31}, manual review=0.
- M&Ms: patients=345; ED/ES phase pairs=690; canonical320=320 patients; auxiliary25=25 patients; fixed FOV=320 mm; minimum foreground retention=1.0; failed=0.

## Closing gates

- Full HDF5 schema acceptance: pass; 2932 HDF5 files / 1466 pairs.
- Runtime manifest and hidden-label isolation: pass; 3 fixed seeds, 1466 rows per seed.
- Raw-source provenance rehash: pass; 1121 source pairs and 1466 image provenance attrs checked.
- DataLoader smoke: pass; workers tested=[0, 4].
- SHA-256 checksum gate: pass; path=checksums/checksums.sha256.
- Transfer-bundle verification: pass.
- Active failure bundles: 0.

## Frozen artifact

- Marker: /Volumes/DataP/LCRSeg/h5/v1/FROZEN
- Inventory: /Volumes/DataP/LCRSeg/reports/preprocessing/h5_inventory.csv (2932 HDF5 files)
- Stored HDF5 bytes: 1843390563; payload/HDF5 ratio: 2.5235.

## Next bounded command (not run in this preprocessing phase)

```bash
/opt/miniconda3/bin/python scripts/two_case_overfit.py --root /Volumes/DataP/LCRSeg --seed 0 --dataset fundus --steps 200
```

Remote transfer is ready but not executed: it still requires an explicit remote absolute root and an explicit remote Python interpreter.

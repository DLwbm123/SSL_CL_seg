# LCR-Seg preprocessing gate

Audits and fixed split creation are complete. No raw file under `/Volumes/DataP`
has been modified and no derived HDF5 has been written.

| Dataset | Audit result | Required decision before HDF5 |
| --- | --- | --- |
| Six-site prostate | 116 pairs; 85 exact geometry matches; 31 geometry mismatches; raw label 2 in 60 cases | Confirm whether raw label 2 is merged into foreground and complete every row in `geometry_decisions_template.csv`. |
| M&Ms | 345/345 local official-metadata matches; vendors: Siemens 95, Philips 125, GE 75, Canon 50; ED/ES labels 0/1/2/3 | Confirm the official raw-label semantics and whether the 25 `Training/Unlabeled` GT files may be used for research evaluation. |
| REFUGE + RIM-ONE + Drishti | 660 paired images/masks; all masks have 0/128/255; 100% foreground retained by 800-center crop | Confirm which code is optic-disc rim and which is optic cup. |

The files to review are:

- `geometry_decisions_template.csv` and `../qc_overlays/prostate_geometry/` for the 31 prostate cases;
- `prostate_label_audit.csv` for the observed prostate label-1/label-2 topology;
- `mnms_vendor_mapping.csv` for local ED/ES and vendor provenance;
- `fundus_audit_summary.json` and `../qc_overlays/fundus/` for the raw fundus encoding.

For transfer after local acceptance, supply all of the following explicitly:

1. An existing absolute `jiangsuiyang` destination, for example `/absolute/path/LCRSeg`.
2. An absolute remote Python interpreter that has `h5py` and `numpy`.
3. A working SSH route to `jiangsuiyang`.

The transfer command will first validate the frozen local bundle, perform a
reviewable dry run by default, then require `--execute`; it never uploads raw
source files or uses deletion semantics.

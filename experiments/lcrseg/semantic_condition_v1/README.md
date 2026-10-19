# Spatial semantic-conditioned residual pilot

Outcome: the frozen pilot **failed its joint SSL / historical segmentation benefit criteria**. This is a SCOPE-inspired convolutional mechanism study, independent of CARE-Seg, not an established TMI contribution or an exact reproduction of the authors' EEG experiments.

- [Frozen protocol and pre-formal amendment](PROTOCOL_ZH.md)
- [Chinese completion report](results/REPORT_ZH.md)
- [Anonymous aggregate results and audits](results/SUMMARY.json)
- [Metrics](results/metrics.csv)
- [Real engineering receipts and dated launch snapshot](ENGINEERING_CHECKS.json)

The five arms each execute 3,500 formal calls from the same completed Siemens FixMatch final EMA to the original Philips 13 labeled patients and unlabeled stream. The four adapter arms use the original frozen U-Net and valid 3×3 head; only 3,328 residual parameters train. The historical class prototypes are private frozen buffers. All cases refer to existing frozen manifests; no test or hidden U labels are admitted.

`module.py` implements the bounded spatial conditioner. `core.py` supplies setup, the uncompressed cosine schedule, original FixMatch loss and residual-only EMA. `run.py` registers these with the existing `philips_transfer_v1` training engine; it does not contain a second training loop. `statistics.py` performs the one bounded source-L pass. `qualification.py` charges discarded engineering calls including the resume replay. `controller.py` is a create-only queue with no automatic retries. `summarize.py` audits the original run including the retained v1 failure and exports no patient rows; it is intentionally strict about that original receipt layout.

Private configuration is supplied through `TASK_SETTINGS`; `TASK_MODE` selects `statistics`, `qualify`, `train`, `controller` or `summarize`. Copy `entry.py` to a neutral launcher location and invoke it through the approved storage wrapper. No patient paths, host configuration or credentials are included here. The caller must explicitly supply:

| Configuration | Meaning |
|---|---|
| `base`, `overlay`, `code` | Existing dependency checkout and this source overlay |
| `data`, `reference` | Frozen data bundle and existing native model reference |
| `source_checkpoint`, `source_val_rows` | Completed step-43600 FixMatch EMA and private source validation rows |
| `split` | Private original Philips 13L train list with empty diagnostic holdout |
| `statistics`, `output`, `budget_root` | Create-only run artifacts on approved storage |
| `arm`, `phase`, `phase_cap` | Registered arm, charged phase and hard per-phase call cap |
| `runtime`, `wrapper`, `entry`, `root` | Controller runtime and neutral wrapped worker entry |

The original v2 controller uses `V2_SETTINGS_<arm>.json`, `V2_QUAL_<arm>.json`, `statistics_ready=true`, and reuses only the previously passed Dense engineering receipt. Those receipts are runtime inputs, not public patient metadata. The first v1 qualification failure remains separate. `budget_events.jsonl` reserves under a lock before each call and acknowledges execution afterward; interrupted unacknowledged reservations conservatively occupy capacity.

Synthetic tests: `python -m unittest experiments.lcrseg.semantic_condition_v1.test_module`. These tests use random tensors, not patients, and do not execute an optimizer. The actual completed run passed all real-data gates. Final cost was 17,692 calls (17,500 formal + 192 engineering); the remaining 2,308 authorized calls were not used for tuning or extension. Historical source training cost 43,600 is separate. All five final EMA checkpoints and exact patient/phase sets passed the private server audit.

Public artifacts exclude weights, source prototypes/features, patient records/images, raw logs, private run configuration and third-party paper/code bundles. Reproduction requires authorized access to the original private assets; their absence must not be worked around by changing the data split or reading hidden labels.

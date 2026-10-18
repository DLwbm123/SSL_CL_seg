# Minimal convolutional SPACE-LoRA pilot

See [PROTOCOL_ZH.md](PROTOCOL_ZH.md) for the frozen study and
[results/REPORT_ZH.md](results/REPORT_ZH.md) for completed results and limits.
This implementation is separate from CARE-Seg.

The original target-domain training loop remains `philips_transfer_v1.run`.
`run.main(cfg)` registers the new arms; `core.py` contains the adaptation;
`statistics.py` collects the source boundary statistics;
`qualification.py` performs separately charged engineering checks;
`summarize.py` audits private artifacts and exports anonymous aggregates.

For a new independently authorized reproduction, copy `entry.py` to a neutral
entry filename outside a method-named directory; copy the existing storage
wrapper to a neutral filename too. All command lines and children must satisfy
project rules. Pass configuration through `TASK_SETTINGS` (JSON), with
`TASK_MODE` one of `prepare`, `statistics`, `qualify`, `train`, `controller`.
Use `CUDA_VISIBLE_DEVICES=3`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and the authorized
storage root via the wrapper. The entry sets the CUBLAS configuration before
importing PyTorch. Do not replay the completed run or overwrite its directories.

Worker configuration keys:

| Key | Meaning |
|---|---|
| `base`, `code` | Full compatible project checkout and source overlay; may be the same checkout |
| `data` | Existing frozen dataset root; no hidden U labels |
| `reference` | Clean, pinned native upstream checkout checked by the existing builder |
| `source_checkpoint` | Completed seed163 FixMatch final checkpoint with 43,600 physical updates |
| `source_val_rows` | Corresponding private Siemens validation records |
| `split` | Private JSON created once by prepare: all original 13 Philips L patients, empty holdout |
| `statistics` | Create-only private boundary-statistics checkpoint, outside Git |
| `arm` | `dense_fixmatch`, `free_lora`, or `space_lora` |
| `output` | Fresh private job directory; qualification and formal directories must differ |
| `preflight_only` | `true` for qualification; `false` for formal training |
| `diagnostic_holdout_excluded` | `false` for this study |
| `audit_source_training` | `true` for the formal final labeled-source training audit |

The controller additionally needs `root`, `storage_root`, `wrapper`, `entry`,
`runtime`, `deadline_unix` and `base`/`code`. Under `root`, prepare
`SETTINGS_<arm>.json` and `QUAL_<arm>.json` with matching frozen settings and
separate fresh outputs. It runs source statistics once, qualifies each arm,
then launches all three arms; any qualification failure blocks formal launch.
It never retries a job. Actual account paths, private patient lists, source
weights, statistics, logs and server configuration are deliberately excluded.

## Costs and assertions

Formal cap: 4,000 actual SGD calls per arm (12,000 total). Engineering cap:
32 per arm (96 total), discarded. No further experiment is implied by this
README. Sampling is arm-independent in the reused `CurrentDomain` provider.
Original source weights are identical at entry; the low-rank arms freeze the
same base/head parameters and share the same pre-projection A initialization.
EMA is computed from effective dense weights, not by averaging factor products.

Run the small mathematical contract suite in the compatible runtime with
`python -m unittest experiments.lcrseg.space_lora_v1.test_contract` before GPU
qualification. Full data-dependent checks and final audits were completed for
the delivered run; results include their aggregate receipts.

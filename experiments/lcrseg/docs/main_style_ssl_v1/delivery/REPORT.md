# MAIN_STYLE_SSL_V1: completed three-dataset study

All 30 final nodes completed. This is a frozen validation report, not an independent-test or comparative superiority claim. Both favorable changes and severe forgetting are retained.

## Fixed protocol and metric definitions

- Seeds: 163 and 164. Fundus has two target orders per seed; Prostate and M&Ms have one order. Sources are shared between the two fundus orders, and are counted once in physical cost.
- 511600 formal optimizer calls: fundus 37200, Prostate 177400, M&Ms 297000. No ablation, additional seed, post-hoc checkpoint selection or independent-test access.
- Final = equal-domain mean at the fixed last checkpoint; Old = mean of previous domains at that checkpoint; Incoming = last domain. Forgetting = signed mean of acquisition-stage Dice minus final Dice over old domains. It is neither peak-history forgetting nor clipped at zero.
- Fundus primary Dice averages rim and cup; cup/disc-union is auxiliary. Prostate uses foreground Dice. M&Ms uses foreground-class mean per the frozen validation implementation. Metrics below are percentages; forgetting is percentage points.

## Final results

| Dataset | Trajectories | Final | Old | Incoming | Forgetting |
|---|---:|---:|---:|---:|---:|
| fundus | 4 | 69.3264 | 66.4986 | 74.9819 | 12.5865 |
| prostate | 2 | 44.6315 | 44.0882 | 47.3481 | 17.0553 |
| mnms | 2 | 71.3649 | 69.3420 | 77.4337 | 4.4088 |

Fundus averages four seed/order trajectories, not four independent seeds. Sample standard deviations are descriptive and are retained in AGGREGATES.json; no confidence interval or significance claim is made.

| Dataset | Seed | Order | Final | Old | Incoming | Forgetting |
|---|---:|---:|---:|---:|---:|---:|
| fundus | 163 | 1 | 73.0023 | 71.6076 | 75.7917 | 6.9620 |
| fundus | 163 | 2 | 68.1607 | 66.6501 | 71.1818 | 13.1752 |
| fundus | 164 | 1 | 70.2108 | 66.8128 | 77.0067 | 11.2811 |
| fundus | 164 | 2 | 65.9317 | 60.9238 | 75.9475 | 18.9276 |
| prostate | 163 | 1 | 43.4136 | 43.1577 | 44.6931 | 20.0062 |
| prostate | 164 | 1 | 45.8495 | 45.0188 | 50.0032 | 14.1043 |
| mnms | 163 | 1 | 70.8130 | 68.6563 | 77.2829 | 1.8389 |
| mnms | 164 | 1 | 71.9168 | 70.0276 | 77.5846 | 6.9787 |

## Favorable and negative observations

- M&Ms final mean is 71.3649%, with 4.4088 pp signed forgetting. Seed163 GE improves by 5.9389 pp after acquisition, while its Siemens drops by 8.4024 pp. Seed164 loses 9.4313/8.0565/3.4481 pp on Siemens/Philips/GE. A positive domain change does not erase the negative ones.
- M&Ms also has severe intermediate forgetting: after GE, Siemens can collapse before partial recovery at Canon. The full matrix and infinite distance values are retained; final recovery does not invalidate those negatives.
- Prostate final mean is 44.6315%. Seed163 I2CVB acquires about80.10%, reaches0% at UCL, and ends23.2121%. Seed164 BMC acquires about63.83% and ends22.9969%. Earlier stage collapses remain in the complete matrix rather than being hidden by the last-stage average.
- Fundus final mean is69.3264%. In order2, acquired Drishti cup Dice falls from76.3548% to52.5621% (seed163), and77.4176% to41.7643% (seed164). This is not a resolved-forgetting result.
- No matched baseline was launched within this study. These values cannot establish improvement over another method, and no independent-test performance is claimed.

## Completion, physical consumption and preserved failures

- Formal ledgers are sequential and complete for all30 nodes, totaling511600 calls. Final/sealed file-presence, committed receipt, size and final-step checks pass; no bytewise identity claim is made.
- All30 node EXIT codes are0. The root controller writes STATUS/FINAL rather than a root EXIT; its absence is not treated as failure. The original Prostate seed164 failed BMC admission and four BLOCKED descendants remain in the private history and anonymous PRIOR_FAILED_BLOCKED.json. That attempt used0 optimizer calls; the RUNMC source was not rerun.
- GPU engineering ledgers contain0+8+108+312+100=528 optimizer calls, separate from the formal budget. The original two resume/geometry failures and device-object failure are retained, as are the subsequent passing checks. Observed formal plus GPU-engineering calls total512128; this is not a universal count of every CPU test invocation.
- The CPU aggregate admission is PASS_REQUIRED_SCOPE (88 required checks). The original overall CPU suite failure and excluded unrelated external-source fixture error are preserved; all_selected_tests_passed remainsfalse.
- Two host migrations replayed0 formal updates and used0 additional optimizer calls for the migration checks. Both used checkpoint-aligned handoffs and retired old workers after observed target growth. Backend changes prevent a claim of bitwise identical training trajectories.
- Metadata-copy and scheduler/Adam compatibility failures during migration were retained privately. They are operational negatives, not replacement scientific results.
- Formal L/U batch reads: 511630/409280; formal image exposures: 1023260/818560. Initial bank batches are included in these worker receipts.
- Natural duration from recorded PLAN anchor (2026-10-14T01:57:24+00:00) to terminal receipt (2026-10-15T05:49:03.404503+00:00): 27.8609h. Host clock offset is about7s. Worker segmented elapsed receipts sum to 83.1889 worker-hours and must not be relabeled as natural wall time or exact active GPU-hours.
- Migration GPU-hold and pause values are separate in OPERATIONS_SUMMARY.json. Complete active GPU time/energy, CPU-test optimizer calls and all engineering image exposures were not globally instrumented; missing totals are explicitly unavailable.

## Public/private boundary and reproduction

- RESULTS.json and VALIDATION_MATRIX.csv contain all aggregate stage/domain/class validation metrics. TRAJECTORIES.csv contains every final seed/order readout; NODE_AUDIT.json contains final statuses and physical counts. Nonfinite distance metrics are preserved as explicit Infinity strings with infinite-case counts, rather than dropped. No patient identifiers, per-case arrays, images, checkpoints, private raw logs or credentials are published.
- Source is a snapshot from the working branch before V2, based on the existing public branch. Private operational history is not pushed. SOURCE_PROVENANCE.json records the executed snapshots and the source-snapshot commit; deployment paths/host aliases in operational examples are placeholders that must be configured locally. These publication substitutions were not applied to running experiments.
- Data/manifests and the external reference model code must be supplied under authorized local paths. PREREG.md and protocol.py define the split/order/budget behavior. This report was reconstructed from existing validation JSON and physical ledgers; no new training or inference was run.
- Prostate V2 is a separate ongoing study. Neither its current warning nor its unfinished results are merged into V1.

## Audit files

`COMPLETION_AUDIT.json`, `PLAN.json`, `RESULTS.json`, `AGGREGATES.json`, `TRAJECTORIES.csv`, `VALIDATION_MATRIX.csv`, `MATRIX_ROWS.json`, `COSTS.json`, `NODE_AUDIT.json`, `ENGINEERING_ATTEMPTS.json`, `PRIOR_FAILED_BLOCKED.json`, `OPERATIONS_SUMMARY.json`, and `SOURCE_PROVENANCE.json`.

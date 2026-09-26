# R1.6 readout supervision

## Final result

**READOUT_BASELINE_GAIN_NOT_ESTABLISHED.** All 56 tasks and 48 endpoint evaluations completed; 64,000 unique new updates were retained. All 56 checkpoint state audits and all 48 prefix/schedule/deployment bindings passed. Every training checkpoint remains attributed to `e94413b499bcf6b348c841d80fbde93263b4c4ef`; the repaired coordinator used `7441339b948f1735e01cd143e617317b118fcb99`.

For the eight fresh-seed cells, primary B2−B0 macro Dice averaged **−0.020380 percentage points**, below the +0.300000 threshold. Five of eight cells improved (required six). UNet averaged −0.070435 pp; DINO averaged +0.029676 pp. UNet's mean rim and cup changes were both negative. The worst macro cell (−0.215698 pp) and worst single-class cell (−0.231791 pp) stayed within their allowed deterioration limits, but the complete conjunction of criteria failed.

| Contrast, equally averaged macro delta (pp) | Seed 261 | Seed 262 | Seed 263 |
|---|---:|---:|---:|
| B2−B0, primary readout | −0.085544 | −0.087301 | +0.046542 |
| B3−B1, secondary readout | −0.047021 | +0.147340 | +0.017950 |
| B1−B0, image alignment | +0.058130 | +0.047712 | +0.346479 |
| B3−B2, image alignment | +0.096653 | +0.282353 | +0.317887 |

The secondary contrast does not replace the primary candidate. Image alignment had positive average macro contrasts in these fixed cells; this does not establish universal necessity or efficacy. Full rim/cup/macro per-cell and per-backbone factors, including interaction, are provided in the CSV tables. Historical B0/Q0 and B1/Q1 endpoint comparisons differed by at most 0.054260 pp across reported metrics; identical loss/gradient regression does not guarantee bitwise endpoint reproduction under the frozen CUDA warn-only setting.

## Forensic interpretation

The fixed prefix batches showed mean full-parameter Lset/Lreadout gradient cosines of 0.984869–0.994829 across the four cells. The additional supervision was largely aligned with the existing supervised objective on these batches; this is compatible with limited incremental benefit, but is not a causal or generalization proof. GT-stratified mean unmatched-query contribution fractions ranged from 0.00002617 to 0.00191958 (0.002617%–0.191958%). At these trained prefixes, unmatched queries contributed little of the aggregated probability mass on average. These diagnostics do not support claiming a large unmatched-query contamination mechanism. Matched/unmatched mask/class/no-object aggregates and prototype-vs-matcher route counts remain fully tabulated without publishing case-level data. All four structural checks passed: changing class output could change semantic/readout output while fixed query/prototype GRQA stayed unchanged.

## Engineering and budget closure

There were 64,098 formal/replay optimizer calls and successful commits, comprising 64,000 unique final-path updates plus 98 successful updates discarded after the coordinator interruption and replayed from valid checkpoints. No optimizer call failed. Two training Python exceptions followed one coordinator publication-race exception. A separate native-qualification assertion failure consumed two synthetic calls before repair; total synthetic calls were 18, and discarded real-L smoke calls were 16. Total physical optimizer calls were **64,132**. All charges and the original deadline were retained. Historical lost-tail work recovered by replay was 98 updates; unresolved lost tail at final closure was zero.

The race repair only changed coordinator collection and stale-queue recovery. Workers resumed on their unchanged training commit, so no checkpoint training provenance was rewritten. There was no scientific-path repair or tuning. Final phase footprint was approximately 14.79 GiB, below 32 GiB; user quota was unavailable and shared filesystem capacity was not treated as quota. All study workers exited. Private checkpoints and raw evidence remain excluded from the public release.

Status: READOUT_BASELINE_GAIN_NOT_ESTABLISHED. Endpoints 48/48. Valid updates 64000/64000.

B2-B0 is primary; B3-B1 is secondary. Seed 261 reuses the historical prefixes. Seeds 262/263 test optimization-seed replication on the same exposed validation images; they are not independent patient confirmation and no statistical significance is asserted. This is supervised baseline control, not evidence of a new SSL, continual-learning or reinforcement-learning contribution. Missing cells remain null. No R2/R3 is authorized.

Paired deltas, readout/image main effects and interaction are reported for macro, rim and cup without selecting an early checkpoint. B0/B1 comparisons against old Q0/Q1 are in HISTORICAL_REPRODUCTION.csv; loss equivalence does not imply identical endpoint scores.

## Evidence and limitations

Physical training commits: 64098; unique retained updates: 64000; replay attempts: 98; optimizer failures: 0; Python exceptions: 2; successful but discarded commits: 98. Qualification and smoke attempts are separate in TRANSACTION_AUDIT.json.

All endpoints use the final step3000 student. B2 remains the sole preregistered primary candidate. Fresh-seed criteria and each threshold are preserved in FRESH_SEED_REPLICATION.json. Factor effects are reported per cell and equally averaged within each backbone. HISTORICAL_REPRODUCTION.csv compares seed261 B0/B1 with the old Q0/Q1; any numerical differences remain visible.

The readout adds an explicit supervised target using existing L labels. Matched/unmatched masks and prototype routes are descriptive aggregates; GT-derived unmatched sets never modify training or deployment. Gradient cosines describe a fixed training batch, not generalization. The structural classifier perturbation demonstrates non-equivalence between prototype GRQA and semantic supervision, not a unique explanation of previous negative results.

CUDA determinism remains warn_only as preregistered. Exact restoration and numerical next-update equivalence are distinct checks; PATCH_LOG records the qualification assertion repair and all consumed attempts. No bitwise training reproducibility is claimed. New seeds reuse the same exposed validation images; there is no independent patient confirmation or statistical significance claim.

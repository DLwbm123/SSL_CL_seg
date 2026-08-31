# PMGC-JASCL V0.1 final feasibility report

**Scientific status: `FAIL_PMGC_FEASIBILITY`.** `G1=True, G2=True, G3=True, G4=False, G5=False, G6=True, G7=True`. All failed gates: **G4, G5**. P4 is the sole primary candidate; P1/P2/P3 cannot rescue it. All 48 registered formal pairs, six integration pairs, and six preparation units completed. This is a fixed normalized virtual-step diagnostic, not method training, a segmentation-performance result or clinical validation.

P4 improves previous-domain CE relative to P0/P1 in 33/48 pairs, with a paired median improvement of 0.000401185. It nevertheless violates the current-domain class Dice bound in three pairs and the P4-vs-P2 foreground-mode CE bound in 40/48 pairs. The worst violations are 0.003450291 Dice drop (limit 0.003) and 0.036544749 CE worsening (limit 0.0001). These pointwise safety failures remain decisive despite the favorable aggregate historical-utility and signal-retention results.

## Published authority

Registration: `PMGC_V0_1_K2_MODE_GRADIENT_CONE`. Method: Prototype-Mode Gradient-Constrained JASCL. Branch: `codex/pmgc-v0-1-feasibility`.

The unique Git and GitHub-API-verified base is `3126e59a63b205f2a075f28efa9f5d83b3911792`, from `codex/mmpr-gs-v0-1-feasibility`. Publication order: old-line closure `e1e4ac3c29b1a39ee7fc19aa84851a42570a54e1`; preregistration `efc187287b4b10b153867b20905cc3ddeefd94d9`; independent authorization `7dcac476d5ca84349d407e2f2f9ca2c8269f872e`; exact execution code `c3e044f7359c89e157561cbdfe6c9cab0ac46bb5`. Registration and authorization preceded private tensor/gradient reads. Exact code publication and the measured synthetic call-graph freeze preceded all real forwards. Source remained detached at that commit throughout execution. No historical tracked file or frozen input was modified; the report/publication additions are separate later commits.

Gate1A remains PASS_MULTI_MODALITY_SUPPORTED with K=2. Gate1B remains FAIL_TRANSPORT_NOT_SUPPORTED. Gate1C remains FAIL_IDENTITY_HISTORY_RELIABILITY_NOT_SUPPORTED, R2 admission false and candidate NONE. MMPR-GS remains FAIL_MATCHED_MASS_RANKING_NOT_SUPPORTED and FAIL_RAW_GRADIENT_COMPATIBILITY, with F1/F4/F5 true and F2/F3 false. Its 0.0780337103884 foreground error reduction, 11/18 improving units, 0.102838216357 worst precision drop, 41/72 raw negative comparisons versus 43/72, -0.0866852483267 median cosine increase, 0.188452165448 RIM worsening, and Q1/Q2 near equality remain unchanged. See the separately published closure for all frozen historical values. No selection, weighting, identity, transport or relation rescue occurred.

## Complete execution and tests

| Phase | Launch UTC | Actual child exit UTC | Child PID | Exit | Real forwards |
| --- | --- | --- | --- | --- | --- |
| input_audit | 2026-08-31T15:40:18.088516+00:00 | 2026-08-31T15:42:41.493977+00:00 | 2499898 | 0 | 0 |
| preparation | 2026-08-31T15:42:42.262885+00:00 | 2026-08-31T15:44:21.684537+00:00 | 2500170 | 0 | 309 |
| integration | 2026-08-31T15:44:26.448172+00:00 | 2026-08-31T15:45:25.055356+00:00 | 2501485 | 0 | 705 |
| audit_integration | 2026-08-31T15:45:26.914673+00:00 | 2026-08-31T15:45:57.517783+00:00 | 2503829 | 0 | 0 |
| formal | 2026-08-31T15:49:35.459356+00:00 | 2026-08-31T15:55:35.816532+00:00 | 2504231 | 0 | 5640 |
| audit_formal | 2026-08-31T15:55:39.959799+00:00 | 2026-08-31T15:56:43.921029+00:00 | 2505852 | 0 | 0 |
| orchestration | 2026-08-31T15:40:16.524970+00:00 | 2026-08-31T15:56:44.440556+00:00 | 2499893 | 0 | 0 |
| archive | 2026-08-31T15:59:56.879778+00:00 | 2026-08-31T16:10:53.053783+00:00 | 2506318 | 0 | 0 |

The orchestration/archive rows add zero forwards; they do not recount their child phases. Every controller and all 18 unit workers have durable server-parent exit 0 and closed manifests. Only GPUs 4/5/6/7 were assigned, with each physical GPU processing its registered unit queue. Other processes were not stopped. No SSH exit or last log line was substituted for child completion.

| Phase | Native forwards | FP64 forwards | Native autograd.grad | FP64 autograd.grad |
| --- | --- | --- | --- | --- |
| formal | 144 | 5496 | 96 | 96 |
| integration | 18 | 687 | 12 | 12 |
| preparation | 174 | 135 | 0 | 468 |
| TOTAL | 336 | 6318 | 108 | 576 |

Exactly **6654/6654 real forwards and 684/684 autograd.grad calls** occurred. Preparation used 309 forwards; the new integration used 705; formal used 5640. All five candidates ran all frozen panels even when a scientific gate would fail. No extra pilot, forward, optimizer step or formal retry was added. The input and artifact audits used zero model forwards and zero autograd calls.

Exact-source tests: **464 passed, 0 failures, 0 errors, 0 skips**, actual exit 0. These comprise 246 unchanged Gate1C tests, 80 unchanged MMPR tests, 102 PMGC cases covering the requested 49 categories, and 36 compatible Gate0 tests. Four individual tests were prospectively deselected, and legacy training/private-input modules were excluded before collection as recorded in the registration and `PMGC_TEST_REPORT.json`; they were not silently skipped or rewritten. The earlier 462-test development snapshot is preserved separately. JUnit SHA256: `2a77564e4a4ff7eca79c47069302944d3e135b8ed4c56c5291b588b7872ae5e4`. The compiler executes the same kernels on explicit synthetic inputs and is separate from real-input counts.

## Frozen method and evaluator

Six seed-transition units use stages 1 and 2, eight original B0 pairs each. Each unit uses all current train_labeled cases (16 for stage1; 10 for stage2) for guards, and eight fixed batches of two distinct cases on each previous/current validation side. Previous domain means the immediately preceding domain. Case identities and sampling/classifier seeds were frozen before model access.

K2 uses the previous best EMA's deterministic 16D decoder features, exact inherited Gate1A coordinates/case weights, five spherical-fit restarts and unchanged ties. Null UIDs are preserved, excluded from directional fit, and retained in global CE. No K reduction or online bank update occurred. Current-mode CE and label-consistent old-function KL guards use only visible current train_labeled GT. Old correctness is posterior-mean argmax equality, without a confidence threshold. KD modes below 32 old-correct pixels are explicitly inactive and never fabricated or merged in P4.

Raw P0 is the repaired B0 supervised CE plus 0.5 PAS probability-consistency gradient. Native teacher probabilities, joint PAS mask and Gaussian draws define the same FP64 shadow objective. All 51 trainable tensors / 484016 elements remain present, including explicit zero placeholders for None gradients. Across 54 real pairs and two precision controls, 108/108 comparisons have defined nonzero-vector metrics; minimum cosine 0.999999971636, maximum relative L2 0.000241841453342. All satisfy the preregistered parity rule, including its explicit both-zero case. Complete per-pair precision/inventory evidence is in `PMGC_EVIDENCE_CROSSCHECK.json` and the private arrays.

The deterministic small-QP projection enforces at most 13 positively normalized halfspaces and records raw guard dots, primal/dual feasibility, complementarity, objective, active set, rank, conditioning and bitwise repeat evidence. No fallback, ridge, clipping rescue or constraint deletion was used. P1 is global supervised only; P2 adds K1 class current/old guards; P3 adds K2 current guards; P4 adds K2 current and active old guards.

Each candidate uses a stateless FP64 displacement of norm 0.001, with immutable cloned buffers. Validation uses posterior-mean classifiers; training-guard panels replay their frozen native Gaussian. CE pools valid pixels and Dice pools exact confusion matrices; foreground Dice averages classes 1 and 2. All metrics use the same before cache and support for every candidate. Validation GT enters evaluator strata/metrics only, never guards or projection. Unsupported modes remain explicit. A near-zero direction would be invalid and fail signal retention while still consuming its registered identity-diagnostic forwards.

## G1–G7 adjudication

| Gate | Subcondition | Frozen requirement | Observed | Pass |
| --- | --- | --- | --- | --- |
| G1 | Foreground centers active | 24/24 | 24 | True |
| G1 | Minimum active pixels per foreground mode | >= 32 | 14598 | True |
| G1 | Minimum cases per foreground mode | >= 2 | 10 | True |
| G1 | Minimum foreground occupancy | >= 0.05 | 0.181483350991 | True |
| G1 | Active foreground KD guards | >= 18 | 22 | True |
| G1 | Finite/null/convergence rules | all pass | True | True |
| G2 | Minimum P4 raw guard dot | >= -1e-10 | -2.70006239589e-13 | True |
| G2 | Undefined required guard comparisons | <= 0 | 0 | True |
| G3 | Previous CE comparison defined vs P0 | 48/48 | True | True |
| G3 | Previous CE better fraction vs P0 | >= 0.6 | 0.6875 | True |
| G3 | Median previous CE improvement vs P0 | >= 0.0001 | 0.000401185004795 | True |
| G3 | Previous CE comparison defined vs P1 | 48/48 | True | True |
| G3 | Previous CE better fraction vs P1 | >= 0.55 | 0.6875 | True |
| G3 | Median previous CE improvement vs P1 | >= 5e-05 | 0.000401185004795 | True |
| G3 | Previous class 1 median CE worsening vs P0 | <= 5e-05 | -0.00911585467527 | True |
| G3 | Previous class 2 median CE worsening vs P0 | <= 5e-05 | -0.00204691956685 | True |
| G3 | Previous foreground Dice median drop vs P0 | <= 0.002 | -0.00120914991003 | True |
| G4 | Current CE median worsening vs P0 | <= 0.0001 | 7.1319476539e-05 | True |
| G4 | Current foreground Dice median drop vs P0 | <= 0.002 | -7.54517512317e-05 | True |
| G4 | Any pair/foreground class Dice drop vs P0 | <= 0.003 | 0.00345029148374 | False |
| G5 | Worst-mode comparisons defined | 48/48 | True | True |
| G5 | Worst-mode previous CE better fraction vs P2 | >= 0.6 | 0.604166666667 | True |
| G5 | Median worst-mode previous CE improvement vs P2 | >= 5e-05 | 0.000340996212396 | True |
| G5 | Positive seed-transition unit medians vs P2 | >= 4 | 4 | True |
| G5 | Any pair/supported foreground mode CE worsening vs P2 | <= 0.0001 | 0.036544748892 | False |
| G6 | P4/P0 median norm ratio | >= 0.5 | 0.688481313096 | True |
| G6 | P4/P0 p10 norm ratio | >= 0.2 | 0.43555103848 | True |
| G6 | Stage 1 median norm ratio | >= 0.4 | 0.744031993113 | True |
| G6 | Stage 2 median norm ratio | >= 0.4 | 0.553382805242 | True |
| G6 | Zero P4 directions | <= 0 | 0 | True |
| G6 | Norm ratios finite | all finite | True | True |
| G7 | Model/checkpoint/bank, gradient and GT isolation | all pass | True | True |

The better fractions use all 48 pairs. For loss metrics, improvement is control delta minus P4 delta; delta is after minus before. Dice drop is control Dice delta minus P4 Dice delta. G4's class bound and G5's new mode worsening bound are pointwise maxima, not means. G5's worst-mode value is the maximum supported foreground mode CE delta; a positive unit requires the median of its eight improvements to be strictly positive. No denominator, threshold or precision rule was changed after any result preview.

| Unit | Pairs | Median P4/P0 norm | Median P4-vs-P2 worst-mode CE improvement |
| --- | --- | --- | --- |
| seed0_stage1 | 8 | 0.658900503066 | -0.000855411795887 |
| seed0_stage2 | 8 | 0.456197798379 | 4.15018174831e-05 |
| seed1_stage1 | 8 | 0.727640675058 | 0.000551098646486 |
| seed1_stage2 | 8 | 0.66862958627 | 0.0073031412338 |
| seed2_stage1 | 8 | 0.813640057984 | 0.00111847981207 |
| seed2_stage2 | 8 | 0.55266830578 | -0.000177447481595 |

## Descriptive controls

| Candidate | Median previous CE delta | Median previous FG Dice delta | Median current CE delta | Median current FG Dice delta |
| --- | --- | --- | --- | --- |
| P0 | 0.00096063774812 | -0.00153577516425 | 7.76781999643e-05 | 6.28041251884e-05 |
| P1 | 0.00096063774812 | -0.00153577516425 | 7.76781999643e-05 | 6.28041251884e-05 |
| P2 | -5.2839986047e-05 | -0.000217638760926 | 8.64612367695e-05 | 0.000142928738508 |
| P3 | 0.000299042334821 | -0.000188785267362 | 0.000228340708343 | 0.000178572641016 |
| P4 | 5.29495223789e-05 | -0.000188385111616 | 0.000109875783219 | 0.000146309245404 |

These are medians of candidate deltas. Gate comparisons instead use medians of paired differences; the two operations are not interchangeable. Full per-class, per-mode, old KL, supervised-mode CE, support, solver and retention rows are published in the eight required CSVs. The additional six-row retention table is derived solely from all 48 immutable P4 rows; its hashes and aggregator are recorded in `PMGC_REPORT_DERIVATIONS.json`. It changes no gate.

G3 is relative utility, not a claim that forgetting disappeared: P4's median previous-domain CE delta remains positive (0.0000529495), and its median previous foreground Dice delta remains negative (-0.000188385), relative to the frozen before model.

## Isolation, archive and limits

`model_optimizer_steps=0`, `transport_optimizer_steps=0`, `method_registered=false`, `training_launched=false`. Current/old/teacher models and historical checkpoints/banks stayed bitwise unchanged. Teacher/bank gradients and parameter.grad fields stayed None. No backward, optimizer, EMA/GAS/PAS/historical K2 update, hidden training GT or test GT use occurred. The existing environment was reused without installing or replacing Torch.

The independent artifact pass separately recomputed mode assignment/support/UID hashes, full-gradient KKT certificates, fixed denominators and pooled metric arithmetic from saved artifacts. It did not rerun predictions or request new gradients, and it is **not independent scientific peer review**. This one-step, fixed-panel diagnostic does not establish training performance, generalization to other domains, or long-horizon forgetting.

Both historical private bundles were reverified file by file after the run: Gate1C 14470 files /17712127650 bytes, content `8a82c7b8f0c72eb4faf619f51d7c1eae67a5f81059bc7f283b6b8df22d563526`; MMPR 22224 files /4813748985 bytes, content `9a7312fef30b0787bf4b8bd89bb2646eba7e5e542186471798d194eb6f24201c`. All nine checkpoints, baseline/K2 manifests, used image/train-label/val-label HDF5 assets and the 2962-entry checksum inventory were rehashed. Unused hidden/test HDF5 labels were not opened for this audit. Protected inputs and HOME compatibility symlinks remain in place.

Complete new evidence: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/evidence_bundle`. Verified integrity copy: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/archives/0ce207953da89541d251ebe984411798aa7ced5fd3f08cb3a91d0ec50f84fd2e`. Payload: **22,170 files / 2,900,789,163 bytes**. Content SHA256 `0ce207953da89541d251ebe984411798aa7ced5fd3f08cb3a91d0ec50f84fd2e`; manifest SHA256 `6fcf4caa19122e9f8a540e138aec0e6fb8ce302d430c75b3f752c5c7a832a2a5`. Every file, byte hash and exact path coverage was verified before atomic promotion; the retained source was reverified afterwards. The archive uses separate regular files on the **same NAS**, not an independent-device disaster backup. Source evidence and all failures are preserved. Old bundles are hash-bound rather than recopied. Archive-operation/publication receipts are external to the sealed runtime bundle to avoid circular hashing and are indexed by the public manifest/receipt.

All large arrays, logs, scratch, test fixtures, caches and archives remain on NAS. Git receives only code and public text diagnostics/receipts, never checkpoint, image or gradient payloads. `PMGC_EXACT_COMMANDS.md` and `PMGC_PHASE_RECEIPTS.json` record the create-only execution; these completed commands must not be rerun.

Four public text copies replace 156 embedded prototype/center coordinate arrays with hashes and private source references. The full coordinates remain in the unchanged NAS reports/archive. `PMGC_PUBLIC_REDACTION_AUDIT.json` binds original and public file hashes and verifies that every other JSON field or CSV cell, including all metrics, supports and gate decisions, is unchanged. Public status/manifest/virtual-step JSON and the mode-support CSV are therefore explicitly sanitized representations, not byte-identical copies of the private originals.

## Study closure and later automation

Current prototype-derived new-method line state: **ENDED**. This fixed PMGC study is closed for independent review after report/receipt publication. No PMGC training, C0 regeneration, Gate2, Prostate, MnMS, sweep, main merge or prototype ranking/relation/transport/memory/gradient variant is started or authorized by this report.

The later explicit user instruction enables the existing current-task heartbeat every **30 minutes**. It does not change this completed protocol or rescue a failure. Subsequent work is limited to separately and prospectively registered finite strong-baseline/external-method comparisons within the existing Fundus/GPU4-7/NAS scope. Success must be defined before those experiments; automatic follow-up does not guarantee a successful research result. No follow-on training was launched during this PMGC execution.

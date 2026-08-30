# Gate1C v2.2 diagnostic-precision pilot

Recorded 2026-08-31 Asia/Shanghai. **PASS_NUMERIC_PRECISION_PILOT**;
execution completed 2026-08-30T17:26:30.873403+00:00. This is a three-pair
engineering result, **not** Gate1C scientific admission or method reproduction.
The original v2.1 attempt remains `BLOCKED_INCOMPLETE_EVIDENCE`; Gate1B and the
original overall Gate1 remain `FAIL_TRANSPORT_NOT_SUPPORTED`.

## Identity and scope

- Prospective numeric preregistration: `6357317749b0ff904e3acd39023b86430d6263ee`.
- Exact published execution/test code: `7fdd4312278eb64dbfb471107bb47e6b897c6859`.
- [Preregistration](DI_DMPA_GATE1C_V22_PRECISION_PREREGISTRATION.md) and
  [machine-readable contract](DI_DMPA_GATE1C_V22_PRECISION_PREREGISTRATION.json).
- Preregistration MD SHA256:
  `2ec5ba23d8136bbb3870776345a0be229e6071ba9a1e58604884dda42b2a433a`.
- Preregistration JSON SHA256:
  `2ceb37fc571b17373261fe631c8a2e416130912e2e882461b9e42795d495aeca`.
- Input contract stays v2.1, including its one explicitly reconstructed legacy
  PAS bank. The reconstruction remains
  `RECONSTRUCTION_SUPPORTED_NOT_HISTORICAL_HASH_VERIFIED`; its historical bank
  hash is still unavailable. The earlier 400 baseline-recovery updates are
  unchanged historical work, not updates made by this pilot.

Only the preregistered pairs ran: seed0/stage0/REFUGE/pair00 and
seed1/stage1/RIM_ONE_r3/pair00 on GPU0; seed0/stage2/Drishti_GS/pair01 on GPU1.
These cover three stages but **only two seeds**, not a complete three-seed panel.
All four phase barriers required all three pairs. There was no candidate,
panel, seed, case or device selection after observing the results.

## Minimal numerical repair

The shared execution/gradient engine is reused. Original FP32 student and EMA
forwards, probabilities, features, stochastic Gaussian returns, PAS, reliability
weights, fixed cases and class strata remain native. Only an isolated copy of the
student receives FP64 diagnostic gradients, replaying exactly the captured FP32
Gaussian values cast to FP64 without advancing RNG. Teacher/PoE gradient targets
first undergo the original FP32 conversion; PoE reporting arrays keep their
original precision and formula. Every FP64 objective has a same-input native
FP32 total-gradient control.

The original `atol=1e-6, rtol=1e-4` decomposition guard is unchanged. The default
native path retains that guard and exactly matches frozen commit `44a2525` on
synthetic full-phase output comparisons. Exception paths now retain bank and
model isolation receipts. Shared replay/RNG/comparison helpers were moved from
the prior reference script without changing their calculations. No new training
loop, dependency, scientific loss, classifier, optimizer or augmentation was added.
Official tracked JASCL source remains at `3c93ca7`; its 3x3 classifier is unchanged.

## Tests and complete coverage

The clean exact-code checkout passed **133 tests, 0 failures, 0 errors, 0 skips**
in 13.41 seconds. The JUnit properties bind the full execution SHA above and
`source_clean=true`; prepare refused unbound/dirty test evidence.
[Exact-code JUnit](gate1c_precision_pilot_results/6357317/tests/gate1c_precision_exact_7fdd431.xml)
SHA256: `1a27af9d98a6d7b7407826399aec9fe429b33233f33ff6fb46e9561794fb453a`.

The seven new checks cover same-Gaussian/state/RNG isolation, canonical FP32
PoE ties, all four phases and exact call counts, native-vs-frozen parity,
missing/duplicate evidence rejection, exception audit retention and zero-gradient
semantics. Earlier development evidence is preserved in the
[development receipt](gate1c_precision_pilot_results/6357317/DEVELOPMENT_TEST_RECEIPT.json):
one full-suite attempt had 132 passes and a Unicode decoding failure because
six macOS AppleDouble transfer sidecars matched the unchanged AST audit glob.
Those six files were moved intact outside the development checkout; no test
was weakened. The next development run and the clean published-code run passed.

One operator limitation was identified by post-run source inspection, without
replaying a command: execution code `7fdd431`'s generic exception logger can append a
failure marker when a completed non-worker command is mistakenly repeated.
No such repeat occurred and the sealed manifest remains intact. Use the JSON
receipts/read-only auditor for status, never the completed prepare/barrier/report
commands. The separate post-run refusal-path repair was published as
`d6bd0707d7a889d7e42e00e61b3d242354335753`, then passed **135/135** tests in its
clean exact-code checkout (15.47 seconds), including no-mutation reentry and
genuine-failure retention checks. See the [development receipt](gate1c_precision_pilot_results/6357317/OPERATOR_REPAIR_DEVELOPMENT_RECEIPT.json)
and [exact-code test receipt](gate1c_precision_pilot_results/6357317/OPERATOR_REPAIR_EXACT_TEST_RECEIPT.json).
The original execution checkout and numeric engine are unchanged; the new
operator code was not used to rerun or relabel the completed pilot.

| Phase | Alignment rows | Global objective comparisons | Class-component rows | Supervised global comparisons |
| --- | ---: | ---: | ---: | ---: |
| draw0 | 168 | 24 | 504 | 3 |
| noise, including cached draw0 | 1,344 | 192 | 0 | 3 |
| posterior mean | 168 | 24 | 0 | 3 |
| PoE | 336 | 48 | 126 | 3 |
| Total | 2,016 | 288 | 630 | 12 |

Observed real computation exactly matched registration: **51 native FP32 +
24 shadow FP64 forwards = 75**; **276 native + 366 FP64 autograd.grad calls**.
No validation forwards, backward calls, parameter.grad writes, optimizer/EMA/GAS
or prototype updates occurred. All method switches and `method_registered`
remain false. Hidden-GT training usage and final-test GT usage are `none`.

All 30 native scoring calls matched frozen Gate0 PAS/R1 pixelwise, covering
8,847,360 pixel-call comparisons. Source student, EMA and shadow state stayed
bitwise unchanged in all 12 model guards; legacy/current/history bank and
gradient-isolation checks passed. All nine original B0 checkpoint hashes stayed
unchanged. Seed1/stage1 exercised the reconstructed-input path without replacing
the original checkpoint.

## Numerical findings

Every registered global objective and supervised comparison met relative
L2 <= 1e-3 and cosine >= 0.9999, with no averaging or favorable-subset rescue.

- Maximum objective relative L2: **0.00012594988782098875**.
- Minimum objective cosine: **0.999999992160753**.
- Maximum supervised relative L2: **4.2237250813564015e-6**.
- Minimum supervised cosine: **0.9999999999932037**.
- Maximum class-component sum absolute residual: **4.440892098500626e-15**.
- Both-zero global objective comparisons: **0**. Zero alignment would remain
  null and could not earn scientific admission credit.
- Descriptive blockwise comparisons also had zero comparability failures;
  their maximum relative L2 was 0.00021758896377494543. Only the registered
  global criteria determine this precision-pilot acceptance.

All native forward/PAS and eight draw0 total-gradient hashes matched each of
the two original golden pair receipts. The previously failing pair matched its
native probability/target/weight/stratum and R2/class-balanced gradient hashes.
Its FP64 R2/class-balanced global vector matched the independently published
reference hash `ef7a732d7fc57d827ad591118d951e239914ad2fd76f1f47ad8142e276bf85f0`.

This supports the diagnosed FP32 network-VJP summation issue and the bounded
same-draw FP64 diagnostic path. It does **not** establish all-72-pair robustness,
reliability benefit, transport benefit or segmentation performance. All three
pilot draw0 EMA maps had zero null pixels, so null handling is covered here by
the synthetic regressions, not by new real null-pixel examples. No C1-C8
scientific verdict, feature-source/K reselection or method training was performed.

## Artifacts, resources and retention

- Clean execution checkout: `/root/SSL_CL_precision_7fdd431`.
- Sealed remote output:
  `/root/LCRSeg/runs/gate1c_v22_precision_pilot/6357317749b0ff904e3acd39023b86430d6263ee/attempt1`.
- [Pilot status](gate1c_precision_pilot_results/6357317/attempt1/PILOT_STATUS.json)
  SHA256: `0e6c3cbe66e460e411a5200fc191f4984223bf848bb1400e6bc943015d3911b0`.
- [Artifact manifest](gate1c_precision_pilot_results/6357317/attempt1/PILOT_ARTIFACT_MANIFEST.json)
  SHA256: `de3969652de90494fc9b7fbd7f0745cd6797cd6088d62d3a12318a909382866c`.
- Independent remote audit: **66 files, 122,825,857 bytes**, including the
  manifest itself. All hashes, coverage, phase receipts and registered global
  comparisons passed. [Audit receipt](gate1c_precision_pilot_results/6357317/REMOTE_ARTIFACT_AUDIT.json)
  and [read-only auditor](gate1c_precision_pilot_results/6357317/audit_pilot.py).
- Published runtime evidence contains **60 JSON files / 4,644,161 bytes**;
  all six raw NPZ files are excluded. [Publication audit](gate1c_precision_pilot_results/6357317/PUBLIC_METADATA_AUDIT.json).
- The private local full archive is complete at
  `/Users/bominwang/Desktop/codes/SSL_CL_seg/runs/gate1c_v22_precision_pilot/6357317749b0ff904e3acd39023b86430d6263ee/attempt1`.
  SCP exited zero, then the independent local audit passed at
  **2026-08-30T17:53:11.791970+00:00**: all **66 files / 122,825,857 bytes**
  match the sealed manifest. [Local archive audit](gate1c_precision_pilot_results/6357317/LOCAL_ARCHIVE_AUDIT.json)
  matches every remote-audit field except the expected root and audit timestamp.
  The existing root `/runs/` ignore rule was verified; raw NPZ files remain
  private. The first report correctly marked this transfer pending, and one
  premature audit failed because the final manifest had not arrived. No remote
  evidence was changed or replaced.

GPU0/GPU1 worker durations in seconds were respectively 15.879/8.554 (draw0),
36.314/20.208 (noise), 7.351/4.471 (posterior), and 20.398/11.941 (PoE), all below
the registered ten-minute per-worker/phase limit. Both GPUs ran concurrently;
one noise-phase sample at 17:22:53 UTC measured 69%/53% utilization and
2,998/2,996 MiB. This is a point sample, not sustained-max-utilization evidence.
Independent phases waited for their registered barriers. The 17:29:19 UTC
post-run check found no pilot/formal worker and both GPUs at 0%/1 MiB.
The final 17:51:54 UTC source/resource check again found no diagnostic worker
or CUDA compute process, both GPUs at 0%/1 MiB, and both exact-code checkouts
clean. Available storage was 9,225,658,368 bytes on `/root/LCRSeg` and
209,936,281,600 bytes on `/tmp`; these are capacity observations, not durability
guarantees.
No paid resource, environment, global network, mount or permission change occurred.
The pilot did not use the separate `/tmp` mount.

## Next finite iteration, not a full-run authorization

1. The private local archive prerequisite is complete and independently verified.
   Preserve all old failed attempts and the sealed pilot; never rerun occupied
   pilot workers.
2. Publish a separate prospective **full v2.2 diagnostic execution/retention
   plan** before any new real forward. Keep the original 72 pairs, three seeds,
   native B0-EMA scoring, K=2 identity history, all C1-C8 conditions and zero
   model updates. Retain the native FP32 path and its old failed verdict, and
   retain the separately tested operator refusal guards described above.
3. First assess read-only reuse of the already complete 495-case v2.1 native
   validation caches: verify every hash and exact code/formula/input lineage;
   never relabel old forwards as newly executed v2.2 evidence. This is a
   proposed storage/runtime optimization, not approved reuse in this pilot.
4. Record explicit capacity and retention bounds, clean-code tests and a
   two-GPU schedule in that new version. The full gradient cache raw payload
   bound is 3,906,994,176 bytes before JSON/container overhead; existing native
   validation caches occupy about 4.86 GB. Do not duplicate them blindly into
   the roughly 9.23 GB free root volume or treat `/tmp` as the only evidence copy.
5. Only complete, versioned evidence may be evaluated against unchanged C gates.
   A C result cannot rescue frozen Gate1B. Before any later performance run,
   preregister a comparable reference, the UNet medical-adaptation distinction,
   main metric, acceptable gap, seeds and finite budget. No new method or sweep
   is started by this report.

The current-method-only long-running goal remains active. B0/C0 stay controls,
not additional methods. The same-thread follow-up remains the existing task,
not a second scheduler; local scheduled work requires the computer and app to
stay running ([official OpenAI documentation](https://learn.chatgpt.com/docs/automations?surface=app)).

## Exact executed commands

[Fully resolved command record](gate1c_precision_pilot_results/6357317/EXACT_COMMANDS.txt)
contains every phase/GPU invocation without placeholders. The forms below
explain the common environment and ordering; none should be replayed on attempt1.

On the purchased node, in `/root/SSL_CL_precision_7fdd431/experiments/lcrseg`,
every invocation used the existing interpreter and this environment prefix:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
LD_LIBRARY_PATH=/lib/x86_64-linux-gnu CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. /root/.venvs/lcrseg-py310/bin/python
```

Append the following arguments for the exact-code test and prepare calls:

```sh
-m pytest -q tests/di_dmpa_gate1c_v2 --ignore=tests/di_dmpa_gate1c_v2/test_real.py --junitxml=/root/gate1c_precision_exact_7fdd431.xml
-m di_dmpa_gate1c_v2.precision_pilot prepare --code-commit 7fdd4312278eb64dbfb471107bb47e6b897c6859 --tests /root/gate1c_precision_exact_7fdd431.xml
```

The phase order was `draw0`, `noise`, `posterior`, `poe`. In each phase, the
following worker arguments ran concurrently for gpu=0 and gpu=1, with the
environment's `CUDA_VISIBLE_DEVICES` set to that same single GPU. Both worker
exit codes were zero before the barrier (which used visible GPUs 0,1):

```sh
-m di_dmpa_gate1c_v2.precision_pilot worker --code-commit 7fdd4312278eb64dbfb471107bb47e6b897c6859 --phase PHASE --gpu GPU
-m di_dmpa_gate1c_v2.precision_pilot barrier --code-commit 7fdd4312278eb64dbfb471107bb47e6b897c6859 --phase PHASE
-m di_dmpa_gate1c_v2.precision_pilot report --code-commit 7fdd4312278eb64dbfb471107bb47e6b897c6859
```

Each literal worker argv is retained in its `WORKER_*_START.json`; the report
command ran once after the final barrier. These are historical commands, **not
instructions to repeat occupied attempt1**. The post-run independent audit was:

```sh
/root/.venvs/lcrseg-py310/bin/python /root/audit_gate1c_precision_7fdd431.py /root/LCRSeg/runs/gate1c_v22_precision_pilot/6357317749b0ff904e3acd39023b86430d6263ee/attempt1 /root/SSL_CL_precision_7fdd431/experiments/lcrseg/docs/di_dmpa_jascl/DI_DMPA_GATE1C_V22_PRECISION_PREREGISTRATION.json
```

The audit script SHA256 is
`8cbba7840e9558cffbe15d73ccd632076b5cd44444d87fd27884600e1bdb0506`.
Source/report publication is limited to `codex/sslcl-long-running-reproduction`;
the original diagnostic branch and main remain untouched.

# Gate1C v2.2 execution progress

## Recovery update: 2026-08-31

SSH access has recovered. The [live byte/metadata precheck](gate1c_v22_preflight/LIVE_RECOVERY_PREFLIGHT.json)
passed with the original failure and pilot artifacts intact; its initial
overbroad bytecode-clean assertion and correction are both retained. The
[independent conditional authorization](GATE1C_V22_EXECUTION_AUTHORIZATION.md)
is now recorded for publication before implementation. **NOT_LAUNCH_READY**:
no new full runner, exact-code synthetic suite, cache numerical audit or new
real integration is complete. Zero new tensor/array loads, forwards or updates.
The earlier connection-blocker account below is historical, not current.

Recorded 2026-08-31 Asia/Shanghai. **NOT_LAUNCH_READY**. This is not a scientific
Gate1C verdict. No new checkpoint tensor, cache array or GT has been loaded,
and no new model forward, optimizer update or formal diagnostic has run in
this preflight.

The preceding numerical pilot and its private local archive remain sealed;
see [its report](GATE1C_V22_PRECISION_PILOT_REPORT.md). This next version is a
[prospective full-diagnostic plan](DI_DMPA_GATE1C_V22_EXECUTION_PREREGISTRATION.md),
not an assertion that cache reuse, launch readiness or method reproduction has
been established.

## Completed local evidence

The stdlib-only auditor was published as
`16eb21343781cbc3556e046c3bd3a0f3c03896cf`. Its clean-code run verified:

- unchanged native validation/cache/evaluator definitions against formal code
  `44a2525`, plus unchanged protected data/model/PAS/metric sources;
- byte-identical numeric `execution.py`, `gradients.py`, `precision.py` against
  tested pilot code `7fdd431`;
- the published frozen inventory's exact 495 cases, nine units,
  72,990,720 pixels and 4,856,574,421 cache bytes.

[Source receipt](gate1c_v22_preflight/SOURCE_AUDIT.json).
Three stdlib tests passed, covering formatting-only changes and rejection of
changed/missing/duplicate definitions or cache fields.
[Test receipt](gate1c_v22_preflight/SOURCE_TEST_RECEIPT.json).
This is **not** a new full Torch test-suite pass. The local default Python has
no Torch/NumPy/pytest; no package or environment was installed.

## Connection blocker and limits

Three SSH attempts to the existing endpoint were closed. The verbose attempt
showed a TCP connection followed by remote close before an
SSH server version/key exchange. A fourth read-only attempt bound to the
existing physical interface returned connection refused. No network/proxy or
SSH configuration was changed. The original Jupyter URL redirected to the
platform login page; the available browser had no authenticated session.
A final SSH recheck after the local preflight also exited 255 with remote close.
[Connection receipt](gate1c_v22_preflight/CONNECTION_PRECHECK.json).

No remote command succeeded during this preflight, so current instance,
process/GPU and disk state are **unknown**, not inferred stopped. The last
successful remote observation at 2026-08-30T17:51:54Z is historical only.
This does not prove lost storage, invalid keys or a new experiment failure.
The user was asked to confirm instance state and the current platform SSH
command. No instance, service or worker was restarted.

The local source/inventory audit cannot stand in for live cache hashes,
numeric array checks, original metadata/CP/bank proofs or resource checks.
Cache reuse is still unapproved. A separate execution authorization and exact
new-code synthetic/real integration evidence remain required. The proposed
36-pair-per-GPU formal schedule has not started.

## Exact completed commands

From `/Users/bominwang/Desktop/codes/SSL_CL_seg`:

```sh
python3 -m unittest discover -s experiments/lcrseg/tests/di_dmpa_gate1c_v2 -p test_source_preflight.py -v
python3 experiments/lcrseg/scripts/audit_gate1c_v22_sources.py
ssh -vv -o BatchMode=yes -o ConnectTimeout=12 -o ConnectionAttempts=1 -p 31192 root@162.14.139.38 true
```

The first two completed successfully on clean preflight code. The SSH command
exited 255 before its remote `true` command could run. Private local interface
addresses and public-key debug details are omitted from this public report.

Next: restore access to the same purchased instance, independently verify all
referenced caches/input files and capacity, publish the separate execution
authorization, then implement/test the minimal shared-runner extension and new
real integration before any full diagnostic. Keep the original C/B failures,
all old artifacts, all scientific conditions and the current-method-only scope.

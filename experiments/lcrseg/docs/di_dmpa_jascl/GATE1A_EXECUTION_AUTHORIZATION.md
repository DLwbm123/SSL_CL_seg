# Gate 1A execution authorization

Scope: **GATE1A_ONLY**. Recorded 2026-08-30T14:08:03+08:00, Asia/Shanghai.
The user authorized this execution after preregistration publication.

- Preregistration ID: `DI_DMPA_GATE1_V1_B0_EMA_PRIMARY`
- Preregistration commit: `cfb62554f1e6a2a36850547485b1857dc9a28a20`
- Preregistration Markdown raw SHA-256: `32acdc5c24bcc5763daa6cb3650fea91f46da7ae3845b1fd0615c781619fbf0a`
- Preregistration JSON raw SHA-256: `6f50bd9df404d987aa70e2035a5c3f3853aa59ce49d21ffface34172cf754cbf`
- Parent prework: `39532af4898bd1ae13c76033c686ed7479389ae8`
- Branch: `codex/di-dmpa-gate1-diagnostics`
- User request raw SHA-256: `865e2b4eafa06fa74b75dc6f7e7ebd51d54a4c5139843fe24c55f87bd598d2a6`

The companion JSON is this turn's authorization record, not an amendment.
Both preregistration files remain byte-for-byte immutable, including their
historical `current_turn_authorizes_diagnostics=false`.

Allowed: shared read-only provenance/sampling/features/geometry infrastructure;
synthetic/unit tests; one read-only real-checkpoint/fixed-coordinate integration
test without mechanism inference; four-panel formal Gate 1A; reports, hashes,
commit and push. B0-EMA alone determines admission/K. All four panels must
complete before adjudication, unless a protocol/numerical hard stop occurs.

Gate 1B, Gate 1C, Gate 2, transport fitting, reliability, formal gradient-conflict,
formal teacher-noise, final theory quantities, training, Prostate, MnMS,
full sweep and main merge are **false/not authorized**.

All training method flags remain false. Model and transport optimizer steps
remain zero. Do not construct a segmentation optimizer or update student,
EMA, classifier, GAS, buffers or checkpoint state.

Push this separate authorization commit and verify its remote SHA before
implementation. Record that exact commit as `authorization_git_commit`;
never call it the preregistration commit. Commit/push/verify diagnostic source
before formal execution. Preserve failures and unique attempt namespaces.
Run metadata must bind preregistration, authorization, exact code, file hashes,
shared sampling hash and inputs; report source identity is recorded separately.

Next action after Gate 1A: **STOP_FOR_INDEPENDENT_REVIEW**.

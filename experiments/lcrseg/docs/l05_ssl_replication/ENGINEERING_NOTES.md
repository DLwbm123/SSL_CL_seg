# Scope and implementation evidence

The new adapter reuses the frozen V0.1 stage loop, HDF5 accessor, objectives, evaluation and deployment. Only the L05_SSL step and optimization seed binding are new. Frozen predecessor files are not changed.

The three flags use_unlabeled_images/use_pseudo_ce/use_unlabeled_kd are separate. All authorized recipes set U-KD false. The step never computes teacher-U logits or a U-KD target/loss. Zero-valued U-KD log/probe placeholders are not a KD branch. A guarded teacher-U entry and actual teacher input sentinel are covered in qualification. No SCD projector is called.

Optimization seed replaces the leading zero of all original initialization, classifier, order, geometry and strong-augmentation streams, with arm excluded. Seeds1/2 preload the official module before initialization, because its import-time code otherwise resets the seed. Seed0 delegates the original key mapping and original initialization behavior. P1 uses the verified historical seed0 common weights; seed0 common is not retrained.

Python 3.12 changed builtin floating sum. The new score adapter explicitly preserves the frozen Python 3.10 left-associated F addition. This repairs local full-precision parity without changing metrics, thresholds, data, or training. The first development fixture failure and its 126 synthetic updates are retained; the second development suite passed with another 126 updates.

Qualification runs three unchanged inherited tests (scoring, role/path isolation, PAS/geometry) and focused new production-chain tests. Old solver tests and old namespace whitelist tests are neither changed nor claimed. No new real-data overfit or diagnostic optimizer update is introduced. Exact-source local/server tests count successful synthetic updates separately. Real runs are not initialized from fixtures.

S has one complete model; L05/L05_SSL have two. Label-only checkpoint schemas retain the inert 195-byte zero prototype/support placeholder, without prototype estimation. GAS always uses supervised CE only. L05_SSL retains the original epoch96 final prototype estimate; there is no post-hoc replacement. Optional extra val PAS precision diagnostics are not run; precision remains NOT_EVALUATED, and coverage is not precision.

The durable background parent owns a finite P1/P2/P3 dependency graph. P2 does not depend on P1 science. A sealed P_GATE alone admits P3 without further chat confirmation. GPU memory waits are part of this authorized execution queue, not a separately scheduled monitor. Later user status requests will inspect the actual run and complete public closeout when terminal. No ongoing Codex polling is required.

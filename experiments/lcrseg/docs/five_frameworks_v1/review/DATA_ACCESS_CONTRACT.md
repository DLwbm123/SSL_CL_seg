# Data access and authorization

This task read repository/documentation/config metadata, public author source text and generated CPU toy tensors only. Real image reads = 0; real label reads = 0; real checkpoint tensor reads = 0; real smoke updates = 0; formal optimizer updates = 0. CPU synthetic optimizer updates are tested and are not counted as formal updates. No GPU/SSH jobs or monitoring were started in this task. Existing earlier experiment processes were not queried or modified.

The synthetic provider has separate labeled and unlabeled methods. The U return type has images, geometry and source identity only; there is no U-label accessor. LCTX validates two different synthetic patients. U methods deduplicate source IDs before exposure; no-U baselines never access U. `DataScope` refuses old/future domain, validation, test and hidden-U-GT categories. This is a code contract, not evidence of a verified real loader.

A future real trainer may receive only current-domain L image/GT and current-domain U image/geometry capabilities after external review and explicit user confirmation. Original real parent/source provenance, dataset manifest permissions, geometry and source-seed binding must be resolved first. Metadata existence is never reported as tensor or byte-level validation.

Evaluation must be a separate student-only capability. Final selection occurs offline after complete trajectories, over development validation patients. It is not online adaptation using old scores and not an independent-patient evaluation. No val/test scores or patient data are present here. Aggregate helpers accept synthetic numeric arrays; no real evaluator loader has been bound.

Approval cannot be manufactured by the code. Required bindings: exact reviewed commit/tree, parent binding, expanded plan, dependency lock; independent caps for target updates, source updates, real-L smoke, VJPs, readout forwards; B/C/D scope; non-template external review evidence; explicit user launch confirmation. `require_approval` tests each field. Because this package has no real parent, `cli run` always rejects even before reading the approval path, and no data loader is reachable.

Public artifacts contain no private runtime paths, keys, patient IDs, images, labels, models or per-patient metrics. User-provided review text and reference reports remain separately labeled as input evidence, not newly verified experiment results.

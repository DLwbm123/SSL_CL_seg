# SHOR-JASCL V0.3 execution authorization

## Binding

This authorization binds preregistration commit `6eaf8b8a299a47dec7a296ef2d784a105a53ab55` and registration ID `SHOR_JASCL_V0_3_SELECTIVE_HISTORICAL_OVERRIDE`.

Preregistration file SHA256 values:

- `SHOR_JASCL_V0_3_PREREGISTRATION.md`: `91673a97368a97d6395aa5e96cb9f2031c398ddbccf54916f491793639fc9273`
- `SHOR_JASCL_V0_3_PREREGISTRATION.json`: `c9c4bfcc1d4ade83c98a5b7171f29b4b927166dfa4adbf7ef9235cbc7e9534b7`

The base is the complete prior receipt `c854bd28b1a69ce001646201a824b8bb75141c67`; the V0.2.1 closure is `9feee43c5e34c427356ceaaafa6f691dd14186a3`.

## Authorized execution

After this document is committed, pushed, and remotely verified, it authorizes implementation and tests followed by exactly one create-only SHOR-JASCL V0.3 validation `formal_01` from exact published source.

The formal run may read the sealed V0.2.1 private bundle read-only, reconstruct train-only ridge OOF logits, select train-only historical-override thresholds, reuse sealed validation descriptors and expert probabilities, seal SHOR candidates, then perform evaluator-only validation evaluation and five fixed train-memory bootstraps. It may publish reports, archive evidence, and create a publication receipt.

The run must remain zero-model-forward. It does not authorize model construction, checkpoint tensor loading except hash audit, autograd, backward, optimizer or router-optimizer steps, parameter-gradient writes, training/fine-tuning, grid expansion, descriptor changes, validation-based threshold selection, test construction/evaluation, data regeneration, or any second attempt.

The source bundle and V0.2.1 execution remain immutable. Failure preserves partial evidence and ends without retry. Completion or scientific failure enters the preregistered hard stop for independent review.

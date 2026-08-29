# LCR-Seg V0.1 routing diagnostic

**Status:** Phase A complete; post-hoc evidence only; no V0.2 result claimed.  
**Protocol:** V0.2 preregistration, M1.  
**Execution date:** 2026-08-20.

## Boundary and input integrity

This diagnostic was run as a separate analysis process after the listed
checkpoints had already been frozen. It used
/home/jiangsuiyang/SSL_CL with the formal Python interpreter and GPU 4, and
wrote only beneath reports/analysis/v0_1_routing/. The frozen h5/v1,
manifests, splits, and checksums directories remained mode 500; no training,
optimizer step, model mutation, or frozen-input write occurred.

Hidden labels were read only from the separate diagnostics manifest in this
post-hoc process. They were used to score frozen predictions and define
interior/boundary regions, never to fit a training component, change a
threshold, or select a method. Component-size thresholds use only current
model predicted components over the corresponding current-site
train_unlabeled records.

## Checkpoint coverage

The audit processed all final/site checkpoints for these frozen LCR runs:

- fundus_seed0_lcrseg_v0_1_full200e;
- fundus_seed0_lcrseg_uniform_relation_kd_full200e; and
- fundus_seed0_lcrseg_no_learnability_full200e.

That is nine LCR checkpoints: REFUGE (160 records; 1,474,560 relation-grid
pixels), RIM-ONE-r3 (63; 580,608), and Drishti-GS (41; 377,856), for each
run. All nine were analyzed successfully. The three Sequential-SSL
checkpoints were also inventoried, but are deliberately marked
not_applicable_no_lcr_relation_state: they contain neither semantic-anchor
state nor old relation targets, so synthesizing L/C values for them would be
invalid.

The generated inventory has 12 checkpoint rows, 450 classwise calibration
rows, 90 regionwise rows, 60 ESS rows, and 9 golden-batch gradient rows.

## Artifacts and integrity hashes

All files below were generated remotely, then transferred without deletion and
matched byte-for-byte locally:

| Artifact | SHA-256 |
| --- | --- |
| checkpoint_inventory.csv | 43eaa508212d24d441c37d2604bb93830de90c05638fd485ee383e41c3752a91 |
| classwise_calibration.csv | 17862362bcea715d87c3eaa853b35f7e8f8e6ce8c0da5b9a02c915186912d15b |
| regionwise_calibration.csv | 6ea9d5451b5ae751aab53d1d24ec4f515aa909156dfdab1bd23b5c9300e259d3 |
| effective_sample_size.csv | 3c8260a81f6f90f753b47d5f57213bc431886c95e9ee0c3c86854c4d19d6da71 |
| gradient_diagnostics.csv | 578ff0c36b40e5510f0cf6771defedb0a89f26ea50727d4127e16582601ef9ec |
| routing_diagnostic_summary.json | 671562face3202d3870bb10e680b4a2c0b13363457fcbc224ae80bbeec30a92c |

The three PNG overviews are present and have nonempty SHA-256 records in the
same directory.

## Class-wise routing evidence

For the full V0.1 Drishti final checkpoint, the lowest-to-highest L bin
pseudo-label accuracies were:

| Relation-predicted class | Lowest L bin | Highest L bin |
| --- | ---: | ---: |
| 0 background | 0.6902 | 0.9960 |
| 1 disc rim | 0.5199 | 0.8497 |
| 2 optic cup | 0.6387 | 0.9741 |

Thus the L endpoints have the expected direction, but strict ten-bin
monotonicity is not universal: class 1 has one adjacent decline. This
supports an admission rule as a testable hypothesis, not as an already proven
guarantee for every class.

For raw C on that same checkpoint, lowest-to-highest-bin old-relation
correctness was 0.4009 to 0.9375 for class 0, 0.0309 to 0.6442 for class 1,
and 0.9066 to 0.9756 for class 2. Despite those endpoint increases, strict
ten-bin monotonicity fails for all three classes (5, 3, and 3 adjacent
declines respectively). Therefore the original continuous C weighting is not
sufficiently class-calibrated to justify treating a high raw C as a globally
reliable continuous multiplier.

## Spatial evidence

The full V0.1 Drishti final checkpoint shows the strongest routing difficulty
near GT-defined boundaries. For current relation classes 0/1/2 respectively,
pseudo-label accuracy is 0.4827/0.6098/0.5776 on boundary pixels versus
0.9681/0.7586/0.9188 in interiors. Current-old agreement is
0.9821/0.4494/0.0975 on those boundary groups versus
0.9975/0.5448/0.4629 in interiors. The small-component group is sparse and
is retained as an explicit diagnostic rather than being generalized from.

These results identify compositional and spatial variation as a plausible
reason that a global C scale can attenuate useful relation supervision. They
do not feed hidden labels or region labels into V0.2 training.

## ESS and gradient evidence

At the full V0.1 Drishti final checkpoint, continuous assimilation retains an
ESS of 88.90% of its valid pixels and continuous consolidation retains 92.59%.
The uniform-relation ablation has consolidation ESS 100.00%; the
no-learnability ablation has assimilation ESS 100.00%. Thus the historic
continuous routing changes the effective sample distribution even before
considering accuracy.

On the fixed deterministic current-site golden batches, full V0.1 has
assimilation/relation gradient norms of 0.17031/0.09787 with cosine 0.73133
at RIM-ONE-r3 and 0.70598/0.09154 with cosine 0.85977 at Drishti-GS. The
uniform-relation counterpart has 0.76847/1.40916 with cosine 0.18084 at
RIM-ONE-r3 and 0.41106/2.73038 with cosine 0.27055 at Drishti-GS. These are
descriptive diagnostics only; no gradient surgery or response to those values
has been introduced.

## Phase-A decision

Phase A is complete and its required artifacts are present. It preserves the
V0.1 conclusion: relation consolidation has signal, but the symmetric
continuous routing is not reliably calibrated across classes and regions.
The evidence is consistent with proceeding to the pre-registered V0.2
asymmetric test only:

1. class-wise progressive admission with unit selected-pixel loss; and
2. labeled-only, rejection-only compatibility routing with a nonzero floor.

It does not pass a V0.2 research gate, authorize any Prostate run, or justify
new unregistered modules or hyperparameter changes.

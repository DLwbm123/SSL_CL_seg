# SPARC-Seg V0.1 feasibility audit

**Final status:** `SPARC_PAS_NOT_SUPPORTED`  
**Optimizer steps:** `0`  
**Hidden-GT training usage:** `none`  
**Hidden-GT analysis usage:** `independent post-hoc only`

## Gate summary

| Gate | Passed | Key evidence |
|---|---:|---|
| A1 current PAS | `False` | Per-class details in JSON and `prototype_validation_quality.csv` |
| A2 stable/plastic | `False` | Per-class details in JSON and `partition_quality.csv` |
| A3 spatial coverage | `True` | 3-pixel processed boundary band, per-class ratios in JSON |
| B feature separation | `True` | median cosine gap `0.14204654722325222`; distance-order fraction `0.7916666666666666` |
| B gradient scale/localization | `True` | median `0.367363635541904`; p10 `0.08489559281742838`; p90 `1.3139548241491081` |
| C1 previous utility | `True` | paired-better fraction `0.859375` |
| C2 current safety | `True` | variant medians in JSON |
| C3/C4 targeting and complementarity | `False` | S3 versus S1/S2/S4/S5 medians in JSON |

## Protocol boundary

- Prototypes used all and only current-site `train_labeled` cases, with per-case normalization, equal case weighting, final normalization, and the frozen 32-cell minimum.
- The visible audit used 32 fixed clean current-unlabeled update batches plus 16 previous/current validation batches for each of six seed-transition pairs.
- Hidden labels were resolved only by separate post-hoc process invocations after visible artifacts were frozen; they were used only for mask/feature quality metrics.
- Uniform relation KD, R0 pseudo targets, frozen checkpoints, data, manifests, and splits were not modified.
- No optimizer step, SPARC method registration, or SPARC training configuration occurred.

## Consequence

Protocol hard stop before SPARC method/config implementation or training.

# ASPR-Seg V0.1 feasibility audit

**Status:** `ASPR_UNLABELED_MEMORY_NOT_SUPPORTED`  
**Hidden GT:** `post_hoc_only`  
**Optimizer steps:** `0`

## Gates

| Gate | Result | Key metrics |
|---|---|---|
| Reliable unlabeled memory | FAIL | `{"coverage_seed_pass_count_by_class": {"1": 3, "2": 3}, "fraction_delta_lu_nonnegative": 0.7777777777777778, "median_delta_lu": 0.011580109596252441, "p10_delta_lu": -0.004098266363143921, "precision_seed_pass_count_by_class": {"1": 0, "2": 0}, "site_class_seed_pairs": 18}` |
| Static prototype drift | PASS | `{"fraction_degradation_ge_0_050": 1.0, "median_static_cosine_degradation": 0.7890832424163818, "memory_source": "combined_labeled_plus_reliable_unlabeled", "pairs": 18}` |
| Evidence-adaptive transport | PASS | `{"fraction_shrinkage_gt_static": 1.0, "median_full_shift_oracle_cosine": 0.8327313363552094, "median_shrinkage_minus_static": 0.6918560229241848, "median_shrinkage_oracle_cosine": 0.8291383385658264, "memory_source": "combined_labeled_plus_reliable_unlabeled", "p10_shrinkage_minus_static": 0.3491638779640198, "pairs": 18}` |
| Site-mode utility | FAIL | `{"median_nearest_distance_reduction": 0.0693180709433151, "memory_source": "combined_labeled_plus_reliable_unlabeled", "minimum_historical_site_occupancy": 105, "minimum_max_site_minus_global_accuracy": -0.008570194244384766, "own_site_seed_pass_count_by_class": {"1": 1, "2": 1}, "seed_class_rows": 6}` |

## Protocol consequence

Protocol hard stop: do not implement or train ASPR V0.1. A new protocol is required to change the proposed method line.

All three seeds were reconstructed independently from frozen R0 checkpoints. Hidden diagnostic labels were read only by the audit script after reconstruction; they were not present in memory builders, calibrators, or training loaders.

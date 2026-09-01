# PRES-DSR-SF V0.2.1 failures and warnings

- Scientific status: `FAIL_SOFT_EXPERT_FUSION_VALUE`; this is not an engineering blocker.
- E1, E2, E3, E5, and E6 passed. E4 failed.
- Within E4, oracle gap, shared gain, historical gain, gain over M1-hard, and positive-seed count passed.
- E4 failed because maximum seed-domain drop was `0.03980300482442001` against a maximum of `0.020`, and current-domain drop was `0.03980300482442001` against a maximum of `0.010`.
- Clean controls cannot rescue the failed primary soft-fusion gate.
- All 14 stage barriers, 12 model guards, 9 checkpoint before/after checks, deterministic backend checks, and archive checks passed.
- Public artifacts omit private paths, case identifiers, descriptors, probabilities, masks, checkpoints, and raw CSV rows.
- No additional attempt or downstream experiment is authorized.

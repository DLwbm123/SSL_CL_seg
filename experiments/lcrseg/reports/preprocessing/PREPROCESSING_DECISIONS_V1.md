# LCR-Seg preprocessing decisions v1

This record implements the fixed choices authorized in
`CODEX_PROMPT_COMPLETE_LCRSEG_PREPROCESSING.md` and
`LCRSeg_NEXT_STEP_FAST_PREPROCESSING_PLAN.md`. Source directories remain
read-only; all derived payloads are written below `/Volumes/DataP/LCRSeg`.

## Frozen label maps

- Prostate whole-prostate binary: `0 -> 0`, `1 -> 1`, `2 -> 1`.
  Config SHA-256: `89487d472b77aa4802370a6f1cf8a20d69d4e9f83ffa09acd9105201ece1f26b`.
- M&Ms: `0 -> background`, `1 -> LV`, `2 -> MYO`, `3 -> RV`.
  Config SHA-256: `33433fe303d58b477e5c4ba7f943311b7bf7948c9516727dee42195249e52687`.
- Fundus: `255 -> background`, `128 -> optic disc rim`, `0 -> optic cup`.
  Config SHA-256: `a3ddfa9000ffb9d2364fe3a39b98ce19f34813c309d2908d58aa59d2e35f3953`.

## Prostate geometry

`auto_rule_v1` evaluated all 31 audited geometry mismatches. All selected
`index_geometry_repair`; none required manual isolation. Decision CSV
SHA-256: `c2e1b921f0e9d6fda925cf09ead78692adc2aadba876342dc3e546495fb8f45f`.

## M&Ms fixed crop FOV

The deterministic eight-patient vendor-stratified pilot passed at all
candidates, then full ED/ES retention selected the smallest globally accepted
FOV: `320 mm x 320 mm`. Full minimum foreground retention was `1.0`.

- `256 mm`: full minimum `0.9081896834`, rejected.
- `288 mm`: full minimum `0.9928557383`, rejected.
- `320 mm`: full minimum `1.0`, accepted.

FOV config SHA-256: `c40bfaad6e4ed5b0b749849b423409de04654678cc16127bc23196ecd033a48c`.

## Cohort policy

- `canonical320`: Siemens 95, Philips 125, GE 50, Canon 50; evaluation
  eligible.
- `auxiliary25`: official `Training/Unlabeled` GE patients; image-only in all
  training manifests and never evaluation eligible. Their local GT HDF5 is
  diagnostics-only.

## HDF5 contract

Schema and preprocessing version are both `v1`. Each payload is written as a
temporary file, flushed, reopened and validated, SHA-256 hashed, then atomically
renamed. Compression is Gzip level 4 with shuffle and Fletcher32 enabled.

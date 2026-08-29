# CRISP-Seg V0.1 functional audit — attempt 1 failure bundle

**Status:** `DIAGNOSED_NUMERICAL_SUBGRADIENT_FAILURE`  
**Optimizer steps:** `0`  
**Formal functional artifacts written:** `none`  
**Frozen checkpoint/data mutation:** `none`

All three independent seed processes exited at the first transition and first update batch with the same exception:

```text
FloatingPointError: invalid C2 virtual gradient norm: nan
```

The failure localized to spatially constant/dead decoder channels in PFC. The registered forward expression was finite, but differentiating `sqrt(sum(x^2))` at an exactly zero centered map produced an undefined `0/0` gradient. The repair defines the zero-vector subgradient as zero while preserving the exact nonzero forward expression `x / (||x||_2 + 1e-8)` and the exact zero-vector forward value.

Evidence:

- Identical driver-log SHA256 for seeds 0/1/2: `841b52a1850586a31358c48a1ca6d2857c5c71a3d86a2520faf7a081596cff7f`
- Attempt-1 `channel_roles.py` SHA256: `5a66c016205695fe312947392b7e18bd125259aaf9ade76582b521c7e2a90949`
- Repaired `channel_roles.py` SHA256: `40f9c562ada398057657a865edc9674aa404a3bfb856d40600827074fcd5536c`
- Audit driver SHA256: `dc79710df362d6b28fd20e64bffd5166ac697cef75be9ca1eb9ee04fe426b0fb`
- Post-repair targeted tests: `13 passed`
- Post-repair one-batch C0–C5 forward totals are bit-identical to the pre-repair smoke values; all six gradient norms are finite and old-model `.grad` count is zero.

The role states remain valid and immutable. A new driver log name must be used for attempt 2; no attempt-1 log or completed role artifact may be overwritten.

# Gate 1C v3 failures and warnings

Primary global candidate/normalization rows with undefined zero-gradient cosine: 0; these remain null and never pass a required comparison.

Unsupported validation points are explicit nulls, not extrapolated. Stage0/missing-history scores and null-feature scores are structural nulls, never fake normalized features.

Official classifier constructor/import RNG side effects are preserved; each registered forward reseeds afterwards. Inactive sigma/grad_update are inventoried, not silently dropped.

All raw stdout/stderr logs, test failures if any, and unit artifacts are retained. No result-driven retry, threshold adjustment, new transport fit or baseline update.

Fresh v3 execution: 990 validation forwards, 75 separate integration forwards and 1800 formal forwards; total 2865. All 495 validation caches, raw native tensors/PAS intermediates, 9 validation guards, 12 integration guards and 288 formal guards were generated in this protocol. R1 reads the direct historically hashed PAS bank; no reconstructed bank or old private cache/golden is used. Reduced-method candidates are decisions for a separate preregistration only; no method or C0 is implemented here.

Reduced method candidate: NONE. Historical-bank claim allowed: False. A passing pixel-normalized R2 can nominate a current-only future candidate but cannot change the R3 Gate1C status.

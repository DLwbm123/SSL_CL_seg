# V94 pooled training advantage scale

Test whether preserving absolute across-state reward-gap amplitudes improves transfer compared with per-state normalization. This follows V93 negative representation evidence and V92 policy concentration diagnostics, not a confirmed mechanism.

Reuse V92 data and the native engine. Change only advantage scaling, with full-network RL and CE controls, both original actor seeds, 1024 updates each. All 136 endpoints must seal before new development scoring. No additional training queries; total 3208 native and 4096 actor updates. Exact formula, budget, diagnostics, thresholds and limits are frozen in PREREGISTRATION.json.

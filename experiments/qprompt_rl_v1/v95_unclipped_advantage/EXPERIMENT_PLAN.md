# V95 retain complete pooled advantage magnitudes

V94 showed a paired development improvement but failed its original primary. Independently test the remaining advantage clipping: remove only magnitude-3 truncation while keeping pooled RMS, gradient clipping, KL, entropy, actor initialization, training data and1024updates unchanged. This is one fixed intervention, not a coefficient search. Both CE/RL and both seeds are retained, with all historical controls.

No training queries or rollouts;4096actor and3208nativeupdates;152endpointssealedbefore192newdevimageevaluations. Exact formula, diagnostics, criteria and boundaries in PREREGISTRATION.json.

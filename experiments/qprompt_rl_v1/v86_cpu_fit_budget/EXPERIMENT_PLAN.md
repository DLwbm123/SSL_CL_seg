# V8.6: fixed actor optimization budget diagnostic

The delivered V8.5 combined intervention failed. Its full-table sharpened-target KL remains about 0.55 at 64 updates, so a longer, fixed optimization budget is a concrete unresolved explanation. This is a new adaptive CPU diagnostic; V8.5 and prior STOP decisions remain unchanged.

Keep both raw and standardized inputs at the same sharpened T=0.25 target, architecture, 601/602 seeds, Adam learning rate, entropy and gradient clipping. From scratch, execute exactly 1,024 updates per held-context fit and per full-table fit. Preserve each 64-update baseline by reuse, and verify the new trajectory's update-64 parameters against the old checkpoint before continuing. The optimizer is not reset at that boundary. Only final update1024 models are scored; intermediate loss logs are descriptive.

The primary standardized model must beat all fold-trained controls for each seed and improve on its own 64-update baseline by at least 0.0005 mean raw expected reward. The raw model is an attribution control, not a replacement primary. No GPU endpoint follows unless a separate development plan is frozen. The new budget is 36,866 CPU actor updates at most, zero student updates and zero image queries. There is no further epoch or seed search in this protocol.

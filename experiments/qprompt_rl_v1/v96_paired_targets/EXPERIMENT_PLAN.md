# V96 paired-stream training targets

Test whether sharing action-gap targets across the two existing trajectories of a context/time improves transfer. Keep both original state vectors; average centered reward targets only. Paired states are different, so this tests target invariance and can erase real state-dependent information; it is not an unbiased same-state noise estimator.

Matched V94 scaling, RMS, clipping, initialization, reference, objective and fit budget remain. Both CE and RL use both seeds. No new training queries, states, streams or rollouts.4096actor/3208nativeupdates;168endpointssealedbefore192newdevimageevaluations. Exact specification and thresholds in PREREGISTRATION.json.

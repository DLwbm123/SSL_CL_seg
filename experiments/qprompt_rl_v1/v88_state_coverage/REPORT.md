# V8.8 query-free deployed-state diagnosis

{
  "unique_primary_step200_states": 4,
  "primary_step200_records": 8,
  "primary_coverage_fraction": 0.25,
  "coverage_priority": false,
  "saturation_median_increase": 0.15625,
  "saturation_priority": false,
  "training_distance_limits": {
    "100": 0.5782143339541037,
    "200": 1.0102295469197238
  },
  "probability_replay_checks": 48,
  "raw_private_features_published": false,
  "causal_claim": false,
  "new_queries": 0,
  "actor_updates": 0,
  "student_updates": 0,
  "V84": "NOT_RUN"
}

Adaptive descriptive analysis of repeatedly observed D1 development. Distances use the frozen full-table scaler. The threshold is a coverage heuristic, not a calibrated OOD test. Duplicate states are collapsed for the primary coverage fraction. Saturation compares both seeds and is descriptive. No counterfactual development reward is inferred. Raw features and checkpoints stay private.

## Interpretation

The preregistered broad-coverage hypothesis is not established: only1/4 unique primary step200 states exceed the same-time training leave-context-out maximum, below the one-half rule. Median hidden saturation rises0.15625, below the0.25 rule. All48 saved originating-policy distributions replay within1e-6 and all four entry100 features match across methods; there is no observed inference/replay engineering mismatch.

The one flagged state is dev2, also the worst V87 context. Its standardized nearest distance is1.15197 against reference1.01023, with20/32 hidden units saturated for seed601. This localized association motivates caution but is not proof that state shift causes the performance deficit. No primary state has a standardized coordinate above5, and none shifts a globally constant training dimension. Blanket input clipping or a larger network is not supported by this diagnostic.

A separate structural mismatch is visible in the owning code: V83 rewards each decision's following100-step rollout using the local25/100 readouts, while deployment scores the complete100-to300 episode using150/300 readouts. Its training step200 entry also follows native action0 rather than a trajectory drawn from the current policy. These definitions are factual; whether aligning them improves performance remains an untested hypothesis.

The next controlled intervention will therefore collect paired training-only counterfactual rollouts on current-policy trajectories, record both local and whole-episode returns from the same runs, and compare otherwise identical policy fits and deployment. It must be separately preregistered. No V84 gate is unlocked and no GRPO success is claimed here.

New cost is exactly zero optimizer updates and zero image/label queries. Existing private features/checkpoints are read on CPU. V87 and all prior STOP results remain unchanged. This is adaptive D1 diagnosis, not independent validation.

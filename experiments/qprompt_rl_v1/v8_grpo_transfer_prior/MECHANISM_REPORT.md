# Mechanism findings

The memory transfer gate averaged 0.9914. Actor distributions changed; they are not frozen or inactive. However, 71.875%/75% of groups skipped policy updates under the original global noise floor. New/old query gains did not pass the held-out-condition development gate.

Paired action-minus-native noise calibration on original A yields 0.00073794 instead of the absolute-reward 0.00202752 estimate. A separately registered V8.1 diagnostic will test this one change. This is a hypothesis, not proof of the failure cause; action semantics, label budget, native schedule, reward design and simulated-to-target mismatch remain possible limitations.

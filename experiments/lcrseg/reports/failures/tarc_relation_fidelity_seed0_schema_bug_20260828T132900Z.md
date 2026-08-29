# TARC relation-fidelity seed-0 CSV schema failure

**Classification:** engineering-only output-schema failure  
**Optimizer steps:** `0`  
**Checkpoint/data mutation:** `none`

The first seed-0 relation-fidelity calculation completed, but the CSV writer inferred its schema from the first `previous_fidelity` row. The later `current_safety` rows contain additional fields, which were therefore omitted by the writer's `extrasaction="ignore"` behavior. The malformed CSV and its premature summary were preserved with this failure bundle and are excluded from all TARC gates.

Correction: construct the deterministic union of all row keys and pass it as the explicit CSV schema. The same frozen checkpoints, visible data views, formulas, and seed are then rerun. No research logic, threshold, or hyperparameter changed.

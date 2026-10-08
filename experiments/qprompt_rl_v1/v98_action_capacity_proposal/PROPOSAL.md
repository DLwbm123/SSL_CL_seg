# V98 proposal: memory-mixture capacity

Status: **PROPOSED_NOT_AUTHORIZED_NOT_RUN**. No experiment has started.

V97 exhaustively rules out the original practical margins on the fixed nine-action trajectories. Repeating actor fits cannot overcome that finite-grid bound. The next bounded hypothesis is insufficient memory-mixture strength: add alpha=.75 to the existing {0,.25,.5}, crossing the same three class-weight vectors. Keep all original action IDs, gate, losses, teacher detachment, native student, entry checkpoints, H300, roles and streams unchanged. This creates 12 diagnostic actions; existing actors remain untouched.

Reuse the 324 old grid outcomes and prefixes. Execute exactly 252 new endpoints and 12 new midpoints across the same four contexts: 26,400 training plus 8 qualification updates; zero actor updates, zero training-query accesses, 2,064 development image-evaluations, no new images. Seal every new endpoint before scoring. Test the same thresholds across all 144^4 combinations; retain all failures and the original nine-action negative result.

This is a feasibility diagnostic, not a trained policy or independent confirmation. No grid targets feed actor training. A positive outcome would require a separate matched CE/RL protocol and later independent evidence. Full matrix, qualification, budget and decisions are in PREREGISTRATION_PROPOSAL.json.

**Approval needed:** permit adding this one alpha level for the diagnostic. The current human instruction explicitly fixes the existing nine actions, so continuing-research authorization alone does not authorize this change. Approval does not unlock new domains, hidden labels, sealed test, low-label full matrix or altered thresholds. No runtime code or launch has been created for V98.

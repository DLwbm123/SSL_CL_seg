# POS/MEO formula-to-code map

**Status:** `BLOCKED_POS_SPECIFICATION_AMBIGUOUS`; no POS/MEO code has been added.

| Source item | Official meaning | Frozen JASCL location/adaptation | State |
| --- | --- | --- | --- |
| Paper Eq. 1, `L_sup` | Supervised pixelwise CE | `di_dmpa_jascl/runner.py:367-373` already computes the current-domain labeled-cycle CE | Reuse unchanged |
| Paper Eq. 3, `L_unsup` | Teacher-pseudo-label consistency CE | `di_dmpa_jascl/modeling.py:314-351` supplies the frozen PAS probability-MSE objective required by this protocol | Explicit cross-framework adaptation; loss and masks unchanged |
| Paper Eq. 5 | Minimize `||alpha_s*g_sup + alpha_u*g_unsup||^2` with nonnegative coefficients summing to one | Intended only for the PAS phase in `di_dmpa_jascl/runner.py:318-415` | Not implemented |
| Paper Eq. 6 | Analytical POS coefficients for ordinary nonzero gradients | The current inventory starts from all `student.parameters()` with `requires_grad` at `runner.py:359` | Blocked: official executable parameter subset, `None` placeholders, zero and tie rules unavailable |
| Paper Eq. 13 | Unit POS direction scaled by `||0.5*g_unsup + 0.5*g_sup||` | Would replace only the gradient written before the existing single `optimizer.step()` at `runner.py:384` | Blocked: POS-zero denominator rule unavailable |
| Paper Eq. 4 | Incorporate coefficients into one weighted objective | B0 currently backpropagates `L_sup + 0.5*L_pas` at `runner.py:373-376` | E0 remains unchanged; E0_RECOMPUTE/E1/E2 not registered |
| Optimizer | Apply one combined update | Existing Adam optimizer is created at `runner.py:119-129`; one PAS step occurs at `runner.py:384` | Reuse unchanged if source gate later passes |
| Scheduler | No POS-specific schedule; use host baseline | Existing polynomial `LambdaLR` is created at `runner.py:125-128` and stepped at `runner.py:243-245` | Reuse unchanged |
| EMA teacher | Teacher produces detached target | `modeling.py:336-351`; EMA update order remains `runner.py:414` | Reuse unchanged |
| GAS | Not part of POS/MEO | GAS updates only in the supervised phase at `runner.py:286-288`; PAS phase explicitly does not update it at `runner.py:383` | Reuse unchanged and isolate POS/MEO |
| PAS decisions | Not changed by POS/MEO | Student/teacher stochastic forwards, confidence/similarity masks, joint validity, and probability loss are in `modeling.py:281-351` | Reuse unchanged |

## Missing official mappings

No authoritative source maps the paper to an executable parameter list, `None`-gradient placeholders, exact equality branches, one/both-zero gradients, MEO with a zero POS norm, or reference synthetic outputs. These are required tests, so choosing local behavior would violate the source-only rule.

The only valid next action is to obtain attributable official clarification. Preregistration, authorization, implementation, parity, and training remain unstarted.

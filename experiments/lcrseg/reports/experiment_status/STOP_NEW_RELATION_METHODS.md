# Stop new relation methods

**BPRC status:** `BPRC_GRADIENT_SCALE_NOT_SUPPORTED`

BPRC-Seg V0.1 did not pass every preregistered feasibility gate. Under the frozen protocol:

- do not register or train a BPRC method;
- do not create BPRC training configs or run an optimizer;
- stop proposing new relation-coordinate or relation-loss variants;
- any future work is limited to source-faithful DC2T/JASCL-PAS reproduction and preregistered strong baselines under a new protocol.

The decision evidence is `reports/experiment_status/BPRC_FEASIBILITY_AUDIT.json`.

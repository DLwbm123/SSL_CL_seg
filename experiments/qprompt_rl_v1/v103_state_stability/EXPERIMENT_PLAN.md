# V103 fixed-probe state stability

V102 found repeatable action rankings but inconsistent state-predictor transfer. Test whether a fixed small probe set improves the mapping, with no new student training or reward queries. The machine-readable preregistration is authoritative.

Compare original state, fixed probe 0, and the mean of four fixed probes. Single-probe control separates common-batch effects from averaging. All original-state replays must pass before novel features are extracted. Reuse the existing extractor, snapshot restoration and V102 fold predictors; do not change native training code.

This is training-only diagnosis on shared source/query roles, not independent patient or domain validation. Retain every fold and cost, including failures.

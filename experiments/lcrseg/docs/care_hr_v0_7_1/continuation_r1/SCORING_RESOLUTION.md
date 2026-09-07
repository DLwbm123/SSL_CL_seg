# R1 scoring resolution

The user explicitly adopts frozen PPC-SHOR V0.6B `shor_v0_4_test.case_metrics`. Historical V0.7 differences and the predecessor BLOCKED report remain unchanged. New scorer/parity/combined-action evaluation live only in V0.7.1. GT=255 masks both prediction and truth support in the evaluator. Empty classes score 1. All-ignore rows retain compatibility weights but supply no safety evidence. Foreground gain means rim/cup macro Dice.

The only modification to a predecessor pure component is an early no-op return for zero current foreground in all three enumerated spaces. This closes the relaxed-space zero-foreground boundary required by the adopted protocol. The old semantic audit and all old namespaces are unchanged.

No real GT or domain rows have been opened for this continuation at registration. The R1 attachment is an explicitly adopted supplement, not a retrospective review signoff.

`run_checks.py` now reports R1 coverage and continues to run the same predecessor regression directories; it does not change old test expectations. Public A1 evidence is released after the tested source commit and supplied externally to its immutable server checkout. The action seal binds that release and the exact server test report.

The executor uses the original HDF5 file hashes after admission, reading each approved label file once into memory for both hashing and decoding. Evaluator CSV projection restricts true-domain/label columns to the already admitted seed-case rows. No probability inference is needed if the nine frozen cache files verify. A missing cache is reported, never substituted.

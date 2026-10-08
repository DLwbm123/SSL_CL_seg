# V107 startup: training coverage collection

RUNNING, not complete. Native qualification passed: two archived stable states replay exactly, two paired two-step continuations replay exactly, and probes/behavior policy preserve full native state and RNG. All eight qualification updates succeeded; the qualification worker exited zero. Four entry workers started on GPUs 4–7. Parent and child command lines and GPU process displays were verified neutral. NAS mount/write probe and memory admission passed.

Fixed planned scope: 12 additional training contexts, 24 stream collections, 48 states/576 returns; 85,208 native updates (including 8 qualification), zero actor updates or linear solves, 5,664 Q_train image evaluations and zero Q_dev accesses. Old 32 states/384 returns are reused unchanged. See PROTOCOL.md for the complete matrix and seeds. No GPU wall-clock limit.

V106 did not establish practical RL benefit: sampled RL minus CE was −0.093915 percentage points new, −0.124227 points old and −0.002156739 utility. RL minus target-matched CE was only +0.000115112 utility, with one seed tied. This new collection tests a coverage hypothesis motivated by observed deployment feature ranges; it is not independent confirmation. The added photometric conditions overlap previously viewed development conditions. Same existing source/cases, no new annotation and no unseen-domain claim.

Next, on completion: audit all attempts including failures, publish complete aggregate collection results, separately freeze matched policy fits and actual sequential deployment. A training-table improvement does not finish the research objective. Private states, checkpoint tensors and raw logs stay on NAS; source, protocol and sanitized receipts are public.

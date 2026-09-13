# Submission scope and missing evidence

This is an editable empirical-study draft, not a claim of acceptance readiness. IEEE TMI lists novelty, technical/writing quality, scope and substantial impact as separate criteria: https://ieeetmi.org/key-criteria-for-publication/ (checked 2026-09-13). Completing a fixed matrix or removing conjunctive performance gates does not satisfy those criteria automatically.

| Missing evidence | Current boundary | Consequence | Work authorized in this package |
|---|---|---|---|
| Independent patients unused in training, U inputs, initialization and model selection | NOT_ESTABLISHED. Existing manifest val/test names alone cannot establish lifetime non-exposure. No new hidden/test GT is opened. | All current intervals and results are conditional development evidence. | One bounded metadata/provenance check; do not search indefinitely or start new evaluation. |
| Third training domain | NOT_INCLUDED. Only two orderings of RIM/Drishti are fixed. | No three-domain continual retention claim; third-domain zero-shot evaluation would not replace training through it. | No additional domain or sequence training. |
| Published continual-learning method comparison | NOT_INCLUDED. C_CE/C_CED/FULLMIX/LCTX are controlled recipes, not a complete literature comparison. | No SOTA ranking; no recovered original KI claim. | Describe gap; do not restore unidentified methods. |
| Separate attribution of donor supervision vs Dice grouping | NOT_IDENTIFIED by current FULLMIX contrast | Any difference is a joint recipe effect. | Report that limitation, do not add factorial arms. |
| Label fractions beyond target half-budget | NOT_INCLUDED | No monotonic label-efficiency curve or all-sequence 10% claim. | Report fixed target sensitivity; no 5%/15% rescue grid. |
| Broad novelty and clinical impact | UNESTABLISHED | Working title and claims remain conditional; no zero-forgetting or clinical safety claim. | Draft and evidence map; no invented evidence. |

All negative or inconsistent new results enter the manuscript. No result direction triggers automatic further experiments. A separate future user decision would be needed to expand the submission evidence beyond this package.

The bounded metadata check found test-role entries (RIM 40, Drishti 25, REFUGE 100), but did not establish their lifetime non-exposure. DATA_SCOPE.json records these counts without patient identities. No test arrays were opened and no new independent evaluation was started.

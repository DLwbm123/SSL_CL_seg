# Gate 0 v2 leakage audit

Source commit: `fb55e8022bc379e2515a46214c6fdf45ea818de6`.
Preflight: **PASS** for frozen Fundus seeds 0/1/2.
Hidden-GT training usage: **none**.

The actual adapter verifies frozen manifest/split hashes. Every training view
is restricted to the current domain and train_labeled/train_unlabeled roles.
Unlabeled records have no label path; their datasets/batches contain no label
tensor. Val/test roles are rejected by the training API.

Case IDs can contain a legacy raw-source `test` string. That string is not the
frozen LCRSeg role: the hashed `primary_20pct_split` field is authoritative.
The PAS gradient audit uses only frozen train_unlabeled images and
train_labeled images/GT. Its v1 checkpoints are read-only.
Evaluation-randomness diagnostics use REFUGE **validation**, not test GT.

Formal v2 precision diagnostics are validation-only evaluator outputs, never
inputs to loss weights, threshold choice, or checkpoint selection.

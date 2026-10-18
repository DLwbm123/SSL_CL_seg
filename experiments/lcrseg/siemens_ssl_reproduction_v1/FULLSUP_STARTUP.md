# Matched full-supervision startup receipt

Status: RUNNING, not a scientific result. Source implementation: `8899993`.
Exactly one seed163 job was launched with a hard cap of 43,600 optimizer calls.
The initial short check confirmed step 1 / physical update 1, finite supervised
terms (original L: 1.0625035763, former U ground truth: 1.0819909573), and a live
GPU worker with a neutral process command. No automatic monitor was created.

Validation before launch:
- Seven synthetic tests passed, including unchanged FixMatch/CPS behavior,
  true-label coverage without a weak pseudo-label forward, and rejection of
  unauthorized or mismatched label records.
- Native finite backward and checkpoint restore passed, zero optimizer updates.
- Initial model tensors matched the existing FixMatch step-zero checkpoint.
- Sampled patient identities and images matched the original image-only stream.
- The original SSL accessor retained no label fields for training U.
- Exactly 48 training patients, with no patient overlap with held-out roles.
- Existing validation rows reproduce 80.3674066377% Dice over 23 patients and
  46 ED/ES volumes; no repeated baseline inference.

Outputs, extra training labels, checkpoints and raw logs remain private on the
approved data volume. Final EMA evaluation and paired comparison are built into
the job. Public results/report delivery remains pending actual completion and
receipt verification; no completed result or test-set performance is claimed.

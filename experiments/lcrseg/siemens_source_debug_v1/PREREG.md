# First-task semi-supervised development on GPU2

Authorized2026-10-17: migrate the project to pro5000, put large files on its data
volume, debug the first task in parallel on physical GPU2, and monitor every30 minutes
until success or an explicit resource boundary requires a decision. The existing
my-gpu GPU0 L-only comparison continues unchanged.

Seed163 only. Aggregate actual optimizer-call ceiling remains130,800 across the
L-only round and all subsequent debugging/control runs; failures and replays count.
Reserve43,601 for L-only,43,600 for this candidate, and43,599 unallocated. No extra
real optimizer qualification calls are outside these allocations. CPU synthetic
checks and native initial-loss/backward/checkpoint restore checks use zero updates.

HARD_U_V1 tests whether the weak, background-dominated soft KL objective limits U
benefit. Replace only U's soft confidence-tapered inverse-frequency-capped KL with
hard pseudo-label class-balanced CE plus present-foreground Dice on the same admitted
pixels. Keep confidence0.7, PAS background admission, lambda_U0.25, ramp, LCTX,
hierarchy, architecture, initialization, augmentation and source schedule unchanged.
Average CE equally across present pseudo classes; compute Dice only for present
foreground classes; detach pseudo targets and mask. Empty masks produce zero loss
and zero gradients. No U ground truth, independent test, GT-guided ROI or pretraining.

Use the original10 Siemens labeled patients, current-domain U and23 validation
patients. Fixed final43,600 scientific steps; save every2,000. Evaluate final train-L
and validation once, reusing existing SSL validation. Report foreground classes and
ED/ES, paired differences and training-validation gap. Validation guides development
and is not an independent test. Do not select or discard runs based on favorable results.

The aspiration is competitive first-task SSL, with84–87% Dice from the discussed
papers serving as a contextual reference rather than a verified same-protocol target.
Judge improvement against the existing65.8176% seed163 result and the running matched
L-only comparison. A claim of comparable paper performance requires aligned conditions.
Hardware changes to Blackwell/Torch2.7.1 are recorded; bitwise trajectory equivalence
is not assumed. Source copies and failed records remain intact during migration.

# Startup verification

Five CPU checks passed: author-style masked-mean FixMatch, reciprocal CPS,
deterministic CE value/gradient equivalence, deterministic strong augmentation,
learning-rate bounds and EMA update. Both native preflights passed finite backward
and initial checkpoint load with zero optimizer calls. Initial objectives were
1.0644010305404663 for FixMatch and6.059787750244141 for CPS.

Two earlier native preflights failed before any optimizer call: CUDA spatial NLL
did not support deterministic reduction, and the plain parent constructor left
parameters frozen. The implementation now uses equivalent log-softmax/gather CE
and explicitly enables the existing dense source training parameters. Failed
preflight files remain intact. A separate controller bootstrap construction failure
also occurred before any worker training; its log and receipt were preserved.

The corrected detached controller launched FixMatch on physical GPU2. Startup
confirmed the process birth, neutral controller/worker command lines, GPU process,
passing formal preflight and successful actual optimizer calls. CPS was initially queued after its zero-update qualification.

The data-volume mount, free space, write/read probe and GPU memory checks passed.
All run artifacts and copied author source references are on the data volume.
There are no benchmark results yet. The new allocation is43,600 FixMatch calls plus
87,200 CPS calls, separate from the exhausted prior study; no extra replay budget.

## Parallel scheduling amendment, 2026-10-17

Following the user request for better GPU utilization, only the waiting controller
was retired; the detached FixMatch worker continued unchanged. CPS was then
started concurrently on GPU2 and passed its formal zero-update preflight.
Both workers have neutral command lines and successful optimizer ledger entries:
FixMatch 9,720 calls; CPS 357 iterations / 714 calls at the startup check.
A five-second GPU sample showed SM utilization 75/100/100/100/75 percent
(mean 90%), versus 73/73/34/85/34 percent (mean 59.8%) before the change.
GPU memory was 6,729 MiB. These short samples do not establish sustained
throughput improvement. No batch size, seed, objective, precision, iteration
count, or optimizer budget changed. Monitoring now follows both workers directly;
the retired controller must not be restarted. No final evaluation is available.

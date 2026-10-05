# R0 early 20% evaluation amendment — 2026-10-05

Before any campaign validation scores were read, the user requested: “能否先看看 20% 的结果”. This authorizes evaluating the completed 20% scope before the 10% and 5% training scopes finish. The original global training-before-validation barrier is explicitly amended for these 18 existing endpoints only; all 42 training/preparation jobs in the 20% scope must already have completion receipts.

Evaluate all methods and both target domains with the existing frozen evaluator, including old REFUGE retention. Reuse the registered evaluation job names and receipts so the main coordinator skips completed evaluations later. Keep all training settings, label subsets, remaining jobs, success thresholds and continuation sequence fixed. Do not select, stop or retune remaining jobs based on these interim scores. No new training, test access or hidden-label access is authorized by this amendment.

Report this as a single-source, single-subset/controller 20% interim result. It cannot satisfy the prespecified 10%/5% low-label screen or replication gate. Publish every method/domain and adverse channel. Retain evaluator costs in the full campaign accounting.

The currently running coordinator retains its original in-memory report function. After it finishes, regenerate the aggregate report using the amended report function before publication, without rerunning training or scoring. The final audit must state that the global training-before-val barrier was amended; it must not claim all campaign training finished before the first val access.

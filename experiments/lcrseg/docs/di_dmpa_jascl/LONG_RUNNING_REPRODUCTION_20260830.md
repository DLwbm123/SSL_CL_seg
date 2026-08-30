# Long-running reproduction authorization and execution plan

Recorded 2026-08-30 (Asia/Shanghai). This is a prospective workflow amendment,
not a replacement for any frozen scientific protocol or result.

## User authorization

The user instructed in the current conversation:

> 将这个会话修改为长期运行的，结果出来后，你自动分析结果，并给出下一步的计划，然后你继续执行，直到几个方法能复现成功，不一定要和原文一样的数值和表现，接近就可以，如果原始代码有问题，你也可以对代码进行自己的修改

The user subsequently narrowed the scope explicitly:

> 几个方法我说错了，就是指你现在在跑的方法

Only the current JASCL / DI-DMPA medical-segmentation research line is in scope.
B0/C0 remain its controls, not additional reproduction targets. The older
multi-method wording in the initially created Goal is superseded by this
user clarification; there is no pending request for an additional method list.

This authorizes continued monitoring, result analysis, planning, implementation,
verification and subsequent in-scope experiments without waiting for approval
after every step. Actual code defects may be repaired. It supersedes the prior
workflow-only requirement to stop after a report and the prohibition on retrying
an infrastructure failure. It does not retroactively turn a failed gate into a
pass or make changed implementations identical to their upstream originals.

The native long-running goal is active in thread
`01a04c7a-b760-7443-928a-b7683bb8d9ee`. The same-thread follow-up automation is
`ssl-cl-seg`, scheduled every 15 minutes. No extra experiment scheduler, new
environment, dependency or paid compute resource is introduced.

## Immediate continuation

1. Preserve Gate 1C validation attempt1, including its 94 passing synthetic
   tests and publication-network timeout before any real checkpoint load.
2. Recheck GitHub, exact code, frozen input hashes, available resources and
   absence of duplicate workers. The 2026-08-30 live recheck successfully read
   `68dedea7ccaa9144913dfc50a096364d7d55f2cf` from the frozen code branch.
3. Use a new immutable validation `attempt2` with the same code, cases and
   seeds. Bind this amendment's commit and file hash in a separate operational
   receipt. Do not overwrite the original preregistration or authorization.
4. Only after all synthetic and real integration checks pass, execute the
   frozen Gate 1C stages: nine validation units, 72 original gradient pairs,
   576 teacher draws, posterior-mean and PoE controls, C1-C8 and all audits.
5. Publish the actual result, including failures, and decide the next finite
   experiment from evidence. Repairs use new commits and attempt directories.

The frozen scientific identities remain:

- Preregistration: `32d32ab5e491f2e14c3edde6b4f319f978217351`.
- Original authorization: `d6b651fd366dd304ab4d190f7eb5ce9d3afe23ea`.
- Diagnostic code: `68dedea7ccaa9144913dfc50a096364d7d55f2cf`.
- B0-EMA, K=2, identity history; R4 unavailable; all original C thresholds,
  sampling plans and stage barriers unchanged; zero optimizer updates in C.
- Gate 1B / overall Gate 1: `FAIL_TRANSPORT_NOT_SUPPORTED`, unchanged.

This document is published on `codex/sslcl-long-running-reproduction` so that
the exact-code branch remains unchanged for the diagnostic publication check.

## Current-method scope and completion criteria

Continue the current Gate 1C, analyze its full result and automatically propose
and execute evidence-based repairs and versioned experiments for this method.
Do not count B0/C0 controls as multiple paper reproductions or add other
methods, projects or datasets. Validate a finite pilot before scaling up.

Before new performance experiments, record the upstream paper/commit, benchmark,
architecture, labels/splits, primary metric, acceptable numerical gap, seeds
and a finite experiment budget before new performance experiments. Do not
invent an already-agreed numerical tolerance. A UNet medical adaptation under
a different benchmark is not a directly comparable reproduction of an original
DeepLab/nonmedical paper. Report that distinction explicitly.

Each iteration must state a falsifiable diagnosis, its minimal code/config
change, tests, commands, resources and acceptance rule. Reuse shared loaders,
PAS, training/evaluation engines and checkpoint machinery. Verify unit and
integration behavior, finite values, gradient flow, checkpoint reload/resume,
complete stage-by-domain matrices and label isolation. Preserve original and
repaired variants, all seeds and failures, provenance hashes and raw evidence.
Never select a favorable panel/seed after seeing results or tune on final-test
or hidden diagnostic GT. Revised scientific hypotheses require new prospective
versions and cannot change the historical verdict.

## Resources and stopping conditions

Use only the purchased host `root@162.14.139.38:31192`, its existing two GPUs,
`/root/.venvs/lcrseg-py310/bin/python` and `/root/LCRSeg`. Parallelize independent
jobs when memory permits; never alter a frozen batch, precision or algorithm
just to inflate GPU utilization. Check existing PIDs and output directories
before every launch. Never run duplicate workers or overwrite a run.

On completion, analyze the full artifacts and give the next plan, then execute
within the confirmed scope. Retain all formal checkpoints and failed attempts.
Do not change global network/storage settings, purchase resources, force-push,
publish private data or merge main. Continue source/report publication only on
the authorized experiment branches, with secret/data checks first.

Only claim the long-running objective complete when the current method meets
its predeclared acceptance rules with complete multi-seed and engineering
evidence. Scientific failure is reportable evidence, not permission for an
unbounded search. Ask for direction if an unavailable
resource or material change of scope blocks progress. Stop future launches if
the user asks to stop; stop the follow-up automation after genuine completion.

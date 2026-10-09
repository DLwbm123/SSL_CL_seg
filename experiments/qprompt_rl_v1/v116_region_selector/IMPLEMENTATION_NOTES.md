# Implementation boundaries

The native StageTrainer optimizer/constraints/EMA/supervised engine is unchanged. New code supplies a detached spatial selector and finite job orchestration. quarter.py is the existing V113 rate adapter; its old coordinator is not invoked. Old gate/actor source files are untouched. No dependencies added.

Before any V116 native/actor execution, the preregistration normalization sentence was clarified from per-image to the existing batch-global mass normalization. This is required for the specified full-budget native loss/gradient equality. The initial preregistration is commit5ac0ed3; the final source/protocol commit is recorded as preregistration_commit in the launch receipt. There was no experiment, query, optimizer call or result-driven tuning between these versions.

The CPU source selfcheck invokes no optimizer/model forward. Native qualification owns the separately declared8native+2syntheticactor calls. Checkpoint/query state restoration uses the existing fullsnapshot API; physical ledgers and selectorprivate RNG are handled outside restoredstudent state. Branch0 retention is fixed. All 32groups forbothseeds run, including equal-return groups.

CE uses highest-return behaviorbranch demonstrations and follows the same online order; it does not control collection, and is not analyticaltargetCE. This pilot cannot establish the historical matchedtarget RL claim. There is no global-stable-state probe because candidatefeatures come from the existing loss forwards. Fractionalbudget is fixed perstate; differing learnedstates can have differing absolute eligiblepixelcounts. Allpublic results must state these limitations.

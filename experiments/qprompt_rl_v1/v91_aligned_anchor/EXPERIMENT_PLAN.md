# V91 reward-aligned KL reference

V90 exposed a matched-data divergence: denseCE improves currentD1 new/old scores, while RL stays near its old policy. This stage separates three factors with the same denseCE initialization: continuedCE, RL anchored toV86, and RL anchored to the retention-aware denseCE itself. All use1024 additionalupdates and the same32-state full-action table. No new reward collection is needed.

Before fitting, derive and evaluate the per-state exact optimum for the fixed KL/entropy objective; report objective gaps and expectedtrainingreturn without using new querylabels or optimizersteps. This distinguishes the objective restriction from network optimization error but is not a student-performance bound.

Newbudget6144actor+4808nativeupdates;64reused+24new endpoints, all88 sealed before288devimageevaluations. Theprimary must beat both matchedcontrols and the alreadystrongdenseCE; priorSTOPs are immutable. SeePREREGISTRATION.json for exactmethods,criteriaandlimits.

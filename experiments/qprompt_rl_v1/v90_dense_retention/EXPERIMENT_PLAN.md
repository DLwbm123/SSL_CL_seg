# V90 continuous retention credit

V89 is publicly sealed as STOP at a9dbe223ad2878dc92c0f5efadc652235039e3c6. This independent iteration tests the observed dead-zone in its training reward. The signed old-score change is measured on existing Q_train_old only; no development labels define new reward coefficients.

Re-score 272 existing terminal checkpoints and eight shared episode entries, requiring zero native updates and 1,120 old-query image evaluations. If old-action variation passes the frozen signal check, fit DENSE_CE and DENSE_RL at both601/602 using the exact V89 training recipe and4096 total actor updates. Reuse all48 previous endpoints and add16 new endpoints (3200 native updates plus8 qualification). All64 endpoints must be sealed before192 new development image evaluations.

The primary, fixed coefficients, all controls, thresholds, data boundaries and exact budgets are in PREREGISTRATION.json. A signal-gate failure stops without optimizer work; a deployment failure remains a negative result. Dense-reward and original-reward fits consume the same1024 updates per seed and complete branch information, while the additional old readout cost is explicitly charged to V90.

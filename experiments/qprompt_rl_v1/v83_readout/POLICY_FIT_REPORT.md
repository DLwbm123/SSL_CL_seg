# Policy fit and held-context diagnostics

Raw reward units; entropy in nats. These are expected categorical values, not argmax deployment.

| Evaluation | Seed | Reward | Actor − uniform | Actor − global | Actor − time | Entropy | Target KL |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full training table | 601 | 0.001114192 | 0.000451857 | -0.000701759 | -0.000810759 | 2.034099256 | 0.211052352 |
| Full training table | 602 | 0.001112922 | 0.000450587 | -0.000703029 | -0.000812030 | 2.034393680 | 0.211392263 |
| LOCO held contexts | 601 | 0.001026636 | 0.000364301 | 0.000030226 | -0.000250827 | 2.031450362 | 0.260422843 |
| LOCO held contexts | 602 | 0.001027500 | 0.000365165 | 0.000031090 | -0.000249963 | 2.031992923 | 0.260420319 |

All action probabilities and fold-specific control choices are in the CSVs. Formal actors were never updated.
LOCO used 1,024 CPU updates and one synthetic qualification update. No student updates or query-image reads.
The held contexts share D1 image pools and auxiliary model sources. No patient-independent or cross-domain inference.

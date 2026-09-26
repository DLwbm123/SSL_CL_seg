# Zero-update forensic

400 paired batches; 400 student + 400 reference forwards; 1,600 autograd.grad calls; **zero optimizer calls**. No validation used.

Per-batch regret p50/p95/max and all gradient norms/cosines are in the CSVs. Values below average batch-level diagnostics; they are not pooled percentiles.

| Cell | Original monopoly fraction | CC monopoly fraction | Original singleton | CC singleton | CC mean regret | seg–original cosine | seg–CC cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| DINOV2_VITS14_QUERY__Drishti_GS__A1 | 0.7000 | 0.0000 | 0.0500 | 0.0000 | 0.015624 | 0.027064195286732602 | -0.013373822301900349 |
| DINOV2_VITS14_QUERY__RIM_ONE_r3__A1 | 1.0000 | 0.0000 | 0.3567 | 0.0000 | 0.014045 | 0.017326309979277085 | -0.03238597465071915 |
| UNET_QUERY_128__Drishti_GS__A1 | 1.0000 | 0.0000 | 0.5217 | 0.0000 | 0.081033 | 0.1697328899123312 | 0.09330325092537477 |
| UNET_QUERY_128__RIM_ONE_r3__A1 | 0.9950 | 0.0000 | 0.6600 | 0.0000 | 0.055557 | 0.05729979581648206 | 0.02880076706603555 |

Monopolization is operationally defined here as >6 queries assigned to one class in an image. CC quota checks are exact. A zero gradient produces null cosine, never an invented zero. Full development proceeds regardless of these findings.

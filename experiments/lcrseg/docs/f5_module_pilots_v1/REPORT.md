# F5 module pilots: completed development experiment

Two optimization seeds, two orders; all 32 target students sealed before readout. No independent patient confirmation.

| Arm | Final | Old | Incoming | Forget |
|---|---:|---:|---:|---:|
|F5|0.654109|0.631689|0.698950|0.136757|
|BOUNDARY|0.644737|0.617479|0.699253|0.149650|
|OT_TRIPLET|0.655096|0.633137|0.699014|0.135314|
|LOSS_ADAPTER|0.647305|0.622088|0.697740|0.146018|

All paired seed/order/class deltas, including negative outcomes, are in RESULTS.json.
Paper adaptations and fixed coefficients are in PREREG.md and PLAN.json. No module combination or tuning was run.
Private weights, patient rows, images, labels and runtime configuration remain on NAS.

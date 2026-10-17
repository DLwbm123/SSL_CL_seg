# Siemens source candidate: CONSERVATIVE_U_V1

Training Dice: 85.9621%.
Validation Dice: 54.9072%.
Existing SSL validation Dice: 65.8176%.
Candidate minus existing SSL: -10.9103pp.

The source U objective is confidence>0.95 hard CE averaged over the full grid, without class balancing or Dice. The network, initialization, data split,
augmentation, base PAS admission, LCTX, hierarchy and optimizer remain unchanged.
The fixed schedule has 43599 steps. All updates count
toward the overall130,800-call budget. This is a selected development candidate with
reused validation, not an independent test or cross-paper ranking. Historical B and
this candidate differ in GPU/backend conditions. GitHub delivery pending verification.

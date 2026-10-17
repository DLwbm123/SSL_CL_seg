# Siemens source hard-pseudo candidate

Training Dice: 33.6671%.
Validation Dice: 14.2786%.
Existing SSL validation Dice: 65.8176%.
Candidate minus existing SSL: -51.5390pp.

Only the source U objective changed to class-balanced hard CE plus present-foreground
Dice. The network, initialization, data split, augmentation, confidence/PAS admission,
LCTX, hierarchy, optimizer and 43,600-step schedule are unchanged. All updates count
toward the overall130,800-call budget. This is a selected development candidate with
reused validation, not an independent test or cross-paper ranking. Historical B and
this candidate differ in GPU/backend conditions. GitHub delivery pending verification.

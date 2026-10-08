# V105 target-matched CE control

Compare forward-KL distillation with existing reverse-KL RL against the exact same fold-local analytical target. Reuse V104 WARM actors; 40 branches of 512 updates plus one synthetic update. No native model update, image inference or reward query. Freeze and retain all folds, seeds and readouts. See PREREGISTRATION.json for the complete contract.

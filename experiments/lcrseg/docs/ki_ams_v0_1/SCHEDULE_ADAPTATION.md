# Schedule adaptation: NOT FROZEN

Known MIX source: 20-epoch warm-up, 20-epoch discrete ramp,100 epochs; CE+GT-Dice, soft U targets at temperature1, teacher confidence strictly>0.7, EMA0.99, U noise0.02, complementary rectangle side2/3 and maximum U weight0.5.

Unknown KI side: epoch definition, optimizer, LR policy, successful updates per domain and boundary state. No mapping has been selected. The single-domain Adam/100-epoch schedule is not installed as a default KI schedule. Resolve the exact parent first, then freeze an explicit compatibility mapping before any formal update or evaluation GT access. No continuous step ramp substitution has been made.

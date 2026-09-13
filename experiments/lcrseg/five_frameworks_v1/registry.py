"""Executable arms and explicitly plan-only future contrasts."""
FAMILIES=('F1','F2','F3','F4','F5')
BASELINES=('B0_PARENT_LCTX','B1_PARENT_CONF_KL','B2_PARENT_PAS_KL',
           'B3_PARENT_LCTX_DENSEG','B4_PARENT_PAS_KL_DENSEG_JOINT')
ABLATIONS={
 'F1_NO_JML':{'family':'F1','lambda_JML':0},
 'F1_LOSS_ONLY_JOINT':{'family':'F1','U_permission':'parent_plus_R'},
 'F1_RANDOM_Q':{'family':'F1','Q':'fixed_random_same_rank'},
 'F1_NO_PAS':{'family':'F1','eligibility':'geometry_only'},
 'F2_NO_STRUCTURE':{'family':'F2','lambda_structure':0,'entry_direction':'native'},
 'F2_NO_DIRECTION_PROBE':{'family':'F2','entry_direction':'native'},
 'F2_U_JOINT':{'family':'F2','U_permission':'parent_all'},
 'F2_RANDOM_LEGAL_A':{'family':'F2','entry_direction':'random_legal_frobenius_matched'},
 'F3_KAPPA_ZERO':{'family':'F3','kappa':0},
 'F3_SCALAR_D':{'family':'F3','D':'per_position_mean_over_coordinates'},
 'F3_GRAD_NORM_MATCH':{'family':'F3','D':'scalar_with_final_R_gradient_norm_matching'},
 'F3_ENTROPY_ONLY':{'family':'F3','prototype_uncertainty_mix':0},
 'F4_SHAPE_ZERO':{'family':'F4','lambda_shape':0},
 'F4_STEPS_ZERO':{'family':'F4','inner_steps':0},
 'F4_FULL_COORDINATE':{'family':'F4','Q':'identity_full_feature_space'},
 'F4_NO_TRUST_PROJECTION':{'family':'F4','trust':'disabled_requires_new_review'},
 'F5_NO_SWD':{'family':'F5','lambda_SWD':0,'retain_extra_clean_U':True},
 'F5_NO_CLASS_CONDITION':{'family':'F5','classes':'pooled_foreground'},
 'F5_RANDOM_Q':{'family':'F5','Q':'fixed_random_same_rank'},
 'F5_NO_NORMALIZATION':{'family':'F5','feature_normalization':False}}


def validate_family(family):
    if family not in FAMILIES+BASELINES:raise ValueError('unknown or plan-only arm: '+family)

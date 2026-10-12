"""Three fixed analytic cases; no model, optimizer, data or parameter search."""
import json
import math

def check():
    rows=[]
    for angle in (.1,.125,.2):
        # Unit children cos(a)u +/- sin(a)v; their equal-vote parent is u.
        parent_distance=2*math.sin(angle/2)
        full_gap=(1-math.cos(2*angle))/.1
        residual_gap=2/.1
        full_ce=math.log1p(math.exp(-full_gap))
        residual_ce=math.log1p(math.exp(-residual_gap))
        assert parent_distance<.25 and full_ce>.35 and residual_ce<3e-9
        assert abs(math.cos(angle)**2+math.sin(angle)**2-1)<1e-14
        rows.append(dict(angle_rad=angle,parent_distance=parent_distance,alignment_hinge=0.,
                         full_feature_sibling_ce=full_ce,tangent_sibling_ce=residual_ce))
    bound=2*.25**2-.25**4/2
    return dict(status='ANALYTIC_DESIGN_CHECK_PASS_NOT_NATIVE_QUALIFICATION',cases=rows,
                symmetric_zero_hinge_full_ce_lower_bound=math.log1p(math.exp(-bound/.1)),
                costs=dict(analytic_cases=3,optimizer_updates=0,model_calls=0,autograd_vjps=0,patient_access=0,GPU_calls=0),
                formal_advancement=False)

if __name__=='__main__':print(json.dumps(check(),indent=2))

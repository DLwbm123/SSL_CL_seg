"""One fixed symmetric-sibling comparison; no model/autograd/data or optimization."""
import json,math
from pathlib import Path

def terms(angle):
    # Unit siblings cos(a)u +/- sin(a)v; unit parent u and background -u.
    coarse=(math.log1p(math.exp(-20))+math.log1p(math.exp(-2*math.cos(angle)/.1)))/2
    parts=math.log1p(math.exp(-(1-math.cos(2*angle))/.1))
    alignment=max(0.,2*math.sin(angle/2)-.25)
    return dict(angle=angle,coarse=coarse,parts=parts,alignment=alignment,
                V21=.1*(coarse+parts+alignment),without_alignment=.1*(coarse+parts))

a,b=terms(.4),terms(.8)
assert b['parts']<a['parts'] and max(a['coarse'],b['coarse'])<1e-5
assert b['V21']>a['V21'] and b['without_alignment']<a['without_alignment']
result=dict(status='PASS_ANALYTIC_NOT_NATIVE_QUALIFICATION',formal_advancement=False,
            fixed_comparison=[a,b],optimizer_updates=0,model_calls=0,autograd_calls=0,patient_calls=0,gpu_calls=0,
            conclusion='Alignment penalizes greater sibling separation in this fixed case even when both contrasts improve in total; not patient causality or retention evidence')
Path(__file__).with_name('ANALYTIC_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))

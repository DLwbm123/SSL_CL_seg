"""One intervention: keep the entry U teacher fixed, versus the original EMA."""
import math
from experiments.qprompt_rl_v1.v3e_u_confirm.protocol import SEEDS, DOMAINS, H, decision as compare

WEIGHTS={'ORIGINAL':0.,'FINE_05':.5,'FIXED_05':.5}
OPTIONS={m:dict(freeze_U_teacher=m=='FIXED_05') for m in WEIGHTS}
ACTIVE={'ORIGINAL':0,'FINE_05':1200,'FIXED_05':1200}
CAPS=dict(source=0,main=36000,smoke=80,development=0,controller=0,replay=1804)
PREDECESSOR_COMMIT='baafffbac4c56495af85164c0d4265a6473e9996'


def decision(rows):
    result=compare(rows,'FIXED_05',('FINE_05','ORIGINAL'),WEIGHTS)
    result['teacher_intervention_supported']=result.pop('stable_candidate')
    result['interpretation']='Reused optimization seeds and development validation patients; joint-positive means, >=4/5 joint seeds, no domain-mean regression against full EMA U. Not causal attribution of prior timing result or independent generalization.'
    return result


def check():
    assert len(SEEDS)*len(DOMAINS)*len(WEIGHTS)*H==CAPS['main']
    assert CAPS['replay']==math.ceil(.05*(CAPS['main']+CAPS['smoke']))
    rows=[dict(seed=s,domain=d,method=m,macro_Dice=.7+.01*(m=='FIXED_05'),old_REFUGE=.7+.01*(m=='FIXED_05')) for s in SEEDS for d in DOMAINS for m in WEIGHTS]
    assert decision(rows)['teacher_intervention_supported']
    for r in rows:
        if r['method']=='FIXED_05' and r['domain']=='Drishti_GS':r['macro_Dice']=.69
    assert not decision(rows)['teacher_intervention_supported']
    print('PASS finite matrix and joint two-domain decision')


if __name__=='__main__':check()

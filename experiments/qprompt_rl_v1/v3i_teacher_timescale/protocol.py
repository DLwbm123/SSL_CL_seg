"""One preregistered order-of-magnitude teacher memory intervention."""
import math
from experiments.qprompt_rl_v1.v3e_u_confirm.protocol import SEEDS, DOMAINS, H, decision as compare

WEIGHTS={'ORIGINAL':0.,'FINE_05':.5,'FIXED_05':.5,'SLOW_05':.5}
OPTIONS={m:dict(freeze_U_teacher=m=='FIXED_05',teacher_decay=.999 if m=='SLOW_05' else .99) for m in WEIGHTS}
ACTIVE={'ORIGINAL':0,'FINE_05':1200,'FIXED_05':1200,'SLOW_05':1200}
CAPS=dict(source=0,main=48000,smoke=100,development=0,controller=0,replay=2405)
PREDECESSOR_COMMIT='2d96276f008d2f77471fae3463bb7a76c4cf977b'


def decision(rows):
    result=compare(rows,'SLOW_05',('FINE_05','FIXED_05','ORIGINAL'),WEIGHTS)
    result['timescale_intervention_supported']=result.pop('stable_candidate')
    result['interpretation']='Reused seeds and development validation; primary SLOW-FINE joint-positive means, >=4/5 joint seeds, no domain regression. All comparisons retained; no independent generalization or post-hoc tolerance.'
    return result


def check():
    assert len(SEEDS)*len(DOMAINS)*len(WEIGHTS)*H==CAPS['main']
    assert CAPS['replay']==math.ceil(.05*(CAPS['main']+CAPS['smoke']))
    rows=[dict(seed=s,domain=d,method=m,macro_Dice=.7+.01*(m=='SLOW_05'),old_REFUGE=.7+.01*(m=='SLOW_05')) for s in SEEDS for d in DOMAINS for m in WEIGHTS]
    assert decision(rows)['timescale_intervention_supported']
    for r in rows:
        if r['method']=='SLOW_05' and r['domain']=='Drishti_GS':r['macro_Dice']=.69
    assert not decision(rows)['timescale_intervention_supported']
    print('PASS fixed matrix and joint two-domain decision')


if __name__=='__main__':check()

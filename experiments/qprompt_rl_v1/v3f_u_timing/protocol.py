import math
from experiments.qprompt_rl_v1.v3e_u_confirm.protocol import SEEDS, DOMAINS, H, decision as compare

WEIGHTS={'ORIGINAL':0.,'FINE_05':.5,'EARLY_05':.5,'LATE_05':.5}
WINDOWS={'ORIGINAL':dict(u_start=0,u_stop=1200),'FINE_05':dict(u_start=0,u_stop=1200),
         'EARLY_05':dict(u_start=0,u_stop=600),'LATE_05':dict(u_start=600,u_stop=1200)}
ACTIVE={'ORIGINAL':0,'FINE_05':1200,'EARLY_05':600,'LATE_05':600}
CAPS=dict(source=0,main=48000,smoke=140,development=0,controller=0,replay=2407)
PREDECESSOR_COMMIT='39f840b6d8a805116774149ade31427f3613388c'


def decision(rows):
    result=compare(rows,'LATE_05',('EARLY_05','FINE_05','ORIGINAL'),WEIGHTS)
    result['timing_supported']=result.pop('stable_candidate')
    result['interpretation']='Development mechanism evidence on reused seeds/patients; equal U-call count and nominal coefficient sum, not equal realized loss/gradient or independent generalization.'
    return result


def check():
    assert len(SEEDS)*len(DOMAINS)*len(WEIGHTS)*H==CAPS['main']
    assert CAPS['replay']==math.ceil(.05*(CAPS['main']+CAPS['smoke']))
    for m,w in WEIGHTS.items():
        assert sum(w>0 and WINDOWS[m]['u_start']<=s<WINDOWS[m]['u_stop'] for s in range(H))==ACTIVE[m]
    assert ACTIVE['EARLY_05']==ACTIVE['LATE_05']==600
    rows=[dict(seed=s,domain=d,method=m,macro_Dice=.7+(m=='LATE_05')*.01,old_REFUGE=.7+(m=='LATE_05')*.01) for s in SEEDS for d in DOMAINS for m in WEIGHTS]
    assert decision(rows)['timing_supported']
    print('PASS fixed 40-cell matrix, matched U-call dose, finite budget and primary timing comparison')


if __name__=='__main__':check()

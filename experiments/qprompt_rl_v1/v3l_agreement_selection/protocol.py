"""One finite paired selection intervention; no threshold sweep."""
import math
from experiments.qprompt_rl_v1.v3e_u_confirm.protocol import SEEDS,DOMAINS,H,decision as compare

WEIGHTS={'ORIGINAL':0.,'FINE_05':.5,'FIXED_05':.5,'AGREE_05':.5,'RANDOM_05':.5}
OPTIONS={m:dict(freeze_U_teacher=m not in ('ORIGINAL','FINE_05'),teacher_decay=.99,
               u_selection={'AGREE_05':'agree','RANDOM_05':'random'}.get(m,'none')) for m in WEIGHTS}
ACTIVE={m:0 if m=='ORIGINAL' else H for m in WEIGHTS}
CAPS=dict(source=0,main=60000,smoke=120,development=0,controller=0,replay=3006)
PREDECESSOR_COMMIT='2acdec5fd217bf31585f343befa525198be8f71a'


def decision(rows):
    results={ref:compare(rows,'AGREE_05',(ref,),WEIGHTS) for ref in ('RANDOM_05','FIXED_05','FINE_05','ORIGINAL')}
    return dict(selection_supported=results['RANDOM_05']['stable_candidate'],
                practical_candidate=all(results[r]['stable_candidate'] for r in ('RANDOM_05','FIXED_05','FINE_05')),
                comparisons=results,interpretation='Primary agreement versus exogenous class-count matched random; practical candidate also needs fixed and default teacher joint criteria. Reused seeds/development patients, no independent confirmation.')


def check():
    assert len(SEEDS)*len(DOMAINS)*len(WEIGHTS)*H==CAPS['main']
    assert CAPS['replay']==math.ceil(.05*(CAPS['main']+CAPS['smoke']))
    assert list(WEIGHTS).index('AGREE_05')<list(WEIGHTS).index('RANDOM_05')
    rows=[dict(seed=s,domain=d,method=m,macro_Dice=.7+.01*(m=='AGREE_05'),old_REFUGE=.7+.01*(m=='AGREE_05')) for s in SEEDS for d in DOMAINS for m in WEIGHTS]
    assert decision(rows)['practical_candidate']
    for r in rows:
        if r['method']=='AGREE_05' and r['domain']=='Drishti_GS':r['macro_Dice']=.69
    assert not decision(rows)['selection_supported']
    print('PASS complete 50-endpoint matrix, bounded budget, paired order and joint decision')


if __name__=='__main__':check()

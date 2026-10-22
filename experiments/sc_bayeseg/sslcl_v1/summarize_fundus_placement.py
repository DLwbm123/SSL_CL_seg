"""Frozen single-branch gates plus all four conditional effects."""
import copy,json
from pathlib import Path
from statistics import mean
from summarize_fundus import summarize

p=Path(__file__).parent
base=json.loads((p/'FUNDUS_RESULTS.json').read_text());new=json.loads((p/'FUNDUS_PLACEMENT_RESULTS.json').read_text());runs={**base['runs'],**new['runs']};seeds=(163,164,165);domains=('REFUGE','RIM_ONE_r3')
gates={}
for arm in ('HL','HU'):
    data=copy.deepcopy(base)
    for seed in seeds:data['runs'][f'FH_s{seed}']=runs[f'{arm}_s{seed}']
    a=summarize(data); positive=all(v>0 for v in a['delta_pp']['F0']['REFUGE'])
    gates[arm]=dict(status='PASS' if a['history_gate']=='PASS' and positive else 'FAIL',mean_delta_pp=a['mean_delta_pp']['F0'],all_old_seed_positive=positive,failures=a['failures'])

def flatten(m):
    r=dict(main_Dice_pp=100*m['mean_Dice'],tail_q10_pp=100*m['lower_decile_Dice'])
    for c in ('cup','disc'):
        r[c+'_Dice_pp']=100*m[c]['Dice'];r[c+'_HD95_pixels']=m[c]['HD95_penalized_pixels'];r[c+'_empty_count']=m[c]['empty_prediction_failures']
    return r

effects={}
for seed in seeds:
    effects[str(seed)]={}
    for d in domains:
        vals={a:flatten(runs[f'{a}_s{seed}']['metrics'][d]) for a in ('F0','FH','HL','HU')};e={}
        for a,b in [('HU','F0'),('FH','HL'),('HL','F0'),('FH','HU')]:e[a+'-'+b]={k:vals[a][k]-vals[b][k] for k in vals[a]}
        e['interaction']={k:vals['FH'][k]-vals['HL'][k]-vals['HU'][k]+vals['F0'][k] for k in vals['F0']};effects[str(seed)][d]=e
new['analysis']=dict(gates=gates,conditional_effects=effects,means={a:{d:100*mean(runs[f'{a}_s{s}']['metrics'][d]['mean_Dice'] for s in seeds) for d in domains} for a in ('HL','HU')},style_candidate_closed=all(v['status']=='FAIL' for v in gates.values()))
(p/'FUNDUS_PLACEMENT_RESULTS.json').write_text(json.dumps(new,indent=2)+'\n');print(json.dumps(dict(gates=gates,means=new['analysis']['means']),indent=2))

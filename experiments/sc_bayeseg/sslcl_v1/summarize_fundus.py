"""Apply the predeclared Fundus gates to aggregate results only."""
import json
from pathlib import Path
from statistics import mean


def summarize(data):
    runs=data['runs']; seeds=(163,164,165); domains=('REFUGE','RIM_ONE_r3')
    def m(arm,seed,domain):return runs[f'{arm}_s{seed}']['metrics'][domain]
    delta={base:{d:[100*(m('FH',s,d)['mean_Dice']-m(base,s,d)['mean_Dice']) for s in seeds] for d in domains} for base in ('F0','FG')}
    failures=[]
    if mean(delta['F0']['REFUGE'])<1:failures.append('old_mean_gain_below_1pp')
    if mean(delta['F0']['RIM_ONE_r3'])<-.5:failures.append('new_mean_drop_over_0.5pp')
    for s in seeds:
        for d in domains:
            a,b=m('FH',s,d),m('F0',s,d)
            for c in ('cup','disc'):
                drop=100*(a[c]['Dice']-b[c]['Dice']); ratio=a[c]['HD95_penalized_pixels']/b[c]['HD95_penalized_pixels']
                if drop < -2:failures.append(dict(seed=s,domain=d,metric=c+'_Dice',delta_pp=drop))
                if ratio > 1.1:failures.append(dict(seed=s,domain=d,metric=c+'_HD95',ratio=ratio))
                if a[c]['empty_prediction_failures']>b[c]['empty_prediction_failures']:failures.append(dict(seed=s,domain=d,metric=c+'_empty'))
            tail=100*(a['lower_decile_Dice']-b['lower_decile_Dice'])
            if tail < -2:failures.append(dict(seed=s,domain=d,metric='lower_decile',delta_pp=tail))
    specificity=mean(delta['FG']['REFUGE'])>=.5 and all(v>0 for v in delta['FG']['REFUGE'])
    return dict(mean_Dice_percent={arm:{d:100*mean(m(arm,s,d)['mean_Dice'] for s in seeds) for d in domains} for arm in ('source_U','F0','FG','FH')},delta_pp=delta,mean_delta_pp={b:{d:mean(v) for d,v in vals.items()} for b,vals in delta.items()},history_gate='PASS' if not failures else 'FAIL',specificity_gate='PASS' if specificity else 'FAIL',failures=failures)

if __name__=='__main__':
    p=Path(__file__).parent/'FUNDUS_RESULTS.json';data=json.loads(p.read_text());data['analysis']=summarize(data);p.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data['analysis'],indent=2))

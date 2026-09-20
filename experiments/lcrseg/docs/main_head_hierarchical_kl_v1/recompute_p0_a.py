"""Read only committed public aggregate exports; never reconstruct pixel data."""
import csv
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
SRC=HERE.parent/'agms_observer_v1/results_20260920'

def run():
    final=list(csv.DictReader((SRC/'FINAL_METRICS.csv').open()))
    domain=list(csv.DictReader((SRC/'DOMAIN_METRICS.csv').open()))
    means={arm:{k:sum(float(x[k]) for x in final if x['arm']==arm)/2 for k in ('Final','Old','Incoming','Forget')}
           for arm in sorted({x['arm'] for x in final})}
    index={(x['arm'],x['order'],x['domain']):x for x in domain}
    deltas=[]
    for x in domain:
        for base in ('A0','A5_CONTROL'):
            y=index[base,x['order'],x['domain']]
            deltas.append(dict(arm=x['arm'],baseline=base,order=int(x['order']),domain=x['domain'],
                               **{k:float(x[k])-float(y[k]) for k in ('rim','cup','disc_union','macro_Dice')}))
    diag=json.loads((SRC/'GRADIENT_DIAGNOSTICS.json').read_text())
    rankings=[dict(order=x['order'],step=x['step'],**r) for x in diag for r in x['L_parent_conditional']['ranking']]
    coverage=json.loads((SRC/'COVERAGE_AND_RISK.json').read_text())
    result=dict(status='P0_A_COMPLETE_PUBLIC_AGGREGATES_ONLY',source_commit='f7d28f4b855a3ab04d2003bcde2f5b2b9a960c3c',
        means=means,domain_class_deltas=deltas,ranking_strata=rankings,
        same_state_U=[dict(order=x['order'],**x['U_same_state']) for x in coverage if x['status']=='measured'],
        unavailable=['conditional-entropy ranking','actual fine-PAS-restricted ranking','per-image/patient joint distributions','patient confidence intervals'],
        ranking_note='Existing main_entropy is 3-class entropy; existing ranks do not restrict to actual fine-PAS. New binary-entropy score remains unvalidated.',
        private_reads=0,optimizer_calls=0,model_forwards=0)
    assert len(final)==10 and len(domain)==30 and len(diag)==8 and len(rankings)==48
    assert abs(means['OBS0']['Final']-means['A0']['Final']+0.00005809238949805451)<1e-12
    (HERE/'P0_A_RECOMPUTED.json').write_text(json.dumps(result,indent=2)+'\n')
    print('P0-A PASS: 5 means, 60 domain contrasts, 48 ranking strata, 2 U summaries; no private reads')

if __name__=='__main__':run()

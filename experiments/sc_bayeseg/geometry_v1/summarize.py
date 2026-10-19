"""Verify small convex certificates and export anonymous geometry results."""
import json,csv,sys
from pathlib import Path
import numpy as np


def main(raw,out):
    raw,out=Path(raw),Path(out);out.mkdir(exist_ok=True)
    r=json.loads((raw/'RESULTS.json').read_text());assert r['status']=='COMPLETE'
    assert r['costs']==dict(extra_reverse=20,adam_candidates=4,optimizer=0,candidate_forwards=0,parameter_applications=0)
    assert json.loads((raw/'EXIT.json').read_text())['exit_code']==0 and not (raw/'FAILURE.json').exists()
    for f,n in [('reverse_calls.jsonl',20),('adam_candidates.jsonl',4)]:assert [v['index'] for v in map(json.loads,(raw/f).read_text().splitlines())]==list(range(1,n+1))
    errors=[]
    for sk,s in r['states'].items():
        ge=s['geometry'];K=np.array(ge['all_normalized_gram']);w=np.array(ge['strict']['weights']);z=ge['strict']['combination_norm']
        assert K.shape==(18,18) and np.max(np.abs(K-K.T))<1e-10
        assert abs(w.sum()-1)<1e-7 and w.min()>=-1e-7
        err=max(abs(float(w@K@w)-z*z),float(np.max(np.abs(-K@w/z-np.array(ge['strict']['normalized_derivatives'])))))
        assert err<1e-7 and ge['strict']['worst_normalized_derivative']<-1e-7 and ge['weak']['kkt_pass']
        errors.append(err)
        for q in s['joint']:
            if q['original_G_comparison']:assert q['original_G_comparison']['loss_equal'] and q['original_G_comparison']['semantic_candidate_norm_error']==0
    assert all(r[k] for k in ['source_payloads_unchanged','last_loaded_model_unchanged','all_parameter_grad_buffers_none','source_physical_ledger_unchanged'])
    audit=dict(status='PASS',costs=r['costs'],ledgers_contiguous=True,no_failure=True,exit_code=0,source_state_unchanged=True,no_optimizer_or_candidate_forward=True,shared_constraints_verified=json.loads((raw/'SHARING_AUDIT.json').read_text()),strict_gram_certificate_max_error=max(errors),both_strict_witnesses_verified=True)
    r['runtime_seconds']=r.pop('finished')-r.pop('started')
    for n,o in [('RESULTS',r),('AUDIT',audit),('SOLVER_SELF_CHECK',json.loads((raw/'SOLVER_SELF_CHECK.json').read_text()))]:(out/(n+'.json')).write_text(json.dumps(o,indent=2)+'\n')
    rows=[]
    for sk,s in r['states'].items():
        for name,d in s['directions'].items():
            for i,cn in enumerate(r['constraints']):rows.append(dict(state=sk,direction=name,constraint=cn,raw_dot=d['constraint_raw_dot'][i],per_unit_direction=d['constraint_dot_per_unit_direction'][i],cosine=d['constraint_cosines'][i]))
    with (out/'DIRECTIONS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main(*sys.argv[1:])

"""One receipt/checkpoint audit and paired CPU summary; no model forward."""
import json,collections
from pathlib import Path
import numpy as np
import torch
from .runtime import exact,finite
from ..siemens_lonly_v1.run import compare


def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def paired(a,b):
    result=compare(a,b)
    def patients(v):
        d={}
        for r in v:d.setdefault(r['patient'],[]).append(float(np.mean(r['dice'])))
        assert all(len(x)==2 for x in d.values());return {k:np.mean(x) for k,x in d.items()}
    aa,bb=patients(a),patients(b);assert aa.keys()==bb.keys()
    values=np.array([100*(bb[k]-aa[k]) for k in sorted(aa)]);rng=np.random.default_rng(163)
    result['patient_bootstrap95_pp']=np.quantile(values[rng.integers(len(values),size=(10000,len(values)))].mean(1),[.025,.975]).tolist();return result


def main(cfg):
    root=Path(cfg['root']);old=Path(cfg['comparison_root']);assert not (root/'AUDIT.json').exists()
    assert read(root/'STATUS.json')['status']=='COMPLETE'
    for n in ['LAUNCH.private.json','WORKER.private.json',*[str(p.relative_to(root)) for p in root.glob('ACTIVE_*.private.json')]]:
        d=read(root/n);p=Path('/proc')/str(d['pid'])/'stat';assert not p.exists() or p.read_text().split()[21]!=str(d['birth']),n
    for n in ['EXIT.json','EXIT_qualify.json','EXIT_train.json','EXIT_evaluate.json','jobs/A/EXIT.json','jobs/A/EVALUATION_EXIT.json']:assert read(root/n)['exit_code']==0,n
    counter=read(root/'budget_counter.json');events=rows(root/'budget_events.jsonl')
    actual=collections.Counter(e['stage']+':'+e['phase'] for e in events if e['kind']=='optimizer')
    assert counter==dict(total=2032,optimizer=2032,functional=0,pending=0,by_phase={'A:engineering':32,'A:formal':2000})
    assert dict(actual)==counter['by_phase'] and sum(e['kind']=='reserved' for e in events)==2032
    assert len(rows(root/'jobs/A/physical_updates.jsonl'))==2000
    assert read(root/'qualification/A/ENGINEERING.json')['actual_calls']==32
    load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    initial=load(root/'jobs/A/initial.pt');final=load(root/'jobs/A/final.pt');latest=load(root/'jobs/A/latest.pt');base=load(old/'jobs/A/initial.pt')
    assert final['step']==final['physical_updates']==2000 and not initial['optimizer']['state'];finite(final);assert exact(final,latest)
    assert all(torch.equal(v,base['model'][k]) for k,v in initial['model'].items() if k.startswith('parent.'))
    assert all(torch.equal(v,initial['model'][k]) and torch.equal(v,final['ema'][k]) for k,v in final['model'].items() if not k.startswith('parent.'))
    assert all(torch.equal(v,initial['model'][k]) for k,v in final['model'].items() if k.endswith(('sigma.weight','grad_update')))
    if cfg.get('no_replay'):assert not any(k=='code' or k.startswith('generator.') for k in final['model'])
    metrics=read(root/'jobs/A/RESULTS.json');assert metrics['source_train']['patients']==10 and metrics['target_train']['patients']==13
    comparisons={}
    for domain,count in [('Siemens',23),('Philips',31)]:
        name='private_'+domain+'_val.jsonl';current=rows(root/'jobs/A'/name);reference=rows(old/'jobs/A'/name)
        keys=lambda rr:{(r['patient'],r['phase']) for r in rr}
        assert keys(current)==keys(reference) and len(keys(current))==count*2 and len({r['patient'] for r in current})==count
        ema=rows(root/'jobs/A'/('private_ema_'+domain+'_val.jsonl'));base_ema=rows(old/'jobs/A'/('private_ema_'+domain+'_val.jsonl'));assert keys(ema)==keys(current)
        comparisons[domain]=dict(versus_A=paired(reference,current),EMA_minus_student=paired(current,ema),EMA_versus_A_EMA=paired(base_ema,ema))
    for role,n,v in [('source_train',10,20),('target_train',13,26)]:
        rr=rows(root/'jobs/A'/('private_'+role+'.jsonl'));base_rows=rows(old/'jobs/A'/('private_'+role+'.jsonl'));assert keys(rr)==keys(base_rows) and len(keys(rr))==v and len({x['patient'] for x in rr})==n
    audit=dict(status='PASS',cost=counter,formal_calls=2000,engineering_calls_discarded=32,common_native_entry=True,empty_optimizer=True,frozen_non_native=True,fixed_native_constants=True,final_latest_exact_and_finite=True,patients_EDES_exact=True,workers_retired=True,no_model_forward=True,no_replay=bool(cfg.get('no_replay')))
    (root/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    (root/'PUBLIC_SUMMARY.json').write_text(json.dumps(dict(audit=audit,metrics=metrics,paired=comparisons,primary='fixed final student',single_seed_exploratory=True),ensure_ascii=False,indent=2)+'\n')

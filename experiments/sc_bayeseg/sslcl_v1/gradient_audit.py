"""Zero-update local gradient alignment, using only target training L/U."""
import json,os,time
from pathlib import Path
import numpy as np
import torch
from .data import Schedule,view
from .model import initialize,supervised
from .train import setup
from sc_bayeseg.window_v1.r_start import write

def run(s):
    setup();out=Path(s['output']);out.mkdir(exist_ok=False);m=initialize(s,'U');t=initialize(s,'U')
    ck=torch.load(s['checkpoint'],map_location='cpu',weights_only=False);assert ck['family']=='U' and ck['step']==5000
    m.load_state_dict(ck['model']);t.load_state_dict(ck['teacher']);m.eval();t.eval().requires_grad_(False)
    params=[p for p in m.parameters() if p.requires_grad];L=Schedule(s['data'],'BMC',seed=s['schedule_seed'],dataset='prostate');U=Schedule(s['data'],'BMC',True,seed=s['schedule_seed'],dataset='prostate')
    current=s['styles']['BMC'];historical=s['styles']['RUNMC'];rows={k:[] for k in ('ORIG_NONE','ORIG_HISTORY','SAME_NONE','SAME_HISTORY')}
    for batch in range(s.get('audit_batches',32)):
        assert time.time()-s['original_started_unix']<85200
        step=1001+batch;x,y,_=L.batch(5000+step);u,uy,_=U.batch(5000+step);assert uy is None
        sup=(supervised(m(x)[0],y)+supervised(m(view(x,current,None,step,s['schedule_seed']))[0],y))/2
        gs=torch.cat([g.reshape(-1) for g in torch.autograd.grad(sup,params)]);sn=gs.norm()
        for same in (False,True):
            for history in (False,True):
                z=view(u,current,historical if history else None,500000+step,s['schedule_seed'])
                with torch.no_grad():q=t.probability(z if same else u).clamp_min(1e-8)
                p=m.probability(z).clamp_min(1e-8);kl=(q*(q.log()-p.log())).sum(1).mean()
                gu=torch.cat([g.reshape(-1) for g in torch.autograd.grad(kl,params)]);un=gu.norm();dot=(gs*gu).sum()
                assert torch.isfinite(gs).all() and torch.isfinite(gu).all()
                row=dict(supervised_loss=float(sup.detach()),consistency_loss=float(kl.detach()),supervised_norm=float(sn),consistency_norm=float(un),dot=float(dot),cosine=float(dot/(sn*un).clamp_min(1e-12)),norm_ratio=float(un/sn.clamp_min(1e-12)))
                rows[('SAME' if same else 'ORIG')+('_HISTORY' if history else '_NONE')].append(row)
    result={k:{metric:dict(mean=float(np.mean(a:=[r[metric] for r in v])),median=float(np.median(a)),min=float(min(a)),max=float(max(a))) for metric in v[0]} for k,v in rows.items()}
    for k,v in rows.items():result[k]['negative_dot_batches']=sum(r['dot']<0 for r in v)
    differences={teacher:{metric:float(np.mean([a[metric]-b[metric] for a,b in zip(rows[teacher+'_HISTORY'],rows[teacher+'_NONE'])])) for metric in ('dot','cosine','norm_ratio','consistency_loss')} for teacher in ('ORIG','SAME')}
    assert all(p.grad is None for p in params) and all(p.grad is None for p in t.parameters())
    write(out/'METRICS.json',dict(status='COMPLETE',optimizer_updates=0,batches=s.get('audit_batches',32),metrics=result,history_minus_none=differences,caveat='Local gradients in eval mode, no optimizer or parameter updates; target training L GT and U images only. Cosine is not accuracy or a causal training trajectory.'))
    write(out/'EXIT.json',dict(exit_code=0))
if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))

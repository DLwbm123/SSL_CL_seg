"""Binary dataset integration and metric checks; no validation labels opened."""
import copy,json,os
from pathlib import Path
import numpy as np
import torch
from .data import Schedule
from .model import initialize
from .train import setup,optimizer,step_loss,ema,aggregate
from .prostate_metrics import binary_phase_metrics
from lcrseg.engine.checkpoint import capture_rng_state,restore_rng_state
from sc_bayeseg.window_v1.r_start import write,save

def run(s):
    setup();root=Path(s['root']);assert json.loads((root/'DATA_AUDIT.json').read_text())['status']=='PASS'
    y0=np.zeros((3,8,8),dtype=np.int64);y0[1,2:5,2:5]=1
    p=torch.nn.functional.one_hot(torch.from_numpy(y0),2).permute(0,3,1,2).float()
    perfect=binary_phase_metrics(p,y0,[3.,1.,2.]);assert perfect['classes'][0]['Dice']==1 and perfect['classes'][0]['HD95_mm']==0
    shifted=binary_phase_metrics(p.roll(1,-1),y0,[3.,1.,2.]);assert abs(shifted['classes'][0]['HD95_mm']-2)<1e-6
    empty=p.clone();empty[:,0]=1;empty[:,1]=0;bad=binary_phase_metrics(empty,y0,[3.,1.,2.]);assert bad['classes'][0]['empty_prediction_failure']
    a=aggregate([dict(patient='synthetic',metrics=perfect)],2,1);assert a['mean_Dice']==1 and len(a['class_Dice'])==1
    x,y,_=Schedule(s['data'],'RUNMC',dataset='prostate').batch(1)
    ds=Schedule(s['data'],'RUNMC',True,dataset='prostate');u,uy,_=ds.batch(1)
    assert uy is None and all(not r['label_h5_relpath'] for r in ds.ds.records)
    assert x.shape==(2,1,128,128) and y.shape==(2,128,128) and set(y.unique().tolist())<={0,1}
    details={};initial=None
    for family in ('B','U'):
        m=initialize(s,family)
        if initial is None:initial={k:v.detach().cpu().clone() for k,v in m.backbone.state_dict().items()}
        else:assert all(torch.equal(v.cpu(),initial[k]) for k,v in m.backbone.state_dict().items())
        t=copy.deepcopy(m).eval().requires_grad_(False);opt=optimizer(m)
        loss,parts=step_loss(m,t,x,y,u,s['styles']['RUNMC'],None,1)
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
        assert all(p.grad is None for p in t.parameters())
        old={k:v.clone() for k,v in t.state_dict().items()}
        torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.);opt.step();ema(t,m)
        for k,v in t.state_dict().items():
            if v.is_floating_point():assert torch.allclose(v,old[k]*.99+m.state_dict()[k]*.01,atol=2e-6)
        ck=dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=capture_rng_state());save(root/f'integration_{family}.pt',ck)
        loaded=torch.load(root/f'integration_{family}.pt',map_location='cpu',weights_only=False);other=initialize(s,family);other.load_state_dict(loaded['model']);oo=optimizer(other);oo.load_state_dict(loaded['optimizer']);assert all(int(v['step'])==1 for v in oo.state.values())
        m.eval();other.eval();restore_rng_state(loaded['rng'])
        with torch.no_grad():before=m.probability(x)
        restore_rng_state(loaded['rng'])
        with torch.no_grad():after=other.probability(x)
        assert before.shape==(2,2,128,128) and torch.equal(before,after)
        details[family]=dict(loss=float(loss.detach()),parts=parts,parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),optimizer_updates=1,checkpoint_prediction_exact=True)
    write(root/'INTEGRATION.json',dict(status='PASS',optimizer_updates=2,shared_initial_backbone_exact=True,metrics_perfect_empty_spacing_pass=True,teacher_detached_EMA_pass=True,val_test_labels_opened=0,details=details))
if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))

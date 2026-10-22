"""One real update per view arm plus RGB/metric contracts, not scientific evidence."""
import copy
import json
import os
from pathlib import Path
import numpy as np
import torch
from .data import Schedule, view
from .fundus import image_metrics, aggregate_fundus, fov
from .model import initialize
from .train import setup, optimizer, step_loss, ema
from lcrseg.engine.checkpoint import capture_rng_state, restore_rng_state
from sc_bayeseg.window_v1.r_start import write


def run(s):
    setup(); root = Path(s['root'])
    assert json.loads((root/'DATA_AUDIT.json').read_text())['status'] == 'PASS'
    y = np.zeros((16,16), dtype=np.int64); y[3:13,3:13]=1; y[6:10,6:10]=2
    p = torch.nn.functional.one_hot(torch.from_numpy(y),3).permute(2,0,1).float()
    perfect = image_metrics(p,y)
    assert all(perfect[n]['Dice']==1 and perfect[n]['HD95_pixels']==0 for n in ('cup','disc','rim'))
    empty = p.clone(); empty[:]=0; empty[0]=1
    bad=image_metrics(empty,y);assert all(bad[n]['empty_prediction_failure'] for n in ('cup','disc'))
    result,_=aggregate_fundus([dict(patient='synthetic',metrics=perfect)])
    assert result['mean_Dice']==1 and result['lower_decile_Dice']==1
    x,y,_=Schedule(s['data'],'RIM_ONE_r3',dataset='fundus',image_size=384).batch(1)
    un=Schedule(s['data'],'RIM_ONE_r3',True,dataset='fundus',image_size=384)
    u,uy,_=un.batch(1);assert uy is None and all(not r['label_h5_relpath'] for r in un.ds.records)
    assert x.shape==(2,3,384,384) and y.shape==(2,384,384)
    current=s['styles']['RIM_ONE_r3'];history=s['styles']['REFUGE'];details={}
    for name,hist,mode in [('F0',None,'history'),('FG',None,'wide'),('FH',history,'history')]:
        changed=view(x,current,hist,1,163,mode)
        assert torch.equal(changed[~fov(x).expand_as(x)],x[~fov(x).expand_as(x)])
        assert changed.min()>=0 and changed.max()<=1 and torch.isfinite(changed).all()
        m=initialize(s,'U');t=copy.deepcopy(m).eval().requires_grad_(False);opt=optimizer(m)
        loss,parts=step_loss(m,t,x,y,u,current,hist,1,style_mode=mode)
        loss.backward();assert torch.isfinite(loss) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters())
        assert all(v.grad is None for v in t.parameters())
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();ema(t,m)
        state=capture_rng_state();other=initialize(s,'U');other.load_state_dict(m.state_dict());fresh=optimizer(other)
        assert not fresh.state
        restore_rng_state(state);m.eval();other.eval()
        with torch.no_grad(): assert torch.equal(m.probability(x),other.probability(x))
        details[name]=dict(loss=float(loss),optimizer_updates=1,parts=parts)
        del m,t,opt,other,fresh,loss;torch.cuda.empty_cache()
    write(root/'INTEGRATION.json',dict(status='PASS',real_optimizer_updates=3,synthetic_metrics_pass=True,RGB_shape=True,fov_preserved=True,EMA_detached=True,fresh_optimizer=True,details=details,val_test_labels_opened=0))

if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))

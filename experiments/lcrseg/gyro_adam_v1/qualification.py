"""Charged two-case qualification including all Adam state and RNG/read restoration."""
import copy
from pathlib import Path
import torch
from . import core
from .run import main
from ..philips_transfer_v1.run import snapshot,restore
from ..gyro_strength_v1.run import guarded_saver
from ..main_style_ssl_v1.runtime import write,append
from ..siemens_ssl_reproduction_v1.methods import pixel_ce

def assert_same(expected,model,ema,opt,data):
    current=snapshot(model,ema,opt,data,16)
    for key in ['model','ema']:
        assert all(torch.equal(v,current[key][k]) for k,v in expected[key].items())
    assert expected['optimizer']['param_groups']==current['optimizer']['param_groups']
    assert expected['optimizer']['state'].keys()==current['optimizer']['state'].keys()
    for k,a in expected['optimizer']['state'].items():
        b=current['optimizer']['state'][k];assert set(a)==set(b)=={'step','exp_avg','exp_avg_sq'}
        assert all(torch.equal(v,b[n]) for n,v in a.items())
    assert expected['torch_rng'].equal(current['torch_rng'])
    assert all(a.equal(b) for a,b in zip(expected['cuda_rng'],current['cuda_rng']))
    assert expected['reads']==current['reads']

def qualify(cfg):
    main(dict(cfg,preflight_only=True))
    out=Path(cfg['output']);source,model,ema,opt=core.setup(cfg);data=core.provider(cfg);data.arm=cfg['arm']
    initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False);restore(initial,model,ema,opt,data)
    x,y,_=data.labeled(69);calls=0
    def guard(*_):
        if calls>=32:raise PermissionError('engineering cap32')
    def charge(*_):
        nonlocal calls
        calls+=1;append(out/'engineering_updates.jsonl',dict(physical_call=calls,optimizer_executed=True))
    opt.register_step_pre_hook(guard);opt.register_step_post_hook(charge)
    def objective():return pixel_ce(model(x),y).mean()
    def step():
        opt.zero_grad(set_to_none=True);v=objective();v.backward();opt.step();core.update_ema(ema,model);return float(v.detach())
    try:
        first=float(objective().detach())
        for i in range(30):
            if i==15:saved=copy.deepcopy(snapshot(model,ema,opt,data,15))
            v=step()
            if i==15:
                expected=copy.deepcopy(snapshot(model,ema,opt,data,16))
                assert expected['optimizer']['state'] and any(v['exp_avg'].norm()>0 for v in expected['optimizer']['state'].values())
                restore(saved,model,ema,opt,data);assert step()==v
                assert_same(expected,model,ema,opt,data)
        final=float(objective().detach());write(out/'ENGINEERING_MEASUREMENT.json',dict(first=first,final=final,actual_calls=calls))
        assert final<first-1e-4,('two-case learning gate',first,final)
        restore(initial,model,ema,opt,data);assert not opt.state
        opt.zero_grad(set_to_none=True);l,u,_=core.losses(cfg['arm'],model,data,0);(l+u).backward();opt.step();core.update_ema(ema,model)
        state=snapshot(model,ema,opt,data,1);state['engineering_actual_calls']=calls;guarded_saver(cfg)(state,out/'engineering_terminal.pt')
        assert calls==32 and all(torch.equal(v.cpu(),initial['model'][k]) for k,v in model.state_dict().items() if k.startswith('projector.') or k=='frames')
        assert all(torch.isfinite(p).all() for p in model.parameters())
        write(out/'ENGINEERING.json',dict(status='PASS',actual_calls=32,discarded=True,first=first,final=final,
          nonempty_momentum_resume_exact=True,adam_all_state_resume_exact=True,RNG_reads_resume_exact=True,full_target_from_entry=True,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    except BaseException as exc:
        write(out/'ENGINEERING_FAILURE.json',dict(status='FAILED',actual_calls=calls,error=str(exc)));raise

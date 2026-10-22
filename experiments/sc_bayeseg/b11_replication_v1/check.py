"""Check historical default reuse against current code on both real source states."""
import copy
import importlib
import json
import os
import sys
import types
from pathlib import Path
import torch
from lcrseg.engine.checkpoint import restore_rng_state
from sc_bayeseg.sslcl_v1 import train as current
from sc_bayeseg.sslcl_v1.model import Model
from sc_bayeseg.sslcl_v1.data import Schedule


def run(s):
    current.setup()
    package = types.ModuleType('_legacy_b11'); package.__path__ = [s['legacy_package']]
    sys.modules[package.__name__] = package
    old = importlib.import_module('_legacy_b11.train')
    legacy_model = importlib.import_module('_legacy_b11.model')
    rows = {}
    for seed in (164, 165):
        d = Path(s['default_paths'][str(seed)])
        context = json.loads((d/'CONTEXT.json').read_text())
        assert context['family']=='B' and context['stage']=='target' and context['use_u'] and context['history']
        assert context['initialization_seed']==seed and context['schedule_seed']==seed and 'training_seed' not in context
        assert context['source']==s['source_paths']['A'+str(seed)]==s['source_paths']['T'+str(seed)]
        assert context['dataset']=='prostate' and context['domain']=='BMC' and context['data']==s['data']
        for key, value in dict(learning_rate=3e-4, consistency_weight=1., fixed_jl=None, ju_weight=.005, image_size=128, steps=5000, evaluation_seed=620163, history_strength=1., ema_decay=.99, ramp_steps=1000).items():
            assert context.get(key,value)==value, key
        assert json.loads((d/'RESULTS.json').read_text())['updates']==5000
        assert json.loads((d/'EXIT.json').read_text())['exit_code']==0
        ck = torch.load(context['source'], map_location='cpu', weights_only=False)
        assert ck['step']==5000 and ck['family']=='B'
        a=legacy_model.Model('B',2).cuda();b=Model('B',2).cuda()
        a.load_state_dict(ck['model']);b.load_state_dict(ck['model'])
        ta=copy.deepcopy(a).eval().requires_grad_(False);tb=copy.deepcopy(b).eval().requires_grad_(False)
        ta.load_state_dict(ck['teacher']);tb.load_state_dict(ck['teacher'])
        old_batches=[old.Schedule(s['data'],'BMC',u,seed=seed,dataset='prostate').batch(5001) for u in (False,True)]
        batches=[Schedule(s['data'],'BMC',u,seed=seed,dataset='prostate').batch(5001) for u in (False,True)]
        for x,y in zip(old_batches,batches):
            assert torch.equal(x[0],y[0]) and x[2]==y[2]
            if x[1] is not None:assert torch.equal(x[1],y[1])
        x,y,_=batches[0];u=batches[1][0];losses=[];grads=[]
        for model,teacher,fn in [(a,ta,old.step_loss),(b,tb,current.step_loss)]:
            restore_rng_state(ck['rng'])
            loss,_=fn(model,teacher,x,y,u,s['styles']['BMC'],s['styles']['RUNMC'],1,seed)
            loss.backward();losses.append(loss.detach().clone());grads.append({n:p.grad.clone() for n,p in model.named_parameters() if p.requires_grad})
        assert torch.equal(*losses) and all(torch.equal(grads[0][n],grads[1][n]) for n in grads[0])
        assert all(torch.isfinite(v).all()for v in grads[1].values())
        ao=old.optimizer(a);bo=current.optimizer(b)
        assert {k:v for k,v in ao.param_groups[0].items()if k!='params'}=={k:v for k,v in bo.param_groups[0].items()if k!='params'}
        rows[str(seed)]=dict(status='PASS',source_step=5000,default_updates=5000,real_loss=float(losses[0]),data_schedule_exact=True,loss_and_all_gradients_exact=True,default_optimizer_equal=True,full_5000_step_replay=False)
        del a,b,ta,tb,ck,grads,ao,bo
    result=dict(status='PASS',seeds=rows,optimizer_updates=0,backwards=4,source_independently_trained=True)
    (Path(s['root'])/'INTEGRATION.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))

if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))

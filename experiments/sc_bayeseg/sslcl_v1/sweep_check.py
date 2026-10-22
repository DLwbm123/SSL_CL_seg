"""Real-batch effect checks plus optimizer/EMA/clipping controls; zero updates."""
import copy,json,os
from pathlib import Path
import torch
from . import hyperparameter_check
from .train import setup,step_loss,optimizer,ema
from .model import initialize
from .data import Schedule
from lcrseg.engine.checkpoint import capture_rng_state,restore_rng_state
from sc_bayeseg.window_v1.r_start import write


def run(s):
    hyperparameter_check.run(s)
    out=Path(s['root'])/'INTEGRATION.json';r=json.loads(out.read_text());r['status']='CHECKING';write(out,r)
    setup();m=initialize(s,'U');t=copy.deepcopy(m).eval().requires_grad_(False)
    ck=torch.load(s['check_source'],map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);t.load_state_dict(ck['teacher'])
    rng=capture_rng_state();x,y,_=Schedule(s['data'],s['target_domain'],dataset='fundus',image_size=384).batch(6001)
    u=Schedule(s['data'],s['target_domain'],True,dataset='fundus',image_size=384).batch(6001)[0]
    counts=dict(model_forward_calls=0,image_forwards=0)
    def hook(module,args):counts['model_forward_calls']+=1;counts['image_forwards']+=len(args[0])
    handles=[v.register_forward_pre_hook(hook) for v in (m,t)]
    def trial(options):
        restore_rng_state(rng);m.zero_grad(set_to_none=True)
        loss,_=step_loss(m,t,x,y,u,s['styles'][s['target_domain']],None,500,**options);loss.backward()
        return float(loss.detach()),[p.grad.detach().clone() for p in m.parameters()]
    loss0,g0=trial({});effects={}
    for key,values in dict(ramp_steps=[100,5000],dice_weight=[0.,2.],augmentation_strength=[0.,.5],strong_supervision_weight=[0.,1.]).items():
        for v in values:
            loss,grads=trial({key:v});changed=any(not torch.equal(a,b) for a,b in zip(g0,grads));assert changed
            effects[f'{key}={v}']=dict(loss=loss,gradient_changed=changed)
    for h in handles:h.remove()
    for options in s['task_options'].values():
        opt=optimizer(m,options.get('learning_rate',3e-4),options.get('weight_decay',0),(options.get('beta1',.9),options.get('beta2',.999)),options.get('adam_eps',1e-8))
        assert not opt.state and opt.param_groups[0]['eps']==options.get('adam_eps',1e-8)
    for decay in [.9,.99,.999]:
        a=torch.nn.Linear(1,1,bias=False);b=copy.deepcopy(a)
        with torch.no_grad():a.weight.fill_(2);b.weight.fill_(4)
        ema(a,b,decay);assert abs(a.weight.item()-(2*decay+4*(1-decay)))<1e-6
    for clip in [.1,1.,10.]:
        a=torch.nn.Parameter(torch.ones(2));a.grad=torch.tensor([30.,40.]);torch.nn.utils.clip_grad_norm_([a],clip);assert abs(a.grad.norm().item()-clip)<1e-5
    for size in [1,2,4]:
        batch=Schedule(s['data'],s['target_domain'],dataset='fundus',image_size=384,batch_size=size).batch(6001);assert len(batch[0])==size
    r.update(status='PASS',effects=effects,optimizer_ema_clip_batch_checks=True)
    for k,v in counts.items():r[k]+=v
    write(out,r)
if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))

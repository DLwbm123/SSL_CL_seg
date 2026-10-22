"""Verify actual optimizer configuration, default parity, and lambda effect; no updates."""
import copy,importlib.util,json,os,time
from pathlib import Path
import torch
from .data import Schedule
from .model import initialize
from .train import setup,optimizer,step_loss
from lcrseg.engine.checkpoint import capture_rng_state,restore_rng_state
from sc_bayeseg.window_v1.r_start import write


def run(s):
    setup();out=Path(s['root']);spec=importlib.util.spec_from_file_location('sc_bayeseg.sslcl_v1._legacy_train',s['legacy_train']);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    m=initialize(s,'U');t=copy.deepcopy(m).eval().requires_grad_(False)
    ck=torch.load(s['check_source'],map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);t.load_state_dict(ck['teacher']);del ck
    initial=copy.deepcopy(m.state_dict());teacher=copy.deepcopy(t.state_dict());rng=capture_rng_state()
    x,y,_=Schedule(s['data'],s['target_domain'],dataset='fundus',image_size=384).batch(1001)
    u=Schedule(s['data'],s['target_domain'],True,dataset='fundus',image_size=384).batch(1001)[0]
    count=dict(model_forward_calls=0,image_forwards=0)
    def hook(module,args):count['model_forward_calls']+=1;count['image_forwards']+=len(args[0])
    handles=[model.register_forward_pre_hook(hook) for model in (m,t)]
    def trial(fn,mode,hist,weight):
        m.load_state_dict(initial);t.load_state_dict(teacher);m.zero_grad(set_to_none=True);restore_rng_state(rng)
        loss,parts=fn(m,t,x,y,u,s['styles'][s['target_domain']],hist,500,style_mode=mode,consistency_weight=weight)
        loss.backward();return float(loss.detach()),parts,[p.grad.detach().clone() for p in m.parameters()]
    rows={}
    for arm,mode,hist in [('F0','history',None),('FG','wide',None),('FH','history',s['styles'][s['source_domain']])]:
        a,pa,ga=trial(old.step_loss,mode,hist,1.);b,pb,gb=trial(step_loss,mode,hist,1.)
        assert a==b and all(torch.equal(v,w) for v,w in zip(ga,gb))
        c,pc,gc=trial(step_loss,mode,hist,.1);expected=.9*.5*pb['soft_KL']
        assert abs((b-c)-expected)<=2e-6*max(1,abs(b),abs(c)) and expected>1e-8
        assert any(not torch.equal(v,w) for v,w in zip(gb,gc))
        rows[arm]=dict(default_loss_exact=True,default_gradients_exact=True,lambda_loss_difference=b-c,expected_difference=expected,lambda_gradient_changed=True)
    rates={}
    for lr in (1e-5,3e-5,1e-4,3e-4):
        opt=optimizer(m,lr);rates[str(lr)]=[g['lr'] for g in opt.param_groups];assert all(v==lr for v in rates[str(lr)]) and not opt.state
    for h in handles:h.remove()
    write(out/'INTEGRATION.json',dict(status='PASS',optimizer_updates=0,legacy_default_parity=rows,actual_optimizer_lrs=rates,**count,original_started_unix=s['original_started_unix']))
if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))

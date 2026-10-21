"""Real-data fixed-JL loss-routing check, with no optimizer updates."""
import copy,json,os
from pathlib import Path
import torch
from .data import Schedule
from .model import initialize
from .train import setup,step_loss
from sc_bayeseg.window_v1.r_start import write

def run(s):
    setup();x,y,_=Schedule(s['data'],'Philips',seed=164).batch(5001)
    u,uy,_=Schedule(s['data'],'Philips',True,seed=164).batch(5001);assert uy is None
    ck=torch.load(s['source_paths']['B'],map_location='cpu',weights_only=False)
    m=initialize(s,'B');m.load_state_dict(ck['model']);t=copy.deepcopy(m).eval().requires_grad_(False);t.load_state_dict(ck['teacher'])
    args=(m,t,x,y,u,s['styles']['Philips'],s['styles']['Siemens'],1000,164)
    torch.manual_seed(164);old,_=step_loss(*args);old.backward()
    old_grad={n:p.grad.detach().clone() for n,p in m.named_parameters() if p.requires_grad}
    rows={}
    for j,c in ((0,0),(1,0),(0,1),(1,1)):
        torch.manual_seed(164);m.zero_grad(set_to_none=True)
        loss,parts=step_loss(*args,fixed_jl=.005,ju_weight=.005*j,consistency_weight=c)
        expected=parts['supervised']+.005*parts['labeled_image_objective']+.005*j*parts['unlabeled_image_objective']+c*parts['soft_KL']
        assert abs(float(loss.detach())-expected)<1e-5
        loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
        assert all(p.grad is None for p in t.parameters())
        if j==c==1:
            assert torch.allclose(loss,old,atol=1e-6,rtol=1e-6)
            assert all(torch.allclose(p.grad,old_grad[n],atol=2e-6,rtol=1e-5) for n,p in m.named_parameters() if p.requires_grad)
        rows[f'J{j}C{c}']=dict(loss=float(loss.detach()),parts=parts)
    write(Path(s['root'])/'INTEGRATION.json',dict(status='PASS',optimizer_updates=0,backward_checks=5,legacy_full_loss_gradient_close=True,arms=rows))
if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))

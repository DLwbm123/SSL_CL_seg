"""Default parity and channel/class likelihood checks; no optimizer steps."""
import importlib.util,json,os
from pathlib import Path
import torch
from sc_bayeseg.model import Model
from sc_bayeseg.window_v1.r_start import write

def run(s):
    torch.set_num_threads(2)
    spec=importlib.util.spec_from_file_location('legacy',s['legacy_model']);legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    torch.manual_seed(163);old=legacy.Model(size=16);new=Model(size=16);new.load_state_dict(old.state_dict());x=torch.randn(2,1,16,16);y=torch.randint(0,4,(2,16,16));outputs=[]
    for model in [old,new]:
        torch.manual_seed(800);model.draw();v,_,_=model.bound(x,y);loss=-v.mean()+model.kl();loss.backward();outputs.append((loss.detach(),[p.grad for p in model.parameters()]));model.clear_draw()
    assert torch.equal(outputs[0][0],outputs[1][0]) and all(torch.equal(a,b) for a,b in zip(outputs[0][1],outputs[1][1]))
    for c,k in [(1,4),(1,2),(3,3)]:
        model=Model(size=16,num_classes=k,input_channels=c);x=torch.randn(2,c,16,16);y=torch.randint(0,k,(2,16,16));model.draw();bound,_,_=model.bound(x,y);(-bound.mean()+model.kl()).backward();assert torch.isfinite(bound).all() and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters());model.clear_draw()
        p,_=model.prediction(x,refine=1,parameter_samples=1,local_samples=2);assert p.shape==(2,k,16,16) and torch.allclose(p.sum(1),torch.ones_like(p[:,0]),atol=1e-6)
        model.set_prior(True)
        for layer in model.bayes_layers():
            for kind in ['mean','logsd']:
                for part in ['w','b']:assert torch.equal(getattr(layer,'prior_'+kind+'_'+part),getattr(layer,kind+'_'+part))
    write(Path(s['root'])/'INTEGRATION.json',dict(status='PASS',default_scalar_and_all_gradients_exact=True,three_channel_class_shapes=True,normalized_predictions=True,posterior_transfer_exact=True,optimizer_updates=0,synthetic_only=True))
if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))

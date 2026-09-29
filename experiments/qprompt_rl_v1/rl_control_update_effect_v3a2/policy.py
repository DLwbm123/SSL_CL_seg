"""Frozen same-capacity learners; training-only normalization and stable scores."""
import numpy as np
import torch
from torch import nn
from .runtime import stable
PRIOR=np.array([.05,.05,.90]);FLAGS=[20,21,26,27,32,33,38,39]
class Policy(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(40,32),nn.Tanh(),nn.Linear(32,3));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias);self.register_buffer('prior_log',torch.tensor(PRIOR,dtype=torch.float32).log())
    def forward(self,z):return self.net(z)+self.prior_log

def objective(logits,reward):
    lp=logits.log_softmax(-1);p=lp.exp();return -(p*reward.detach()).sum(-1).mean()+.001*(p*(lp-logits.new_tensor(PRIOR).log())).sum(-1).mean()
def actions(scores):
    scores=np.atleast_2d(scores);assert np.isfinite(scores).all();return np.array([next(a for a in (2,1,0) if row.max()-row[a]<=1e-8) for row in scores])
def probabilities(scores):
    scores=np.atleast_2d(scores);v=np.exp(scores-scores.max(1,keepdims=True));return v/v.sum(1,keepdims=True)
def normalize_fit(x,base=False):
    x=np.asarray(x,float).copy()
    if base:x[:,16:]=0
    mu=x.mean(0);std=x.std(0);constant=std==0;sd=np.maximum(std,1e-8);mu[FLAGS]=0;sd[FLAGS]=1;constant[FLAGS]=False
    return dict(mean=mu.tolist(),std=sd.tolist(),constant=constant.tolist(),base=base,train_min=x.min(0).tolist(),train_max=x.max(0).tolist())
def normalize(x,s):
    x=np.asarray(x,float).copy()
    if s['base']:x[:,16:]=0
    x=(x-np.array(s['mean']))/np.array(s['std']);x[:,s['constant']]=0;assert np.isfinite(x).all();return x

def neural(x,d,seed,ledger,key,shuffle=False,synthetic=False):
    torch.manual_seed(seed);model=Policy();opt=torch.optim.Adam(model.parameters(),lr=.003,betas=(.9,.999),eps=1e-8,weight_decay=0);x=torch.tensor(x,dtype=torch.float32);d=torch.tensor(d,dtype=torch.float32);permutation=list(range(len(d)))
    if shuffle:permutation=torch.randperm(len(d),generator=torch.Generator().manual_seed(stable(seed,'shuffle'))).tolist();d=d[permutation]
    initial=model(x).detach().softmax(-1).mean(0).tolist()
    for step in range(256):
        opt.zero_grad(set_to_none=True);loss=objective(model(x),d);assert torch.isfinite(loss);loss.backward();ledger.step(opt,'controller_synthetic' if synthetic else 'controller',key+'/'+str(step))
    return model,dict(initial_probability=initial,final_probability=model(x).detach().softmax(-1).mean(0).tolist(),permutation=permutation,steps=256)

def fit(method,x,d,seed,ledger,key):
    if method=='NC_OPT':return dict(method=method,scores=(np.log(PRIOR)+d.mean(0)/.001).tolist())
    scaler=normalize_fit(x,method.startswith('BASE'));z=normalize(x,scaler);result=dict(method=method,scaler=scaler)
    if method.endswith('RIDGE'):
        z=np.column_stack((z,np.ones(len(z))));w=np.linalg.solve(z.T@z+np.diag([.01*len(z)*3]*40+[0]),z.T@d);w[:,2]=0;result['weights']=w.tolist()
    else:
        model,diag=neural(z,d,seed,ledger,key,method=='EFFECT_SHUFFLE');result.update(state={k:v.tolist() for k,v in model.state_dict().items()},diagnostics=diag)
    return result

def predict(fit,x):
    method=fit['method'];x=np.atleast_2d(x)
    if method=='FINE':return dict(actions=[2]*len(x),probabilities=np.tile([0.,0.,1.],(len(x),1)).tolist(),scores=np.tile([0.,0.,1.],(len(x),1)).tolist())
    if method=='EFFECT_RULE':scores=np.column_stack((x[:,16]+x[:,22],x[:,28]+x[:,34],np.zeros(len(x))))
    elif method=='NC_OPT':scores=np.tile(fit['scores'],(len(x),1))
    elif method.endswith('RIDGE'):scores=np.log(PRIOR)+np.column_stack((normalize(x,fit['scaler']),np.ones(len(x))))@np.array(fit['weights'])/.001
    else:
        model=Policy();model.load_state_dict({k:torch.tensor(v) for k,v in fit['state'].items()});scores=model(torch.tensor(normalize(x,fit['scaler']),dtype=torch.float32)).detach().numpy()
    return dict(actions=actions(scores).tolist(),probabilities=probabilities(scores).tolist(),scores=scores.tolist())

"""Frozen CPU full-information policy and constrained constant controls."""
import hashlib,itertools
import numpy as np
import torch
from torch import nn
PRIOR=np.array([.05,.05,.90],dtype=np.float64)
BETA=.001
LOWER=1/30

def seed(*parts):return int(hashlib.sha256(repr(parts).encode()).hexdigest()[:8],16)%(2**31)

def optimum(d):
    d=np.asarray(d,dtype=np.float64);assert d.shape==(3,) and np.isfinite(d).all();best=None
    for bits in itertools.product((False,True),repeat=3):
        active=np.array(bits);free=~active
        if not free.any():continue
        p=np.full(3,LOWER);logits=np.log(PRIOR[free])+d[free]/BETA;w=np.exp(logits-logits.max());p[free]=(1-active.sum()*LOWER)*w/w.sum()
        if p.min()<LOWER-1e-12:continue
        val=float(p@d-BETA*np.sum(p*np.log(p/PRIOR)))
        if best is None or val>best[0]:best=(val,p)
    assert best is not None;return best[1]

class Policy(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(16,32),nn.Tanh(),nn.Linear(32,3));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('offset',torch.tensor([1.,1.,52.]).log())
    def forward(self,z):return .9*torch.softmax(self.offset+self.net(z),dim=-1)+1/30

def objective(p,d):
    prior=p.new_tensor(PRIOR);return -(p*d.detach()).sum(-1).mean()+BETA*(p*(p.log()-prior.log())).sum(-1).mean()

def fit(z,d,fit_seed,charge,transaction,shuffle=False,kind='controller'):
    torch.manual_seed(fit_seed);model=Policy();opt=torch.optim.Adam(model.parameters(),lr=.003,betas=(.9,.999),eps=1e-8,weight_decay=0);z=torch.as_tensor(z,dtype=torch.float32);d=torch.as_tensor(d,dtype=torch.float32).detach()
    if shuffle:d=d[torch.randperm(len(d),generator=torch.Generator().manual_seed(seed(fit_seed,'shuffle')))]
    with torch.no_grad():initial=model(z).numpy().copy()
    for t in range(256):
        opt.zero_grad(set_to_none=True);loss=objective(model(z),d);assert torch.isfinite(loss);loss.backward();charge(opt,kind,f'{transaction}/{t}')
    return model,dict(initial_probability=initial.mean(0).tolist(),final_probability=model(z).detach().mean(0).tolist(),optimizer_steps=256)

def ridge(z,d,test):
    z=np.asarray(z,float);mu=z.mean(0);sd=np.maximum(z.std(0),1e-6);x=np.column_stack(((z-mu)/sd,np.ones(len(z))));xt=np.column_stack(((np.asarray(test)-mu)/sd,np.ones(len(test))));penalty=np.diag([.01*len(z)*3]*16+[0.]);w=np.linalg.solve(x.T@x+penalty,x.T@np.asarray(d,float));pred=xt@w;pred[:,2]=0
    return np.stack([optimum(v) for v in pred])

def folds(groups):
    distinct=sorted(set(groups),key=lambda g:hashlib.sha256(g.encode()).hexdigest());assert len(distinct)>=4,'fewer than four verified U groups'
    mapping={g:i%4 for i,g in enumerate(distinct)};return [mapping[g] for g in groups]

def choose(d,delta=0.):
    # Exact tie priority: fine, coarse, skip; dead zone is relative to fine.
    best=max((2,1,0),key=lambda a:d[a]);return best if best!=2 and d[best]-d[2]>delta else 2

def check():
    from scipy.optimize import minimize
    torch.manual_seed(0);p=Policy();z=torch.zeros(7,16);assert torch.allclose(p(z),torch.tensor(PRIOR,dtype=torch.float32).expand(7,-1),atol=1e-7)
    for d in ([0.,0.,0.],[.01,-.03,0.],[-2.,3.,0.],[.001,.003,0.]):
        q=optimum(d);assert abs(q.sum()-1)<1e-12 and q.min()>=LOWER-1e-12
        objective_np=lambda x:float(-x@np.array(d)+BETA*np.sum(x*np.log(x/PRIOR)))
        res=minimize(objective_np,PRIOR,method='SLSQP',bounds=[(LOWER,1)]*3,constraints=[{'type':'eq','fun':lambda x:x.sum()-1}],options={'ftol':1e-12,'maxiter':1000});assert res.success and abs(objective_np(q)-res.fun)<1e-8
    assert np.allclose(optimum(np.zeros(3)),PRIOR) and choose([0.,0.,0.])==2 and choose([1.,1.,0.])==1 and choose([1e-7,0.,0.],1e-6)==2
    groups=['a','b','a','c','d','e'];f=folds(groups);assert f[0]==f[2] and len(set(f))==4
    x=np.arange(160,dtype=float).reshape(10,16)/160;pred=ridge(x,np.zeros((10,3)),x);assert np.allclose(pred,PRIOR)
    reward=torch.tensor([[-1.,-.5,0.]]);loss=objective(p(z[:1]),reward);loss.backward();assert p.net[-1].bias.grad[2]<0
    return dict(status='PASSED',optimizer_calls=0,analytic_checks=4,prior_exact=True,group_isolation=True,tie_rule=True,negative_credit_direction=True)

if __name__=='__main__':print(check())

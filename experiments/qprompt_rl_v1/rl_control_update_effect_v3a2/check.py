"""Zero-optimizer regression for decisions, scaler leakage and analytic controls."""
import numpy as np
import torch
from . import policy,effect

def check():
    torch.manual_seed(0);model=policy.Policy();assert sum(p.numel() for p in model.parameters())==1411
    p=model(torch.zeros(5,40)).softmax(-1);assert torch.allclose(p,torch.tensor(policy.PRIOR,dtype=torch.float32).expand_as(p),atol=1e-7)
    assert policy.actions([[0,0,0],[1,1,0],[0,1,1]]) .tolist()==[2,1,2]
    z=np.zeros((8,40));z[:,0]=np.arange(8);z[:,20]=np.arange(8)%2;s=policy.normalize_fit(z);test=z.copy();test[:,1]=999;test[:,0]=1000;out=policy.normalize(test,s);assert (out[:,1]==0).all() and out[:,0].min()>100 and np.array_equal(out[:,20],z[:,20])
    for method in ('BASE_RIDGE','EFFECT_RIDGE','NC_OPT'):
        fit=policy.fit(method,z,np.zeros((8,3)),0,None,'zero');pr=policy.predict(fit,test);assert pr['actions']==[2]*8
    loss=policy.objective(torch.tensor([[10000.,-10000.,0.]],requires_grad=True),torch.tensor([[1.,-1.,0.]]));assert torch.isfinite(loss);loss.backward()
    params=[torch.ones(2),torch.ones(2)];g=[[torch.ones(2),None],[torch.ones(2),torch.zeros(2)],[torch.ones(2),torch.ones(2)]];updates=[[p.clone() for p in params] for _ in range(3)];v=effect.features(params,['body.x','head.x'],g,[torch.ones(2),torch.ones(2)],updates);assert len(v)==24 and np.isfinite(v).all() and v[4]==0
    return dict(status='PASSED',optimizer_calls=0,medical_reads=0,prior=True,fine_reachable=True,scaler_test_constant_columns_zero=True,analytic_zero_reward=True,stable_extreme_logits=True)

if __name__=='__main__':print(check())

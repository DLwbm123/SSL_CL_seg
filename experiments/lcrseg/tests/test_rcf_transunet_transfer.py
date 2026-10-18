"""Focused checks for statistical decisions and isolated LoRA state."""
import copy
import numpy as np
import pytest
import torch
from torch import nn
from rcf_seg_study_v1.reliability import fit_gmm, posterior, threshold
from rcf_seg_study_v1.model import TaskLinear, Adapter, routing_prototypes


def test_gmm_clean_semantics_and_covariance():
    rng=np.random.default_rng(9)
    x=np.r_[rng.normal(-2,.2,(100,2)),rng.normal(2,.2,(100,2))]
    s=fit_gmm(x);p,_=posterior(x,s)
    assert p[:100,s['clean']].mean()<.01
    assert p[100:,s['clean']].mean()>.99
    assert np.linalg.eigvalsh(s['covariances']).min()>0
    assert fit_gmm(np.ones((20,2))) is None
    with pytest.raises(ValueError):fit_gmm(np.array([[np.nan,0]]))


def test_threshold_prefix_and_asymmetry():
    x=np.array([.99,.95,.8,.2])
    assert threshold(x,.90)==.8
    assert threshold(x,.90,previous=.9)==pytest.approx(.87)
    assert threshold(np.array([.2,.1]),.9)>1
    assert threshold(np.array([.98,.97]),.98,previous=.7)==.98


def test_lora_initial_identity_and_history_isolation():
    torch.manual_seed(2)
    layer=TaskLinear(nn.Linear(16,16)).eval()
    layer.base.requires_grad_(False)
    layer.adapters['old']=Adapter(16);layer.adapters['old'].requires_grad_(False)
    layer.adapters['new']=Adapter(16);layer.active='new'
    x=torch.randn(5,16)
    before=copy.deepcopy(layer.state_dict())
    assert torch.equal(layer(x),layer.base(x))
    optimizer=torch.optim.SGD(layer.adapters['new'].parameters(),lr=.1)
    layer(x).square().mean().backward();optimizer.step()
    assert not torch.equal(layer.adapters['new'].up.weight,before['adapters.new.up.weight'])
    assert layer.base.weight.grad is None
    for n,v in layer.state_dict().items():
        if not n.startswith('adapters.new'):assert torch.equal(v,before[n])
    layer.active='source';assert torch.equal(layer(x),layer.base(x))


def test_label_free_router_separates_tasks_and_is_deterministic():
    torch.manual_seed(4)
    a=torch.randn(50,8)*.01;a[:,0]+=1
    b=torch.randn(50,8)*.01;b[:,1]+=1
    pa,pb=routing_prototypes(a),routing_prototypes(b)
    assert torch.equal(pa,routing_prototypes(a))
    assert torch.isfinite(pa).all() and len(pa)<=40
    q=torch.nn.functional.normalize(a[:4],dim=1)
    assert ((q@pa.T).topk(3).values.mean(1)>(q@pb.T).topk(3).values.mean(1)).all()

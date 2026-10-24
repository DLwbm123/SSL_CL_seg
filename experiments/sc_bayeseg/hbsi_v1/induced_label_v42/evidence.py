"""Integrate local label-function weights over image-derived RBF bases."""
import math
import torch
from ..spatial_correspondence_v28.method import feature


def support_indices(truth):
    labels=truth.flatten().long()
    classes=labels.unique(sorted=True)
    chosen=[]
    for c in classes:
        members=torch.nonzero(labels==c).flatten()
        n=min(64//len(classes),len(members))
        chosen.append(members[torch.arange(n,device=labels.device)*len(members)//n])
    index=torch.cat(chosen)
    assert len(index)==len(index.unique()) and 1<=len(index)<=64
    return index,labels[index]


def label_evidence(a,b,labels):
    # Unit RBF length scale; total coefficient prior variance one, noise one.
    z=torch.exp(-.5*(a[:,None,:]-b[None,:,:]).square().sum(-1))
    kernel=z@z.T/len(b)
    cov=torch.eye(len(a),device=a.device,dtype=a.dtype)+kernel
    chol=torch.linalg.cholesky(cov)
    target=(2*labels-1).to(a.dtype).detach()
    alpha=torch.cholesky_solve(target[:,None],chol)[:,0]
    quad=(target*alpha).sum();logdet=2*chol.diagonal().log().sum()
    value=.5*(quad+logdet+len(a)*math.log(2*math.pi))/len(a)
    return value,dict(label_nll=value.detach(),label_quadratic_per_observation=quad.detach()/len(a),label_logdet_per_observation=logdet.detach()/len(a),basis_activation_mean=z.detach().mean(),basis_activation_min=z.detach().min(),label_kernel_diagonal_mean=kernel.detach().diagonal().mean(),label_fit_mse=(target-kernel@alpha).detach().square().mean(),label_single_class=value.new_tensor(float(labels.unique().numel()==1)),label_support_count=value.new_tensor(len(a)),label_support_fg_fraction=(labels==1).to(a.dtype).mean())


def auxiliary(s,L,x):
    assert L['x'].shape==x.shape==(1,1,128,128)
    obs=L['observations']
    assert obs['obs_mask'].all() and (obs['y_obs'].sum(1)==1).all()
    # Observed one-hot masks are Bool on CUDA; convert before argmax.
    truth=obs['y_obs'].long().argmax(1)
    index,labels=support_indices(truth)
    a=feature(s.phi,L['x'])[0].flatten(1).T[index]
    b=feature(s.phi,x)[0,:,8::16,8::16].flatten(1).T
    assert a.shape==(len(index),16) and b.shape==(64,16)
    return label_evidence(a,b,labels)


def check_fixture():
    a=torch.tensor([[-1.,0.],[-.8,.1],[.9,-.1],[1.,0.]],dtype=torch.float64,requires_grad=True)
    b=torch.tensor([[-.9,.1],[.8,.2]],dtype=torch.float64,requires_grad=True)
    y=torch.tensor([0,0,1,1])
    value,_=label_evidence(a,b,y)
    z=torch.exp(-.5*(a[:,None]-b[None]).square().sum(-1))
    cov=torch.eye(4,dtype=a.dtype)+z@z.T/2
    dense=-torch.distributions.MultivariateNormal(torch.zeros(4,dtype=a.dtype),covariance_matrix=cov).log_prob((2*y-1).to(a.dtype))/4
    assert torch.allclose(value,dense,atol=1e-12,rtol=1e-12)
    # Weight-space determinant lemma and Woodbury supply an independent formula.
    design=z/math.sqrt(2);small=torch.eye(2,dtype=a.dtype)+design.T@design
    target=(2*y-1).to(a.dtype);v=design.T@target
    alternative=.5*(target@target-v@torch.linalg.solve(small,v)+torch.linalg.slogdet(small)[1]+4*math.log(2*math.pi))/4
    assert torch.allclose(value,alternative,atol=1e-12,rtol=1e-12)
    assert torch.allclose(value,label_evidence(a,b.flip(0),1-y)[0],atol=1e-12)
    mixed=y[torch.tensor([0,2,1,3])]
    assert value<label_evidence(a,b,mixed)[0]
    ga,gb=torch.autograd.grad(value,(a,b))
    assert ga.norm()>0 and gb.norm()>0 and torch.isfinite(ga).all() and torch.isfinite(gb).all()
    for which,gradient in ((0,ga),(1,gb)):
        aa,bb=a.detach().clone(),b.detach().clone()
        plus=[aa.clone(),bb.clone()];minus=[aa.clone(),bb.clone()]
        plus[which][0,0]+=1e-5;minus[which][0,0]-=1e-5
        fd=(label_evidence(*plus,y)[0]-label_evidence(*minus,y)[0])/2e-5
        assert torch.allclose(fd,gradient[0,0],atol=1e-8,rtol=1e-5)
    empty=torch.zeros(128,128,dtype=torch.long)
    idx,lab=support_indices(empty);assert len(idx)==64 and lab.sum()==0
    empty[1,1]=1;idx,lab=support_indices(empty)
    assert len(idx)==33 and (lab==1).sum()==1
    assert torch.isfinite(label_evidence(a,b,torch.zeros_like(y))[0])
    return dict(dense_Gaussian_match=True,weight_space_match=True,label_permutation_sensitive=True,class_and_basis_permutation_invariant=True,both_feature_finite_difference=True,empty_retained=True,no_duplicate_support=True,fixture_reverse_calls=1)

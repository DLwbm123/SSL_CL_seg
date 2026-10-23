"""Image affinities act directly on the posterior predictive segmentation mean."""
import json
import torch
from ..semantic_alignment_v23.method import Objective as BaseObjective
from ..semantic_evidence_v22.method import sample_semantics
from ..image_graph_v32.targets import edges
from ..posterior_teacher_v25.method import parameter_groups,AUDIT_STEPS


def normalized_cut(prob,wx,wy):
    assert prob.ndim==3 and prob.shape[0]==2
    degree=torch.zeros_like(prob[0])
    degree[:,:-1]+=wx;degree[:,1:]+=wx
    degree[:-1,:]+=wy;degree[1:,:]+=wy
    volume=(prob*degree).sum((-2,-1))
    association=2*((prob[:,:,:-1]*prob[:,:,1:]*wx).sum((-2,-1))+(prob[:,:-1,:]*prob[:,1:,:]*wy).sum((-2,-1)))
    # Empty classes contribute one, the continuous near-empty constant limit.
    loss=(1-association/volume.clamp_min(1e-12)).sum()
    total=degree.sum()
    return loss,dict(foreground_degree_fraction=(volume[1]/total).detach(),foreground_mean=prob[1].detach().mean(),minimum_class_volume=volume.min().detach(),volume_floor_classes=(volume<=1e-12).double().sum().detach(),mean_edge_affinity=(total/(2*(wx.numel()+wy.numel()))).detach())


def auxiliary(s,x):
    assert x.shape[1:]==(1,128,128)
    probs=[];rng=torch.Generator().manual_seed(35000000+101*s.steps)
    for _ in range(2):
        theta=s.q.rsample(rng);a=s.current.rsample(theta,rng)
        for _ in range(2):probs.append(sample_semantics(s,x,theta,a,rng)[0])
    prob=torch.stack(probs).mean(0);terms=[];stats=[]
    for image,p in zip(x,prob):
        wx,wy=edges(image[0].detach(),True)
        loss,record=normalized_cut(p,wx,wy);terms.append(loss);stats.append(record)
    return torch.stack(terms).mean(),{k:torch.stack([v[k] for v in stats]).mean() for k in stats[0]}


def check_fixture():
    x=torch.tensor([[0.,0.,2.],[0.,0.,2.],[0.,0.,2.]],dtype=torch.float64)
    wx,wy=edges(x,True);p=torch.tensor([[.9,.8,.1],[.85,.7,.2],[.8,.9,.15]],dtype=x.dtype)
    prob=torch.stack((1-p,p)).requires_grad_();loss,_=normalized_cut(prob,wx,wy)
    W=torch.zeros(9,9,dtype=x.dtype)
    for i in range(3):
        for j in range(3):
            k=i*3+j
            if j<2:W[k,k+1]=W[k+1,k]=wx[i,j]
            if i<2:W[k,k+3]=W[k+3,k]=wy[i,j]
    q=prob.flatten(1);d=W.sum(1)
    reference=sum((v@(W@(1-v)))/(d@v) for v in q)
    assert torch.allclose(loss,reference,atol=1e-12,rtol=1e-12)
    assert torch.allclose(loss,normalized_cut(prob.flip(0),wx,wy)[0])
    assert torch.allclose(loss,normalized_cut(prob,wx*7,wy*7)[0])
    for v in (0.,.3,.5,1.):
        z=torch.full_like(p,v);assert abs(float(normalized_cut(torch.stack((1-z,z)),wx,wy)[0])-1)<1e-12
    aligned=(x>0).double();checker=((torch.arange(3)[:,None]+torch.arange(3)[None,:])%2).double()
    assert normalized_cut(torch.stack((1-aligned,aligned)),wx,wy)[0]<normalized_cut(torch.stack((1-checker,checker)),wx,wy)[0]
    grad=torch.autograd.grad(loss,prob)[0];assert torch.isfinite(grad).all() and grad.norm()>0
    direction=torch.arange(18,dtype=x.dtype).reshape_as(prob)/18
    numerical=(normalized_cut(prob.detach()+1e-6*direction,wx,wy)[0]-normalized_cut(prob.detach()-1e-6*direction,wx,wy)[0])/2e-6
    assert torch.allclose((grad*direction).sum(),numerical,atol=1e-7,rtol=1e-7)
    return dict(dense_formula_match=True,class_permutation_invariant=True,global_affinity_scale_invariant=True,constant_and_empty_risk_one=True,aligned_better_than_checkerboard=True,finite_difference_gradient=True,fixture_reverse_calls=1)


class Objective:
    def __init__(self,state,refiner,use_u,audit_path=None):
        self.s=state;self.counter=refiner.counter;self.use_u=use_u
        self.base=BaseObjective(state,refiner,False);self.audit_path=audit_path;self.pending=None

    def estimate(self,L,U,counts,generator,sampling_weights=None):
        s=self.s;assert U is not None and 'observations' not in U
        if self.use_u:s.check_batch(U,'U')
        else:assert U['role']=='appearance_donor_L' and U['stage']==s.active['id']
        base,record=self.base.estimate(L,None,counts,generator,sampling_weights)
        aux,stats=auxiliary(s,U['x']);loss=base+aux
        if not torch.isfinite(loss):raise FloatingPointError('posterior affinity risk')
        self.counter.add('affinity_global_draws',2);self.counter.add('affinity_semantic_forwards',4*len(U['x']))
        if self.audit_path is not None and s.steps in AUDIT_STEPS:
            groups=parameter_groups(s);params=[p for group in groups.values() for p in group]
            gb=torch.autograd.grad(base,params,retain_graph=True,allow_unused=True)
            gu=torch.autograd.grad(aux,params,retain_graph=True,allow_unused=True)
            row=dict(step=s.steps,auxiliary_pool='U' if self.use_u else 'L',extra_reverse_calls=2,groups={});offset=0
            for name,pp in groups.items():
                b=torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for p,g in zip(pp,gb[offset:offset+len(pp)])])
                v=torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for p,g in zip(pp,gu[offset:offset+len(pp)])]);offset+=len(pp)
                nb,nv=float(b.norm()),float(v.norm())
                row['groups'][name]=dict(base_gradient_norm=nb,auxiliary_gradient_norm=nv,auxiliary_over_base=nv/nb if nb else None,cosine=float(torch.dot(b,v)/(b.norm()*v.norm())) if nb and nv else None)
            self.pending=(row,{name:[p.detach().clone() for p in pp] for name,pp in groups.items()})
        return loss,dict(record,scaled=loss.detach(),base_objective=base.detach(),affinity_risk=aux.detach(),auxiliary_pool_U=loss.new_tensor(float(self.use_u)),**stats)

    def after_update(self,step):
        if self.pending is None:return
        row,before=self.pending;assert step==row['step']+1
        for name,params in parameter_groups(self.s).items():
            row['groups'][name]['actual_combined_Adam_displacement']=float(torch.cat([(p.detach()-q).reshape(-1) for p,q in zip(params,before[name])]).norm())
        with self.audit_path.open('a') as f:f.write(json.dumps(row)+'\n')
        self.pending=None

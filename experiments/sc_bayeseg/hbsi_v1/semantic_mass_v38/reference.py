"""Image-level empirical foreground-mass reference from existing training labels."""
import torch
import torch.nn.functional as F
from ..data_adapter import leaf_observations
from ..stream_state import digest
from ..appearance_transport_v24.transport import DonorData


class MassData(DonorData):
    def batch(self,role,size,generator):
        return super().batch(role,8 if role=='U' else size,generator)


@torch.no_grad()
def fit(s,data,path):
    before=digest(s.payload());fractions=[]
    assert data.counts['L']==80 and data.counts['U']==271
    assert s.config.height==s.config.width==128 and s.config.classes==2
    for record in data.records['L']:
        y=data.read(record,'label')[None,None].double().to(s.q.mean.device)
        y=F.interpolate(y,(128,128),mode='nearest')[:,0].long()
        obs=leaf_observations(y,s.config.classes,s.config.label_groups)
        assert obs['obs_mask'].all() and (obs['y_obs'].sum(1)==1).all()
        fractions.append(obs['y_obs'][:,1].double().mean())
    reference=torch.stack(fractions).sort().values
    assert len(reference)==80 and torch.isfinite(reference).all()
    assert digest(s.payload())==before
    with path.open('xb') as f:torch.save(dict(reference=reference.cpu(),source_payload=before,evidence=data.evidence),f)
    return reference,dict(status='COMPLETE',L_label_value_reads=80,fit_image_value_reads=0,model_forward_calls=0,source_updates=0,reference_slices=80,empty_foreground_slices=int((reference==0).sum()),foreground_fraction_mean=float(reference.mean()),foreground_fraction_min=float(reference.min()),foreground_fraction_max=float(reference.max()),source_payload_unchanged=True,U_labels_read=False)


def empirical_w1(mass,reference):
    assert mass.ndim==reference.ndim==1 and reference.numel()%mass.numel()==0
    ordered=mass.sort().values.repeat_interleave(reference.numel()//mass.numel())
    return (ordered-reference.detach().sort().values).abs().mean()


def check_fixture():
    m=torch.tensor([0.,.5],dtype=torch.float64);r=m.new_tensor([0.,.2,.6,.8])
    value=empirical_w1(m,r)
    assert torch.allclose(value,m.new_tensor(.15))
    assert torch.equal(value,empirical_w1(m.flip(0),r.flip(0)))
    assert empirical_w1(m,m).item()==0
    m=m.new_tensor([.1,.7]).requires_grad_();r=m.new_tensor([.2,.8])
    gradient=torch.autograd.grad(empirical_w1(m,r),m)[0]
    assert torch.equal(gradient,torch.full_like(m,-.5))
    assert empirical_w1(torch.zeros(2),torch.zeros(4)).item()==0
    return dict(exact_quantile_transport=True,permutation_invariant=True,identical_distribution_zero=True,non_tie_gradient=True,empty_foreground_supported=True,fixture_reverse_calls=1)

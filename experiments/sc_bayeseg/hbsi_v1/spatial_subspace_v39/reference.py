"""Fixed affine spatial mask subspace from existing training L labels."""
import torch
import torch.nn.functional as F
from ..data_adapter import leaf_observations
from ..stream_state import digest
from ..appearance_transport_v24.transport import DonorData


@torch.no_grad()
def build_reference(masks):
    assert masks.ndim==2 and masks.dtype==torch.float64
    mu=masks.mean(0);centered=masks-mu
    _,singular,vh=torch.linalg.svd(centered,full_matrices=False)
    tolerance=torch.finfo(masks.dtype).eps*max(masks.shape)*singular.max()
    rank=int((singular>tolerance).sum())
    basis=vh[:rank].contiguous()
    return dict(mean=mu,basis=basis)


def residual(probability,reference):
    mu=reference['mean'].detach();basis=reference['basis'].detach()
    delta=probability.reshape(-1,mu.numel())-mu
    return delta-(delta@basis.T)@basis


def subspace_risk(probability,reference):
    return residual(probability,reference).square().mean()


@torch.no_grad()
def fit(s,data,path):
    before=digest(s.payload());masks=[]
    assert data.counts['L']==80 and data.counts['U']==271
    assert s.config.height==s.config.width==128 and s.config.classes==2
    for record in data.records['L']:
        y=data.read(record,'label')[None,None].double()
        y=F.interpolate(y,(128,128),mode='nearest')[:,0].long()
        obs=leaf_observations(y,s.config.classes,s.config.label_groups)
        assert obs['obs_mask'].all() and (obs['y_obs'].sum(1)==1).all()
        masks.append(obs['y_obs'][:,1].double().reshape(-1).cpu())
    masks=torch.stack(masks);reference=build_reference(masks)
    basis=reference['basis'];rank=len(basis)
    assert rank>0 and rank<80
    orth_error=float((basis@basis.T-torch.eye(rank,dtype=basis.dtype)).abs().max())
    training_error=float(residual(masks,reference).abs().max())
    assert orth_error<1e-10 and training_error<1e-9
    assert digest(s.payload())==before
    with path.open('xb') as f:torch.save(dict(reference=reference,source_payload=before,evidence=data.evidence),f)
    summary=dict(status='COMPLETE',L_label_value_reads=80,fit_image_value_reads=0,model_forward_calls=0,source_updates=0,reference_slices=80,empty_foreground_slices=int((masks.sum(1)==0).sum()),rank=rank,pixels=16384,orthogonality_max_error=orth_error,training_projection_max_error=training_error,foreground_fraction_mean=float(masks.mean()),source_payload_unchanged=True,U_labels_read=False)
    return {k:v.to(s.q.mean.device) for k,v in reference.items()},summary


def check_fixture():
    masks=torch.tensor([[0.,0.,0.,0.],[1.,0.,0.,0.],[0.,1.,0.,0.]],dtype=torch.float64)
    ref=build_reference(masks)
    assert len(ref['basis'])==2 and subspace_risk(masks,ref)<1e-25
    good=masks[1:2];bad=masks.new_tensor([[0.,0.,1.,0.]])
    assert good.sum()==bad.sum() and subspace_risk(good,ref)<1e-25 and torch.allclose(subspace_risk(bad,ref),bad.new_tensor(.25))
    p=masks.new_tensor([[.1,.2,.3,.4]]).requires_grad_()
    grad=torch.autograd.grad(subspace_risk(p,ref),p)[0]
    assert torch.allclose(grad,2*residual(p.detach(),ref)/p.numel(),atol=1e-12)
    rotated=dict(ref,basis=ref['basis'].flip(0)*-1)
    assert torch.allclose(subspace_risk(p,ref),subspace_risk(p,rotated),atol=1e-12)
    assert subspace_risk(torch.zeros_like(p),ref)<1e-25
    constant=build_reference(torch.zeros_like(masks))
    assert constant['basis'].shape==(0,4) and torch.allclose(subspace_risk(bad,constant),bad.new_tensor(.25))
    return dict(exact_span_projection=True,equal_mass_different_location=True,analytic_gradient=True,basis_sign_order_invariant=True,empty_mask_supported=True,rank_zero_supported=True,fixture_reverse_calls=1)

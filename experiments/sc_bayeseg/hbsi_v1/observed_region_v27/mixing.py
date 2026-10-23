"""Loss only on unchanged labeled pixels; donor pixels never receive labels."""
import torch
from ..paired_semantic_v19.method import balanced_risk


@torch.no_grad()
def mix_context(x,donor,step):
    assert x.shape==donor.shape and x.ndim==4
    h,w=x.shape[-2:];assert h==w==128
    y=torch.arange(h,device=x.device)[None,None,:,None]
    z=torch.arange(w,device=x.device)[None,None,None,:]
    axis=(y if step%4<2 else z);middle=(h if step%4<2 else w)//2
    retain=(axis<middle if step%2==0 else axis>=middle).expand(len(x),1,h,w)
    valid=(axis<middle-4 if step%2==0 else axis>=middle+4).expand_as(retain)
    mixed=torch.where(retain,x,donor)
    assert torch.equal(mixed[retain.expand_as(x)],x[retain.expand_as(x)])
    return mixed,valid,retain


def masked_risk(prob,truth,mask):
    assert mask.shape==(len(prob),1,*prob.shape[-2:]) and bool(mask.any())
    mask=mask.to(prob.dtype)
    return balanced_risk(prob*mask,truth*mask)


def check_fixtures():
    x=torch.arange(128*128,dtype=torch.float64).reshape(1,1,128,128)/16384
    d=-x
    for step in range(4):
        mixed,valid,retain=mix_context(x,d,step)
        assert retain.sum()==8192 and valid.sum()==7680 and not bool((valid&~retain).any())
        assert torch.equal(mixed[retain],x[retain]) and torch.equal(mixed[~retain],d[~retain])
        assert torch.equal(mix_context(x,x,step)[0],x)
    prob=torch.full((1,2,128,128),.5,dtype=torch.float64,requires_grad=True)
    truth=torch.zeros_like(prob);truth[:,0]=1;truth[:,0,20:30,20:30]=0;truth[:,1,20:30,20:30]=1
    _,valid,_=mix_context(x,d,0)
    risk=masked_risk(prob,truth,valid).mean()
    changed=torch.where(valid,truth,truth.flip(1));assert torch.equal(risk,masked_risk(prob,changed,valid).mean())
    grad=torch.autograd.grad(risk,prob)[0]
    assert torch.count_nonzero(grad[~valid.expand_as(grad)])==0
    assert torch.count_nonzero(grad[valid.expand_as(grad)])>0
    empty=truth.clone();empty[:,0]=1;empty[:,1]=0
    assert torch.isfinite(masked_risk(prob,empty,valid)).all()
    return dict(retained_pixels_exact=True,donor_labels_ignored=True,seam_excluded_pixels=512,observed_pixels=7680,output_loss_gradient_outside_mask_zero=True,empty_foreground_finite=True,fixture_reverse_calls=1)

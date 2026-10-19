"""CPU-only arithmetic checks; no private data or optimizer calls."""
import copy
import torch
from torch import nn
from sc_bayeseg.jointwrite_v1.run import displacement, stats


def run():
    torch.set_num_threads(2)
    result=stats(torch.tensor([[1.,-1.],[3.,1.]]))
    assert result==dict(mean=1.,min_sample_mean=0.,min_draw_group_mean=0.,max_draw_group_mean=2.,positive_draw_sample_fraction=.75)
    model=nn.Module();model.shape_encoder=nn.Sequential(*[nn.Identity() for _ in range(7)],nn.Linear(2,2))
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    gradients={key:{n:torch.full_like(p,fill) for n,p in model.named_parameters()} for key,fill in [('labels',1.),('image',2.)]}
    with torch.no_grad():
        for p in model.parameters():p.add_(1.)
    groups=displacement(model,before,gradients)
    for key in ['shape_encoder.mean_head','shape_encoder.logsd_head']:
        assert abs(groups[key]['displacement_squared_norm']-3.)<1e-5
        assert abs(groups[key]['label_dot_displacement']-3.)<1e-5
        assert abs(groups[key]['image_dot_displacement']-6.)<1e-5
    print('CPU_ARITHMETIC_CHECKS_PASS; optimizer_updates=0')


if __name__=='__main__':run()

"""Analytically integrated per-image linear readout of masked recognition features."""
import math
import torch
import torch.nn.functional as F
from ..masked_evidence_v34.method import indices,hide


def feature(phi,x):
    image=x
    if phi.config.coordinate_channels:
        yy,xx=torch.meshgrid(torch.linspace(-1,1,x.shape[-2],device=x.device,dtype=x.dtype),torch.linspace(-1,1,x.shape[-1],device=x.device,dtype=x.dtype),indexing='ij')
        image=torch.cat((x,torch.stack((xx,yy))[None].expand(len(x),-1,-1,-1)),1)
    h=torch.tanh(phi.enc(image))
    low=torch.tanh(phi.down(F.avg_pool2d(h,2)))
    return torch.tanh(phi.up(torch.cat((h,F.interpolate(low,x.shape[-2:],mode='bilinear',align_corners=False)),1)))


def conditional_nll(z_visible,y_visible,z_hidden,y_hidden):
    # Unit coefficient prior and unit independent observation-noise variance.
    identity=torch.eye(z_visible.shape[1],dtype=z_visible.dtype,device=z_visible.device)
    a=identity+z_visible.T@z_visible
    ca=torch.linalg.cholesky(a)
    mean_w=torch.cholesky_solve(z_visible.T@y_visible[:,None],ca)
    predicted=(z_hidden@mean_w)[:,0]
    residual=y_hidden-predicted
    ct=torch.linalg.cholesky(a+z_hidden.T@z_hidden)
    v=z_hidden.T@residual[:,None]
    quadratic=residual.square().sum()-(v*torch.cholesky_solve(v,ct)).sum()
    logdet=2*(ct.diagonal().log().sum()-ca.diagonal().log().sum())
    n=y_hidden.numel();risk=.5*(quadratic+logdet+n*math.log(2*math.pi))/n
    variance=1+(z_hidden*torch.cholesky_solve(z_hidden.T,ca).T).sum(1)
    assert torch.isfinite(risk) and quadratic.detach()>=-1e-9 and logdet.detach()>=-1e-9
    return risk,dict(heldout_prediction_mse=residual.detach().square().mean(),conditional_quadratic_per_pixel=quadratic.detach()/n,conditional_logdet_per_pixel=logdet.detach()/n,predictive_variance_mean=variance.detach().mean())


def auxiliary(s,x):
    assert x.shape==(1,1,128,128)
    index=indices(x,s.steps);masked=hide(x,index)
    f=feature(s.phi,masked);assert f.shape==(1,16,128,128)
    z=f[0].flatten(1).T;z=torch.cat((z,torch.ones_like(z[:,:1])),1)
    visible=torch.ones(16384,device=x.device,dtype=torch.bool);visible[index]=False
    y=x.flatten().detach()
    value,stats=conditional_nll(z[visible],y[visible],z[index],y[index])
    return value,dict(conditional_feature_nll=value.detach(),heldout_fraction=value.new_tensor(len(index)/16384),**stats)


def check_fixture():
    zv=torch.tensor([[1.,0.],[0.,1.],[1.,1.]],dtype=torch.float64,requires_grad=True)
    zh=zv.new_tensor([[.3,.7],[.5,-.2]]);yv=zv.new_tensor([.2,.8,.4]);yh=zv.new_tensor([.1,.6])
    risk,stats=conditional_nll(zv,yv,zh,yh)
    a=torch.eye(2,dtype=zv.dtype)+zv.T@zv
    mean=zh@torch.linalg.solve(a,zv.T@yv)
    cov=torch.eye(2,dtype=zv.dtype)+zh@torch.linalg.solve(a,zh.T)
    dense=-torch.distributions.MultivariateNormal(mean,covariance_matrix=cov).log_prob(yh)/2
    assert torch.allclose(risk,dense,atol=1e-12,rtol=1e-12)
    joint_z=torch.cat((zv,zh));joint=torch.eye(5,dtype=zv.dtype)+joint_z@joint_z.T
    schur=joint[3:,3:]-joint[3:,:3]@torch.linalg.solve(joint[:3,:3],joint[:3,3:])
    assert torch.allclose(cov,schur,atol=1e-12,rtol=1e-12)
    grad=torch.autograd.grad(risk,zv)[0];plus=zv.detach().clone();minus=plus.clone();plus[0,0]+=1e-5;minus[0,0]-=1e-5
    finite=(conditional_nll(plus,yv,zh,yh)[0]-conditional_nll(minus,yv,zh,yh)[0])/2e-5
    assert torch.allclose(grad[0,0],finite,atol=1e-8,rtol=1e-5)
    zero=torch.zeros_like(zv);zero_h=torch.zeros_like(zh)
    zvalue,_=conditional_nll(zero,yv,zero_h,yh)
    assert torch.allclose(zvalue,.5*(yh.square().mean()+math.log(2*math.pi)))
    x=torch.arange(16384,dtype=zv.dtype).reshape(1,1,128,128)/16384;idx=indices(x,1600);changed=x.clone().flatten(2);changed[:,:,idx]+=7
    assert torch.equal(hide(x,idx),hide(changed.reshape_as(x),idx))
    assert torch.isfinite(grad).all() and grad.norm()>0
    return dict(dense_predictive_logprob=True,joint_gaussian_Schur_match=True,finite_difference_feature_gradient=True,zero_design_unit_noise=True,hidden_values_not_in_input=True,fixture_reverse_calls=1)

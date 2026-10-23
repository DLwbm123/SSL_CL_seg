"""One bounded correctness check of the new detach directions, not a research gate."""
import torch
from .candidates import self_check
from .method import ranking, semantic_score
from ..semantic_evidence_v22.method import ImageEvidence,sample_semantics


def check_paths(s,L):
    fixture=self_check();rng=torch.Generator().manual_seed(232000)
    x=L['x'];truth=L['observations']['y_obs'].to(x.dtype)
    theta=s.q.rsample(rng);a=s.current.rsample(theta,rng)
    prob,_=sample_semantics(s,x,theta,a,rng)
    anchor,_,_=ranking(ImageEvidence(s.model,a),x,theta,prob,truth,s.steps)
    phi=[p for p in s.phi.parameters() if p.requires_grad]
    gg=torch.autograd.grad(anchor,phi+[theta,a],allow_unused=True,retain_graph=True)
    assert all(g is None for g in gg[:-2])
    assert all(g is not None and torch.isfinite(g).all() for g in gg[-2:])
    # Independent Pi tests direct scorer edges, not total derivatives via Pi.
    leaf=prob.detach().requires_grad_()
    score=semantic_score(s,x,leaf,theta,a).sum()
    gp,gt,ga=torch.autograd.grad(score,[leaf,theta,a],allow_unused=True)
    assert gt is None and ga is None and gp is not None and torch.isfinite(gp).all() and gp.abs().sum()>0
    routed=semantic_score(s,x,prob,theta,a).sum()
    actual=torch.autograd.grad(routed,s.phi.c_head.weight,allow_unused=True)[0]
    assert actual is not None and torch.isfinite(actual).all() and actual.abs().sum()>0
    return dict(candidate_fixture=fixture,L_ranking_phi_gradient_absent=True,U_direct_score_theta_a_gradient_absent=True,
                U_probability_gradient_nonzero=True,U_actual_phi_gradient_nonzero=True,reverse_calls=3,optimizer_updates=0)

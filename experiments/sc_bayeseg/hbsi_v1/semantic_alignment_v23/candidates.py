"""L-only detached comparisons covering soft errors and real false foreground."""
import torch
from torch.nn import functional as F


def foreground_dice(prob,truth):
    p,t=prob[:,1],truth[:,1]
    return (2*(p*t).flatten(1).sum(1)+1e-6)/(p.flatten(1).sum(1)+t.flatten(1).sum(1)+1e-6)


def pairs(prob,truth,step):
    """Every positive moves the same detached candidate halfway toward L truth."""
    p=prob.detach();t=truth.detach();fg=t[:,1:2]
    hard=F.one_hot(p.argmax(1),p.shape[1]).permute(0,3,1,2).to(p.dtype)
    changed=F.max_pool2d(fg,7,1,3) if step%2==0 else -F.max_pool2d(-fg,7,1,3)
    # Empty masks need actual false foreground, not a translated empty mask.
    h,w=fg.shape[-2:];yy=torch.arange(h,device=fg.device)[:,None];xx=torch.arange(w,device=fg.device)[None,:]
    disk=((yy-h//2)**2+(xx-w//2)**2 <= (min(h,w)//8)**2).to(p.dtype)[None,None]
    empty=fg.flatten(1).sum(1)==0
    changed=torch.where(empty[:,None,None,None],disk,changed)
    corrupt=torch.cat([1-changed,changed],1)
    result=[]
    for neg in (p,.98*hard+.02/p.shape[1],.98*corrupt+.02/p.shape[1]):
        pos=.5*(neg+t)
        support=(pos[:,1]-neg[:,1]).abs().flatten(1).sum(1).detach().clamp_min(1.)
        valid=(foreground_dice(pos,t)-foreground_dice(neg,t))>1e-12
        result.append((pos.detach(),neg.detach(),support,valid.detach()))
    return result


def self_check():
    for empty in (True,False):
        truth=torch.zeros(1,2,16,16,dtype=torch.float64);truth[:,0]=1
        if not empty:truth[:,0,5:11,5:11]=0;truth[:,1,5:11,5:11]=1
        prob=torch.full_like(truth,.5,requires_grad=True)
        for step in (0,1):
            pp=pairs(prob,truth,step)
            assert len(pp)==3 and all(bool(v.all()) for _,_,_,v in pp)
            for pos,neg,support,_ in pp:
                assert not pos.requires_grad and not neg.requires_grad and not support.requires_grad
                assert bool((foreground_dice(pos,truth)>foreground_dice(neg,truth)).all())
                assert bool((support>=1).all()) and torch.isfinite(support).all()
                torch.testing.assert_close(pos.sum(1),torch.ones_like(pos[:,0]))
    return dict(empty_false_positive_contrast=True,positive_quality_order=True,all_candidates_detached=True)

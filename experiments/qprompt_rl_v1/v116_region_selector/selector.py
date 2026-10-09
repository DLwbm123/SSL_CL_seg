"""Fixed pixel-budget spatial selection with detached, prediction-only evidence."""
import math
from functools import lru_cache
import torch
from torch import nn
from torch.nn import functional as F


class Actor(nn.Module):
    def __init__(self, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.net=nn.Sequential(nn.Linear(18,32),nn.Tanh(),nn.Linear(32,1))
            nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return self.net(x).squeeze(-1)


@lru_cache(maxsize=4)
def tiles(h,w):
    assert h%4==w%4==0
    return torch.arange(h*w).reshape(4,h//4,4,w//4).permute(0,2,1,3).reshape(16,-1)


def features(p,q,qm,qflip,target,valid,step):
    p,q,qm,qflip,target=[v.detach().float() for v in (p,q,qm,qflip,target)]
    cls=q.argmax(1);mask=valid.bool()&(q.max(1).values>.7)
    edge=torch.zeros_like(mask)
    edge[:,:,1:]|=cls[:,:,1:]!=cls[:,:,:-1];edge[:,1:,:]|=cls[:,1:,:]!=cls[:,:-1,:]
    mid=(q+qm)/2
    js=((q*(q.clamp_min(1e-8).log()-mid.clamp_min(1e-8).log())).sum(1)+(qm*(qm.clamp_min(1e-8).log()-mid.clamp_min(1e-8).log())).sum(1))/(2*math.log(2))
    kl=(target*(target.clamp_min(1e-8).log()-p.clamp_min(1e-8).log())).sum(1).clamp(0,10)/10
    fields=torch.stack([q.max(1).values,qm.max(1).values,-(q*q.clamp_min(1e-8).log()).sum(1)/math.log(3),js,kl,(cls==qm.argmax(1)).float(),(qm.argmax(1)==qflip.argmax(1)).float(),edge.float(),*[(cls==i).float() for i in range(3)]],1)
    den=F.adaptive_avg_pool2d(mask[:,None].float(),(4,4))
    avg=F.adaptive_avg_pool2d(fields*mask[:,None],(4,4))/den.clamp_min(1e-8)
    frac=den/den.sum((-2,-1),keepdim=True).clamp_min(1e-8)
    x=torch.cat((avg,frac,torch.full_like(frac,step/300)),1).flatten(2).transpose(1,2).cpu()
    assert x.shape[1:]==(16,13) and torch.isfinite(x).all() and not x.requires_grad
    return x,mask.cpu(),cls.cpu()


def choose(x,eligible,classes,mode,actor,g,trace=None,fraction=.5):
    assert fraction in (.5,1.) and mode in ('RL','CE','RANDOM','CONFIDENCE','COVERAGE')
    b,h,w=eligible.shape;index=tiles(h,w);selected=torch.zeros_like(eligible);summary=[]
    for image in range(b):
        ids=[v[eligible[image].flatten()[v]] for v in index];sizes=torch.tensor([len(v) for v in ids])
        total=int(sizes.sum());budget=int(total*fraction);left=budget;used=torch.zeros(16,dtype=torch.bool);class_count=torch.zeros(3);decisions=0
        while left:
            legal=(sizes>0)&~used;assert legal.any()
            coverage=class_count/class_count.sum().clamp_min(1)
            novelty=1-(x[image,:,8:11]*coverage).sum(1)
            state=torch.cat((x[image],coverage.expand(16,3),torch.full((16,1),(budget-left)/max(budget,1)),novelty[:,None]),1)
            with torch.no_grad():
                if mode in ('RL','CE'):
                    prob=actor(state).masked_fill(~legal,-torch.inf).softmax(-1);action=int(torch.multinomial(prob,1,generator=g))
                elif mode=='RANDOM':action=int(torch.multinomial(legal.float(),1,generator=g))
                else:
                    options=legal.nonzero().flatten().tolist()
                    action=max(options,key=lambda j:((float(novelty[j]),float(x[image,j,0]),-j) if mode=='COVERAGE' else (float(x[image,j,0]),-j)))
            if trace is not None:trace.append((state.clone(),legal.clone(),action))
            count=min(left,int(sizes[action]));positions=ids[action]
            if count<len(positions):positions=positions[(torch.arange(count)*len(positions))//count]
            selected[image].flatten()[positions]=True
            class_count+=torch.bincount(classes[image].flatten()[positions],minlength=3).float()
            left-=count;used[action]=True;decisions+=1
        assert int(selected[image].sum())==budget and not (selected[image]&~eligible[image]).any()
        summary.append(dict(eligible=total,selected=budget,tiles=decisions,classes=[int(v) for v in class_count]))
    return selected,summary


def selected_loss(p,q,target,valid,mask):
    admitted=valid.bool()&(q.detach().max(1).values>.7)
    raw=q.new_tensor([.5,1.,1.5])[q.detach().argmax(1)]
    weight=raw*mask.to(p.device);weight=weight*admitted.sum()/weight.sum().clamp_min(1e-8)
    loss=((target.detach()*(target.detach().clamp_min(1e-8).log()-p.float().clamp_min(1e-8).log())).sum(1)*weight).sum()/valid.sum().clamp_min(1)
    return loss


def pack(trace):
    if not trace:return dict(x=torch.empty(0,16,18),legal=torch.empty(0,16,dtype=torch.bool),action=torch.empty(0,dtype=torch.long))
    return dict(x=torch.stack([v[0] for v in trace]),legal=torch.stack([v[1] for v in trace]),action=torch.tensor([v[2] for v in trace]))


def terms(actor,data):
    if len(data['action'])==0:
        zero=sum(v.sum()*0 for v in actor.parameters());return zero,zero
    logits=actor(data['x']).masked_fill(~data['legal'],-torch.inf)
    log=logits.log_softmax(-1);prob=log.exp();safe=log.masked_fill(~data['legal'],0.)
    return log.gather(1,data['action'][:,None]).mean(),-(prob*safe).sum(1).mean()


def update(rl,ce,rlopt,ceopt,branches,rewards,call):
    r=torch.tensor(rewards);adv=(r-r.mean())/r.std(unbiased=False).clamp_min(1e-4)
    log,entropy=zip(*(terms(rl,v) for v in branches))
    loss=-(torch.stack(log)*adv).mean()-.01*torch.stack(entropy).mean()
    rlopt.zero_grad(set_to_none=True);loss.backward();grad=float(nn.utils.clip_grad_norm_(rl.parameters(),1.));assert math.isfinite(grad)
    call('selector_RL',rlopt.step)
    winner=max(range(len(rewards)),key=lambda i:(rewards[i],-i));celoss=-terms(ce,branches[winner])[0]
    ceopt.zero_grad(set_to_none=True);celoss.backward();cegrad=float(nn.utils.clip_grad_norm_(ce.parameters(),1.));assert math.isfinite(cegrad)
    call('selector_CE',ceopt.step)
    return dict(rl_loss=float(loss.detach()),ce_loss=float(celoss.detach()),rl_gradient_norm=grad,ce_gradient_norm=cegrad,winner=winner,advantages=adv.tolist())


def selfcheck(original):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(116)
        logits=torch.randn(2,3,8,8,requires_grad=True);p=logits.softmax(1)
        q=torch.softmax(torch.randn_like(p)*4,1).detach().requires_grad_();qm=torch.softmax(torch.randn_like(p)*4,1).detach().requires_grad_();v=torch.ones(2,8,8,dtype=torch.bool)
        old,_,target=original(p,q,v,11,qm,qm);x,eligible,classes=features(p,q,qm,qm,target,v,100)
        before=torch.get_rng_state();a=Actor(601);assert torch.equal(before,torch.get_rng_state()) and sum(z.numel() for z in a.parameters())==641
        for mode in ('RL','CE','RANDOM','CONFIDENCE','COVERAGE'):
            trace=[];mask,stats=choose(x,eligible,classes,mode,a,torch.Generator().manual_seed(7),trace)
            again,_=choose(x,eligible,classes,mode,a,torch.Generator().manual_seed(7));assert torch.equal(mask,again)
            assert all(r['selected']==r['eligible']//2 and sum(r['classes'])==r['selected'] and 0<=r['tiles']<=16 for r in stats)
            assert torch.equal(before,torch.get_rng_state())
            log,en=terms(a,pack(trace));assert torch.isfinite(log+en)
            grads=torch.autograd.grad(-log,a.parameters());assert all(torch.isfinite(z).all() for z in grads)
        full,_=choose(x,eligible,classes,'CONFIDENCE',a,None,fraction=1.);new=selected_loss(p,q,target,v,full)
        assert torch.equal(new,old) and torch.equal(torch.autograd.grad(old,logits,retain_graph=True)[0],torch.autograd.grad(new,logits,retain_graph=True)[0])
        partial=selected_loss(p,q,target,v,mask);gs=torch.autograd.grad(partial,(logits,q,qm),allow_unused=True);assert torch.isfinite(gs[0]).all() and gs[1:] == (None,None)
        empty,_=choose(x,eligible&False,classes,'COVERAGE',a,None);zero=selected_loss(p,q,target,v&False,empty);assert torch.isfinite(zero) and zero==0
        # Coverage changes the legal ranking after a first choice, independently of confidence.
        test=torch.zeros(1,16,13);test[0,:,0]=torch.arange(16)/16;test[0,:,8]=1;test[0,0,8]=0;test[0,0,9]=1
        seen=[];choose(test,torch.ones(1,4,4,dtype=torch.bool),torch.zeros(1,4,4,dtype=torch.long),'COVERAGE',a,None,seen)
        assert seen[0][2]==15 and seen[1][2]==0
    return dict(status='PASS',optimizer_calls=0,model_forwards=0,checks=['exact pixel budget','legal unique tiles','private RNG replay','full budget native loss and gradient','teacher detach','empty support','coverage history','finite actor gradients'])

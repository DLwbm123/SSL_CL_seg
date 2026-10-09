import torch
from torch import nn
from torch.nn import functional as F
from ..five_frameworks_v1.kernels import StageSubspaceAdapter
from ..five_frameworks_v1.model import Model
from ..five_frameworks_v1.train_stage import StageTrainer
from ..five_frameworks_v1.gate import digest

ARMS = ('F5', 'BOUNDARY', 'OT_TRIPLET', 'LOSS_ADAPTER')
SPEC = dict(boundary_weight=.05, interior_weight=.05, boundary_radius=2,
            ot_weight=.2, ot_epsilon=.1, ot_iterations=100, ot_margin=.1,
            ot_samples=32, ot_min_support=8, ot_modes=2,
            adapter_branches=3, adapter_rank=1, adapter_scale=1., orthogonal_weight=.001)


def edge_map(values):
    """Total-variation class transitions, on the same output grid."""
    dx=(values[:,:,:,1:]-values[:,:,:,:-1]).abs().sum(1)/2
    dy=(values[:,:,1:,:]-values[:,:,:-1,:]).abs().sum(1)/2
    return torch.maximum(F.pad(dx,(0,1)),F.pad(dy,(0,0,0,1)))


def boundary_interior(logp, labels, mix_mask=None):
    if logp.shape[1]!=3 or labels.shape!=logp.shape[:1]+logp.shape[2:]:
        raise ValueError('registered three-class output/label grid required')
    if not bool((((labels>=0)&(labels<3))|(labels==255)).all()):
        raise ValueError('invalid labels')
    valid=labels!=255
    # Ignore neighborhoods, image edges and synthetic mix seams cannot be boundaries.
    radius=SPEC['boundary_radius'];kernel=2*radius+1
    invalid=F.max_pool2d((~valid)[:,None].float(),kernel,1,radius)[:,0]>0
    safe=valid&~invalid
    safe[:,:radius]=False;safe[:,-radius:]=False
    safe[:,:,:radius]=False;safe[:,:,-radius:]=False
    if mix_mask is not None:
        if mix_mask.shape!=labels.shape or mix_mask.dtype!=torch.bool:
            raise ValueError('mix mask must have the output grid')
        high=F.max_pool2d(mix_mask[:,None].float(),kernel,1,radius)[:,0]
        low=-F.max_pool2d(-mix_mask[:,None].float(),kernel,1,radius)[:,0]
        safe &= high==low
    target=F.one_hot(labels.masked_fill(~valid,0),3).permute(0,3,1,2).float()
    boundary=edge_map(target)>0
    band=F.max_pool2d(boundary[:,None].float(),kernel,1,radius)[:,0]>0
    mask=band&safe
    prediction=edge_map(logp.exp()).clamp(1e-5,1-1e-5)
    bce=F.binary_cross_entropy(prediction,boundary.to(prediction),reduction='none')
    bl=(bce*mask).sum()/mask.sum().clamp_min(1)
    terms=[];points=0
    for i in range(len(labels)):
        for c in (1,2):
            inside=(labels[i]==c)&safe[i]&~band[i]
            n=int(inside.sum())
            if n<SPEC['ot_min_support']:continue
            gt=inside.to(logp);pred=logp[i,c].exp()*gt
            gt=gt/gt.sum();pred=pred/pred.sum().clamp_min(1e-12)
            # Two spatial marginals adapt the paper's temporal CDF; this is not topology preservation.
            # Torch 2.2 has no deterministic floating CUDA cumsum; transfer only two marginals.
            terms.append(sum((pred.sum(axis).cpu().cumsum(0)-gt.sum(axis).cpu().cumsum(0)).square().mean()
                             for axis in (0,1)).to(logp)/2)
            points+=n
    interior=torch.stack(terms).mean() if terms else logp.sum()*0
    return bl,interior,dict(boundary_pixels=int(mask.sum()),interior_pixels=points,
                            interior_classes=len(terms))


def sinkhorn_cost(x,y):
    """Uniform empirical entropic OT, with fixed iterations in log coordinates."""
    if x.ndim!=2 or y.ndim!=2 or x.shape[1]!=y.shape[1] or min(len(x),len(y))==0:
        raise ValueError('nonempty NxD distributions required')
    x,y=x.double(),y.double();cost=torch.cdist(x,y).square()
    eps=SPEC['ot_epsilon']
    # Envelope gradient of the converged primal, without a 100-iteration autograd graph.
    with torch.no_grad():
        logk=-cost/eps
        a=cost.new_full((len(x),),-__import__('math').log(len(x)))
        b=cost.new_full((len(y),),-__import__('math').log(len(y)))
        u,v=torch.zeros_like(a),torch.zeros_like(b)
        for _ in range(SPEC['ot_iterations']):
            u=a-torch.logsumexp(logk+v[None,:],1)
            v=b-torch.logsumexp(logk+u[:,None],0)
        logp=logk+u[:,None]+v[None,:];p=logp.exp()
    residual=torch.maximum((p.sum(1)-a.exp()).abs().max(),(p.sum(0)-b.exp()).abs().max())
    if not torch.isfinite(p).all() or float(residual.detach())>1e-3:
        raise FloatingPointError('fixed Sinkhorn marginal residual exceeds 1e-3')
    return (p*cost+eps*p*(logp-1)).sum()


def sinkhorn_divergence(x,y):
    return sinkhorn_cost(x,y)-(sinkhorn_cost(x,x)+sinkhorn_cost(y,y))/2


@torch.no_grad()
def two_modes(values):
    """Current-batch teacher centroids only; no learned or historical prototype bank."""
    if len(values)<8:raise ValueError('insufficient class support')
    first=values[0];second=values[(values-first).square().sum(1).argmax()]
    centers=torch.stack((first,second))
    for _ in range(5):
        index=torch.cdist(values,centers).argmin(1)
        centers=torch.stack([values[index==k].mean(0) if bool((index==k).any()) else centers[k]
                             for k in range(2)])
    return F.normalize(centers,dim=1,eps=1e-6)


def distribution_triplet(student,teacher,u_classes,l_classes,valid,rng):
    s=F.normalize(student,dim=1,eps=1e-6).permute(0,2,3,1)
    t=F.normalize(teacher.detach(),dim=1,eps=1e-6).permute(0,2,3,1)
    reference={};counts={};terms=[]
    for c in (1,2):
        values=t[l_classes==c];n=min(SPEC['ot_samples'],len(values))
        if n>=SPEC['ot_min_support']:
            chosen=torch.randperm(len(values),generator=rng)[:n].to(values.device)
            reference[c]=two_modes(values[chosen])
    for c in (1,2):
        values=s[(u_classes==c)&valid];n=min(SPEC['ot_samples'],len(values));counts[c]=n
        if n<SPEC['ot_min_support'] or len(reference)!=2:continue
        chosen=torch.randperm(len(values),generator=rng)[:n].to(values.device)
        anchor=values[chosen]
        # The anchor self-cost cancels exactly in the difference of divergences.
        difference=sinkhorn_cost(anchor,reference[c])-sinkhorn_cost(anchor,reference[3-c])
        difference=difference-(sinkhorn_cost(reference[c],reference[c])-sinkhorn_cost(reference[3-c],reference[3-c]))/2
        terms.append(F.relu(difference+SPEC['ot_margin']))
    loss=torch.stack(terms).mean().to(student) if terms else student.sum()*0
    return loss,dict(samples=counts,valid_classes=len(terms),reference_classes=len(reference))


class LossIndexedAdapter(StageSubspaceAdapter):
    """Linear rank-one UAM adaptation; all losses still share the same output."""
    def __init__(self,original,rng):
        super().__init__(original.q,original.previous)
        self.r.data.copy_(original.r)
        k=self.q.shape[1];branches=SPEC['adapter_branches']
        self.up=nn.Parameter(self.q.new_zeros(branches,k,1))
        down=F.normalize(torch.randn(branches,1,k,generator=rng),dim=-1).to(self.q)
        self.down=nn.Parameter(down)

    def current_matrix(self):
        return self.r+SPEC['adapter_scale']*(self.up@self.down).mean(0)

    def optimizer_parameters(self):
        return [self.r,self.up,self.down]

    def orthogonal(self):
        v=F.normalize(self.down[:,0],dim=1,eps=1e-6);gram=v@v.T
        return (gram-torch.eye(len(v)).to(v)).square().sum()/(len(v)*(len(v)-1))

    @torch.no_grad()
    def update_teacher(self,teacher,decay=.99):
        # EMA the effective linear transform; EMA of factors would change its meaning.
        teacher.r.mul_(decay).add_(self.current_matrix(),alpha=1-decay)
        teacher.up.zero_()


def make_model(parent,arm,options,previous,rng):
    if arm not in ARMS:raise ValueError('unregistered pilot arm')
    model=Model(parent,'F5',ratio=options['rank_ratio'],previous=previous)
    if arm=='LOSS_ADAPTER':model.sidecar=LossIndexedAdapter(model.sidecar,rng)
    return model


class PilotTrainer(StageTrainer):
    def __init__(self,*args,arm,enabled=True,**kwargs):
        self.arm=arm;self.enabled=enabled;self.module_stats={};self.gradient_rows=[]
        if arm not in ARMS:raise ValueError('unregistered pilot arm')
        super().__init__(*args,**kwargs)

    @classmethod
    def for_resume(cls,model,provider,options=None,cwmi=None,execution=None,*,arm,enabled=True):
        return cls(model,provider,options,cwmi,initialize=False,execution=execution,arm=arm,enabled=enabled)

    def loss_contract(self):
        return dict(pilot_arm=self.arm,enabled=self.enabled,module_spec=SPEC,
                    implementation=digest(__import__('pathlib').Path(__file__).read_text()))

    def labeled_addition(self,logp,labels,mix_mask):
        if not self.enabled:return None
        if self.arm=='BOUNDARY':
            bl,interior,stats=boundary_interior(logp,labels,mix_mask)
            self.module_stats={**stats,'boundary_loss':float(bl.detach()),'interior_loss':float(interior.detach())}
            active=self.step>=__import__('math').ceil(self.options['warmup_fraction']*self.options['total_steps'])
            return SPEC['boundary_weight']*bl+(SPEC['interior_weight']*interior if active else 0)
        if self.arm=='LOSS_ADAPTER':
            orth=self.model.sidecar.orthogonal()
            self.module_stats={'orthogonal':float(orth.detach())}
            return SPEC['orthogonal_weight']*orth
        return None

    def f5_addition(self,student,teacher,u_classes,l_classes,valid):
        if self.enabled and self.arm=='OT_TRIPLET':
            loss,stats=distribution_triplet(student,teacher,u_classes,l_classes,valid,self.rng('pilot_ot'))
            self.module_stats={**stats,'triplet_loss':float(loss.detach())}
            return SPEC['ot_weight']*loss
        return None

    def gradient_audit(self,labeled,unlabeled):
        if not self.native or self.execution.bindings.get('execution_scope')!='formal':return
        total=self.options['total_steps']
        if unlabeled is None or self.step+1 not in {max(1,int(total*p)) for p in (.25,.5,.75,1.)}:return
        params=self.model.sidecar.optimizer_parameters()
        a=torch.autograd.grad(labeled,params,retain_graph=True,allow_unused=True)
        b=torch.autograd.grad(unlabeled,params,retain_graph=True,allow_unused=True)
        def flat(gs):return torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(params,gs)])
        a,b=flat(a),flat(b);na,nb=a.norm(),b.norm()
        self.gradient_rows.append(dict(step=self.step+1,L_norm=float(na),U_norm=float(nb),
                                  cosine=float(a@b/(na*nb)) if na>0 and nb>0 else None))
        self.telemetry['probe_vjps']+=2

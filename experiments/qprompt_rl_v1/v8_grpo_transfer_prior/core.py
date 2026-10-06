"""V8 current-image scoped transfer loss; native optimizer and student lifecycle."""
import copy
import math
import random
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import torch
from torch import nn
from experiments.qprompt_rl_v1.v3b_lrref_endpoint import engine as e

ACTIONS = [(a,b) for a in (0.,.25,.5) for b in ((1.,1.,1.),(.5,1.25,1.25),(.5,1.,1.5))]
CAPS = dict(qualification=128, auxiliary=8000, entries=1200, audit=14400,
            prior=51200, development=4000, target=74200, online=25600,
            low_label=63600, low_online_main=6400, low_online_branch=6400,
            low_qualification=32, actor_qualification=128, actor_prior=512,
            actor_target=256, actor_low=64)

class Ledger:
    """Append before invocation; restore never owns this physical accounting."""
    def __init__(self,root):
        self.path=Path(root)/'PHYSICAL_LEDGER.jsonl';self.count=Counter()
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                r=__import__('json').loads(line)
                if r['event']=='attempt':self.count[r['category']]+=1
    def call(self,category,key,fn):
        if category not in CAPS or self.count[category]>=CAPS[category]:raise RuntimeError('physical cap '+category)
        self.count[category]+=1
        r=dict(category=category,key=key,ordinal=self.count[category])
        e.append(self.path,dict(event='attempt',**r))
        try:out=fn()
        except BaseException:
            e.append(self.path,dict(event='failure',**r));raise
        e.append(self.path,dict(event='success',**r));return out
    def wrap(self,t):
        original=t.optimizer.step
        def step(*a,**kw):return self.call(t.category,t.key,lambda:original(*a,**kw))
        step._wrapped_by_lr_sched=True;t.optimizer.step=step


def split_roles(data):
    rows=e.primitive.metadata(data)
    pools={k:sorted(r['case_id'] for r in rows if r['site_or_vendor']=='REFUGE' and r['primary_20pct_split']==k) for k in ('train_labeled','train_unlabeled')}
    l,u=pools.values()
    if (len(l),len(u))!=(40,160):raise ValueError('frozen image count')
    random.Random(8601).shuffle(l);random.Random(8601).shuffle(u)
    roles={};start=0
    for role,n in zip(('M_fit','A_fit','Q_train_old','Q_train_new','Q_dev_old','Q_dev_new'),(16,8,4,4,4,4)):
        roles[role]=l[start:start+n];start+=n
    roles.update(U_memory=u[:80],U_adapt=u[80:])
    assert sum(map(len,roles.values()))==len(set(x for v in roles.values() for x in v))
    return roles


def subset(ds,ids):
    out=copy.copy(ds);wanted=set(ids);out.rows=[copy.deepcopy(r) for r in ds.rows if r['case_id'] in wanted];out.checked=set()
    if len(out.rows)!=len(wanted):raise PermissionError('role contains unavailable images')
    return out


def photo(x,condition):
    kind,factor=condition
    if kind=='identity':return x
    if kind=='brightness':return (x*factor).clamp(0,1)
    if kind=='contrast':
        weights=x.new_tensor([.2989,.587,.114]).reshape(1,3,1,1)
        mean=(x*weights).sum(1,keepdim=True).mean((-2,-1),keepdim=True)
        return ((x-mean)*factor+mean).clamp(0,1)
    raise ValueError('unregistered photo condition')


class Provider(e.NativeCurrentDomain):
    def __init__(self,*args,roles,fit='A_fit',n=8,condition=('identity',1.),**kw):
        super().__init__(*args,**kw);self.condition=condition;self.roles=copy.deepcopy(roles)
        if self.domain!='REFUGE':raise PermissionError('first-domain provider cannot open target data')
        self._l=subset(self._l,roles[fit][:n])
        self._u=subset(e.primitive.CurrentData(args[0],0,'train_unlabeled'),roles['U_memory' if fit=='M_fit' else 'U_adapt']) if kw.get('allow_u',True) else None
        self.fit_role=fit
    def semantic_metadata(self):
        return dict(super().semantic_metadata(),fit_role=self.fit_role,fit_cases=[r['case_id'] for r in self._l.rows],u_cases=[r['case_id'] for r in self._u.rows] if self._u is not None else [],condition=list(self.condition))
    def _batch(self,*args):
        x,y,ids=super()._batch(*args);return photo(x,self.condition),y,ids


def transfer_loss(p,q,valid,action,qm=None,qflip=None):
    if action not in range(9):raise ValueError('unknown action')
    alpha,b=ACTIONS[action]
    if qm is None and alpha:raise PermissionError('memory-unavailable action')
    q=q.detach();valid=valid.bool();mask=valid&(q.max(1).values>.7)
    gate=torch.zeros_like(valid) if qm is None else valid&(qm.detach().max(1).values>.7)&(qm.detach().argmax(1)==qflip.detach().argmax(1))
    target=q if qm is None else (1-alpha*gate[:,None])*q+alpha*gate[:,None]*qm.detach()
    raw=q.new_tensor(b)[q.argmax(1)];weight=raw*mask
    weight=weight*mask.sum()/weight.sum().clamp_min(1e-8)
    if action==0:loss=e.u_loss(p,q,valid,2)[0]
    else:loss=((target*(target.clamp_min(1e-8).log()-p.float().clamp_min(1e-8).log())).sum(1)*weight).sum()/valid.sum().clamp_min(1)
    stats=dict(admitted=int(mask.sum()),valid=int(valid.sum()),gate=float(gate.sum()/valid.sum().clamp_min(1)),weight_raw=float((raw*mask).sum()),weight_normalized=float(weight.sum()))
    return loss,stats,target


class Actor(nn.Module):
    def __init__(self,seed):
        super().__init__()
        # Actor construction must not perturb the student RNG either.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed);self.net=nn.Sequential(nn.Linear(24,32),nn.Tanh(),nn.Linear(32,9))
            nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,z,memory=True):
        logits=self.net(z)
        if not memory:logits=logits.masked_fill(torch.arange(9,device=z.device)>=3,-torch.inf)
        return logits
    def sample(self,z,g,memory=True):
        return int(torch.multinomial(self(z,memory).softmax(-1).detach().cpu(),1,generator=g))


def actor_update(actor,opt,z,actions,rewards,floor,ledger,category,key,entry=None):
    r=torch.tensor(rewards,dtype=torch.float32);std=float(r.std(unbiased=False));rows=[]
    if std<floor:return dict(skipped=True,std=std,updates=[])
    adv=(r-r.mean())/max(std,floor);a=torch.tensor(actions);old=actor(z).detach().log_softmax(-1)
    for epoch in range(4):
        log=actor(z).log_softmax(-1);prob=log.exp();ratio=(log[a]-old[a]).exp()
        loss=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()+.01*(prob*log).sum()
        if entry is not None:loss=loss+.01*(prob*(log-entry)).sum()
        opt.zero_grad(set_to_none=True);loss.backward();gn=float(nn.utils.clip_grad_norm_(actor.parameters(),1.))
        ledger.call(category,key, opt.step)
        with torch.no_grad():
            now=actor(z).log_softmax(-1);kl=float((now.exp()*(now-old)).sum())
        rows.append(dict(epoch=epoch,kl=kl,gradient_norm=gn,clip_fraction=float(((ratio<.8)|(ratio>1.2)).float().mean()),entropy=float(-(prob*log).sum())))
        if kl>.02:break
    return dict(skipped=False,std=std,advantages=adv.tolist(),updates=rows)


def snapshot(t):
    v=e.snapshot(t)
    v.update(options=e.cpu(t.options),physical_optimizer_updates=t.physical_optimizer_updates,
             ema_modes=[m.training for m in t.ema.modules()],memory=e.cpu(t.memory.state_dict()),
             memory_modes=[m.training for m in t.memory.modules()],requires_restore=t.requires_restore,
             checked=[copy.deepcopy(ds.checked) if ds is not None else None for ds in (t.provider._l,t.provider._u)],
             provider_identity=e.cpu(t.provider.semantic_metadata()),condition=t.provider.condition)
    return v


def restore(t,v):
    if t.provider.semantic_metadata()!=v['provider_identity'] or t.provider.condition!=v['condition']:raise ValueError('provider restore identity')
    e.restore(t,v);t.memory.load_state_dict(v['memory']);t.options=copy.deepcopy(v['options'])
    t.physical_optimizer_updates=v['physical_optimizer_updates'];t.requires_restore=v['requires_restore']
    for module,key in ((t.ema,'ema_modes'),(t.memory,'memory_modes')):
        for m,mode in zip(module.modules(),v[key]):m.training=mode
    for ds,checked in zip((t.provider._l,t.provider._u),v['checked']):
        if ds is not None:ds.checked=copy.deepcopy(checked)


class Trainer(e.Trainer):
    action=0
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw);self.memory=self.model.teacher();self.memory_available=True
        self.entry_weights=[x.effective_weight().detach().cpu().clone() for x in self.model.parent.adapters]
        self.category='qualification';self.key='qualification'
    @contextmanager
    def readonly(self):
        saved=snapshot(self)
        try:yield
        finally:restore(self,saved)
    def losses(self):
        l,p,q,v=self.components()
        with torch.no_grad():
            u,_,_=self.provider.unlabeled(self.cursor)
            qm=self.memory(u,mode='teacher').softmax(1) if self.memory_available else None
            qf=self.memory(u.flip(-1),mode='teacher').softmax(1).flip(-1) if self.memory_available else None
        loss,stats,_=transfer_loss(p,q,v,self.action,qm,qf)
        self.last.update(active_U=True,unlabeled_loss=float(loss.detach())*.5,transfer=stats)
        return l,.5*loss,None
    def extract(self):
        with self.readonly():
            l,p,q,v=self.components();base,_=e.state_vector(q,p,v,l,self.step,self.optimizer.param_groups[0]['lr']/self.scheduler.base_lrs[0]);base[0]=self.step/self.options['total_steps']
            extra=[float(self.memory_available),len(self.provider._l)/(len(self.provider._l)+len(self.provider._u))]
            with torch.no_grad():
                if self.memory_available:
                    u,_,_=self.provider.unlabeled(self.cursor);m=self.memory(u,mode='teacher').softmax(1);f=self.memory(u.flip(-1),mode='teacher').softmax(1).flip(-1)
                    mix=(m+q)/2;js=((m*(m.clamp_min(1e-8).log()-mix.clamp_min(1e-8).log())).sum(1)+(q*(q.clamp_min(1e-8).log()-mix.clamp_min(1e-8).log())).sum(1))/(2*math.log(2))
                    avg=lambda a,mask:float(a[mask].mean()) if mask.any() else 0.
                    extra += [avg((m.max(1).values>.7).float(),v),avg((m.argmax(1)==f.argmax(1)).float(),v),avg(js,v),avg(js,v&(q.argmax(1)==1)),avg(js,v&(q.argmax(1)==2))]
                else:extra += [0.]*5
                delta=sum(float((a.effective_weight()-w.to(a.a)).square().sum()) for a,w in zip(self.model.parent.adapters,self.entry_weights))**.5
                norm=sum(float(w.square().sum()) for w in self.entry_weights)**.5
                extra += [delta/max(norm+delta,1e-12)]
            z=torch.cat((base.cpu(),torch.tensor(extra)));assert z.shape==(24,) and torch.isfinite(z).all();return z


def permit(config,seed=168,order=1,stage=0):
    domain=e.primitive.DOMAINS[0] if stage==0 else e.DOMAINS[order-1]
    identity=dict(domain=domain,seed=seed,order=order,stage=stage,manifest=e.primitive.MANIFEST_SHA,split=e.primitive.SPLIT_SHA)
    return e.ExecutionPermit(dict(execution_scope='formal',code_commit=config['commit'],authorized_manifest_digests=[e.digest(identity)],authorization='V8 image-level user amendment'),('V8',),CAPS,e._PERMIT_SEAL)


def create(config,roles,payload,ledger,n=8,condition=('brightness',.8),seed=168,horizon=900):
    auth=permit(config,seed);source=dict(domain='REFUGE',seed=seed,node_id='clean_auxiliary')
    provider=Provider(config['data'],seed,1,0,source,'cuda:0',auth,roles=roles,n=n,condition=condition,allow_u=True)
    native=e.build(config['reference'],torch.device('cuda:0'),seed);native.load_state_dict(payload['student'])
    model=e.Model(e.NativeLRParent(native,seed,source),'B0_PARENT_LCTX').to('cuda:0')
    opts=dict(lr=.001,weight_decay=4e-5,total_steps=horizon,warmup_fraction=.2,U_ramp_fraction=.2,parent_lr_multiplier=.5,lr_B_over_A=1.,lambda_U=.5)
    t=Trainer(model,provider,opts,execution=auth);ledger.wrap(t);return t

"""Bound native runner, isolated feature extraction and durable physical accounting."""
import copy, hashlib, json, os, random, time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from experiments.lcrseg.five_frameworks_v1.native_parent import NativeLRParent, build
from experiments.lcrseg.five_frameworks_v1.native_data import NativeCurrentDomain, primitive
from experiments.lcrseg.five_frameworks_v1.native_runner import write, read
from experiments.lcrseg.five_frameworks_v1.integration import ExecutionPermit, _PERMIT_SEAL
from experiments.lcrseg.five_frameworks_v1.gate import digest
from experiments.lcrseg.five_frameworks_v1.model import Model
from experiments.lcrseg.five_frameworks_v1.train_stage import StageTrainer
from experiments.lcrseg.five_frameworks_v1.recipes import collected_forward, generator
from experiments.lcrseg.five_frameworks_v1.checkpoint import atomic_save
from experiments.lcrseg.five_frameworks_v1.semantics import tensor_fingerprint
from .maths import u_loss, quality, state_vector
from . import effect

DOMAINS=('RIM_ONE_r3','Drishti_GS')
METHODS=('ORIGINAL','FINE','BASE_POLICY','EFFECT_POLICY','BASE_RIDGE','EFFECT_RULE')
def stable(*x):return int(hashlib.sha256(repr(x).encode()).hexdigest()[:12],16)%(2**31)
def append(path,v):
    with Path(path).open('a') as f:f.write(json.dumps(v,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def cpu(v):
    if torch.is_tensor(v):return v.detach().cpu().clone()
    if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
    if isinstance(v,list):return [cpu(x) for x in v]
    if isinstance(v,tuple):return tuple(cpu(x) for x in v)
    return copy.deepcopy(v)
def same(a,b):
    if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

def rng_state():return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),python=random.getstate(),numpy=np.random.get_state())
def rng_restore(v):
    torch.set_rng_state(v['torch'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in v['cuda']]);random.setstate(v['python']);np.random.set_state(v['numpy'])

class Ledger:
    def __init__(self,root):
        self.root=Path(root);self.session=read(self.root/'SESSION.json');self.count=Counter();self.key='';self.category='smoke'
        self.caps=dict(smoke=240,development=1080,main=28800,controller=1024,replay=1506)
        if (self.root/'PHYSICAL_LEDGER.jsonl').exists():
            for line in (self.root/'PHYSICAL_LEDGER.jsonl').read_text().splitlines():
                v=json.loads(line)
                if v['event']=='attempt':self.count[v['category']]+=1
    def call(self,category,key,fn):
        assert time.time()<self.session['optimizer_deadline'],'OPTIMIZER_DEADLINE'
        assert self.count[category]<self.caps[category], 'physical budget '+category
        self.count[category]+=1
        row=dict(category=category,key=key,time=time.time(),ordinal=self.count[category])
        append(self.root/'PHYSICAL_LEDGER.jsonl',dict(event='attempt',**row))
        result=fn();append(self.root/'PHYSICAL_LEDGER.jsonl',dict(event='success',**row));return result
    def step(self,opt,category,key):return self.call(category,key,opt.step)
    def wrap(self,t):
        fn=t.optimizer.step
        def step(*a,**k):return self.call(self.category,self.key,lambda:fn(*a,**k))
        step._wrapped_by_lr_sched=True;t.optimizer.step=step
    def update(self,t,category,key,action):
        self.category,self.key=category,key;t.action=action;t.update()
        row=dict(key=key,phase=category,step=t.step,cursor=t.cursor,action=action,loss=t.last['labeled_loss'],time=time.time(),physical=dict(self.count))
        if category=='development' and not (self.root/'FIRST_TRAINING_UPDATE.json').exists():write(self.root/'FIRST_TRAINING_UPDATE.json',row)
        if t.step==1 or t.step%25==0:write(self.root/'progress.json',row)

class Provider(NativeCurrentDomain):
    def __init__(self,*args,development=False,**kwargs):
        super().__init__(*args,**kwargs);self.development=development
        self.roles={};self.feedback={}
        if development:
            patients=sorted(set(self._patients.values()))
            role={'online':set(patients[:2]),'audit':set(patients[2:4]),'fit':set(patients[4:])}
            assert len(role['fit'])>=2 and not role['online']&role['audit']
            original=self._l
            for k,ids in role.items():
                ds=copy.copy(original);ds.rows=[r for r in original.rows if self._patients[r['case_id']] in ids];ds.checked=set();self.roles[k]=sorted(ids)
                if k=='fit':self._l=ds
                else:self.feedback[k]=ds
    def clean_fit(self,step):
        # Same source pair as the supervised update, before geometry augmentation.
        epoch,index=divmod(step,self.steps_per_epoch);n=len(self._l);cycle,offset=divmod(2*index,n)
        order=torch.randperm(n,generator=generator(self.seed,self.order,self.stage,epoch,f'labeled/order/{cycle}')).tolist()
        rows=[self._l[order[offset]],self._l[order[(offset+1)%n]]]
        return torch.stack([r['image'] for r in rows]).to(self.device),torch.stack([r['label'] for r in rows]).to(self.device)

class Trainer(StageTrainer):
    action=2
    def components(self):
        labeled,_,_=super().losses()
        x,_,_=self.provider.labeled(self.cursor)
        u,valid,_=self.provider.unlabeled(self.cursor)
        with torch.no_grad():q=self.ema(u,mode='teacher').softmax(1)
        logp,_=collected_forward(lambda v,s:self.model(v,scale=s),u,x,self.rng('UL'))
        return labeled,logp.exp(),q,valid
    def losses(self):
        if self.action in (-1,0):return super().losses()
        labeled,p,q,valid=self.components();ul,_=u_loss(p,q,valid,self.action)
        self.last.update(active_U=True,unlabeled_loss=float(ul.detach())*.5)
        return labeled,.5*ul,None
    @contextmanager
    def readonly(self):
        rng=cpu(rng_state());modes=[(m,m.training) for m in self.model.modules()]
        buffers=[(b,b.detach().clone()) for b in self.model.buffers()]
        telemetry=copy.deepcopy(self.telemetry);last=copy.deepcopy(self.last);reads=(self.provider.l_reads,self.provider.u_reads)
        try:yield
        finally:
            with torch.no_grad():
                for b,v in buffers:b.copy_(v)
            for m,mode in modes:m.training=mode
            self.telemetry=telemetry;self.last=last;self.provider.l_reads,self.provider.u_reads=reads;rng_restore(rng)
    def clean(self,x):
        self.model.eval();p=self.model.parent
        return p.native_readout(p.features(x,'eval'),x.shape[-2:],'eval').softmax(1)
    def extract(self,full=True):
        with self.readonly():
            params=[p for p in self.model.parameters() if p.requires_grad];names=[n for n,p in self.model.named_parameters() if p.requires_grad]
            l,p,q,v=self.components();base,_=state_vector(q,p,v,l,self.step,self.optimizer.param_groups[0]['lr']/self.scheduler.base_lrs[0])
            if not full:return base.cpu().tolist()+[0.]*24
            targets=[l]+[l+.5*u_loss(p,q,v,a)[0] for a in (1,2)]
            grads=[list(torch.autograd.grad(loss,params,retain_graph=i<2,allow_unused=True)) for i,loss in enumerate(targets)]
            x,y=self.provider.clean_fit(self.cursor);score=quality(self.clean(x),y)
            # quality() is higher-is-better; feature equations expect loss gradient.
            qgrad=torch.autograd.grad(-score,params,allow_unused=True)
            previews=[effect.preview(self.optimizer,params,g)[0] for g in grads]
            z=base.cpu().tolist()+effect.features(params,names,grads,qgrad,previews)
            assert len(z)==40 and np.isfinite(z).all();return z
    def feedback_score(self,role):
        assert self.provider.development
        with self.readonly(),torch.no_grad():
            ds=self.provider.feedback[role];scores=[]
            for i in range(len(ds)):
                b=ds[i];scores.append(float(quality(self.clean(b['image'][None].to(self.provider.device)),b['label'][None].to(self.provider.device))))
            return float(np.mean(scores))


def snapshot(t):
    return cpu(dict(student=t.model.state_dict(),ema=t.ema.state_dict(),optimizer=t.optimizer.state_dict(),scheduler=t.scheduler.state_dict(),scaler=t.scaler.state_dict(),rng=rng_state(),step=t.step,cursor=t.cursor,epoch=t.epoch,probe=t.probe,telemetry=t.telemetry,last=t.last,action=t.action,reads=(t.provider.l_reads,t.provider.u_reads),modes=[m.training for m in t.model.modules()],grads=[p.grad for p in t.model.parameters()],prototypes=t.prototypes.values,support=t.prototypes.support))
def restore(t,v):
    t.model.load_state_dict(v['student']);t.ema.load_state_dict(v['ema']);t.optimizer.load_state_dict(copy.deepcopy(v['optimizer']));t.scheduler.load_state_dict(copy.deepcopy(v['scheduler']));t.scaler.load_state_dict(copy.deepcopy(v['scaler']))
    for k in ('step','cursor','epoch','probe','telemetry','last','action'):setattr(t,k,copy.deepcopy(v[k]))
    for m,mode in zip(t.model.modules(),v['modes']):m.training=mode
    for p,g in zip(t.model.parameters(),v['grads']):p.grad=None if g is None else g.to(p).clone()
    t.prototypes.values=copy.deepcopy(v['prototypes']);t.prototypes.support=copy.deepcopy(v['support'])
    t.provider.l_reads,t.provider.u_reads=v['reads'];rng_restore(v['rng']);t.requires_restore=False

def create(config,seed,domain,development,ledger):
    root=Path(config['source']);receipt=read(root/f'SOURCE_S{seed}'/'receipt.json');payload=torch.load(root/f'SOURCE_S{seed}'/'student.pt',map_location='cpu',weights_only=False)
    assert receipt['status']=='SEALED' and payload['step']==8000 and payload['identity']==receipt['identity'] and payload['identity']['seed']==seed
    assert tensor_fingerprint(payload['student'])==receipt['student_hash']
    source=dict(node_id=receipt['node_id'],student_hash=receipt['student_hash'],seed=seed,domain='REFUGE')
    order=DOMAINS.index(domain)+1;identity=dict(domain=domain,seed=seed,order=order,stage=1,manifest=primitive.MANIFEST_SHA,split=primitive.SPLIT_SHA)
    permit=ExecutionPermit(dict(execution_scope='formal',code_commit=config['commit'],authorized_manifest_digests=[digest(identity)]),('V3B',),dict(student=31626),_PERMIT_SEAL)
    provider=Provider(config['data'],seed,order,1,source,'cuda:0',permit,allow_u=True,development=development)
    native=build(config['reference'],'cuda:0',seed);native.load_state_dict(payload['student']);del payload
    model=Model(NativeLRParent(native,seed,source),'B0_PARENT_LCTX').to('cuda:0')
    opts=dict(lr=.001,weight_decay=4e-5,total_steps=3200 if order==1 else 2100,warmup_fraction=.2,U_ramp_fraction=.2,parent_lr_multiplier=.5,lr_B_over_A=1.,lambda_U=.5)
    t=Trainer(model,provider,opts,execution=permit);ledger.wrap(t);return t

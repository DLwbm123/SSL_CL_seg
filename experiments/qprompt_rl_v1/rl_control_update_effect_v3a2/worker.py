"""Read-isolated feature worker and frozen disposable student branch worker."""
import os,sys,copy,time,gc
from pathlib import Path
import numpy as np
import torch
from .runtime import *
from . import effect
from rl_control_bandit_v2.runner import Worker,cpu_copy,setup
from rl_control_bandit_v2.method import LocalSchedule,u_loss,quality
from rl_control_bandit_v2.data import UImages
from rl_control_bandit_v2.qualification import same
from rl_control_fine_intervention_v3a.execution import fingerprint
from qprompt.state import make_reference,ema_update

class Guard:
    def __init__(self):
        self.allowed=set();import h5py
        original=h5py.File
        def checked(name,*args,**kwargs):self.check(name);return original(name,*args,**kwargs)
        h5py.File=checked
        def audit(event,args):
            if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):self.check(args[0])
        sys.addaudithook(audit)
    def check(self,name):
        p=Path(os.fsdecode(name)).resolve()
        if p.suffix in ('.h5','.hdf5') and p not in self.allowed:raise PermissionError('feature process denied unbound payload')
        if p.is_relative_to(ROOT/'panels') or p.is_relative_to(ROOT/'fits') or p.is_relative_to(ROOT/'predictions') or p.is_relative_to(PREVIOUS/'tasks'):raise PermissionError('feature process denied feedback or learner artifacts')
    def bind(self,w,item):
        self.allowed={w.L.root/w.L.rows[i][k] for i in item['indices'] for k in ('image','label')};self.allowed.add(w.U.root/w.U.rows[item['u']]['relative'])

class Student(Worker):
    def __init__(self,cell,ledger,feature=False):
        self.task=cell;self.ledger=ledger;self.feature=feature;self.folder=ROOT/('features' if feature else 'panels')/cell;self.folder.mkdir(parents=True,exist_ok=True);self.kind='student';self.arm='V3A2'
        spec=read(ROOT/'CONTEXT_MANIFEST.private.json')[cell];assert spec['eligible'];seed=spec['seed'];backbone=spec['backbone'];domain=spec['domain'];setup(seed)
        self.meta=dict(task=cell,code_commit=CODE,config_sha=CONFIG,seed=seed,backbone=backbone,domain=domain,prefix_sha=spec['prefix']['sha256'],schedule_sha=spec['schedule_sha'])
        self.model,_=c.build(backbone,False,old.ASSETS,old.SOURCE);self.model.cuda().train();self.opt,sch=c.optimizer_for(self.model,backbone)
        p=spec['prefix'];assert digest(p['path'])==p['sha256'];state=torch.load(p['path'],weights_only=False,map_location='cpu');assert state['code_commit']==p['source_commit'] and state['global_step']==state['data_cursor']==state['scheduler']['last_epoch']==2000 and state['bank'] is None and state['reference'] is None and state['arm']=='QUERY_PREFIX';assert all(int(s['step'])==2000 for s in state['optimizer']['state'].values());c.restore(state,self.model,self.opt,sch);del state
        self.source_scheduler=copy.deepcopy(sch.state_dict());self.sch=LocalSchedule(self.opt);self.teacher=make_reference(self.model);self.policy=self.reference=self.popt=None;self.t=self.retained=0;self.elapsed=0.;self.model.zero_grad(set_to_none=True)
        self.L=old.dataset(domain,'train_labeled');self.U=UImages(os.environ['EXEC_U'],domain);assert self.L.rows==spec['l_rows'] and self.U.rows==spec['u_rows'];self.root=self.snapshot();self.root_fp=fingerprint(self.root)
    def snapshot(self):
        value=super().snapshot();value['grads']=[None if p.grad is None else p.grad.detach().cpu().clone() for p in self.model.parameters()];value['modes']={n:m.training for n,m in self.model.named_modules()};return value
    def restore(self,state):
        super().restore(state)
        for p,g in zip(self.model.parameters(),state['grads']):p.grad=None if g is None else g.to(p.device).clone()
        for n,m in self.model.named_modules():m.training=state['modes'][n]
    def reset(self):
        self.restore(self.root);assert same(self.snapshot(),self.root) and fingerprint(self.root)==self.root_fp
        for i,live in self.opt.state_dict()['state'].items():
            for k,v in live.items():
                if torch.is_tensor(v) and v.device.type=='cpu':assert v.untyped_storage().data_ptr()!=self.root['optimizer']['state'][i][k].untyped_storage().data_ptr()
    def charge_step(self,opt,kind,key):self.ledger.step(opt,kind,self.task+'/'+key)
    def score(self,indices,purpose):
        assert not self.feature,'feature process cannot score feedback';before=self.snapshot();result=super().score(indices,purpose);assert same(before,self.snapshot());return result
    def advance(self,x,y,us,q,v,action,key):
        info=self.student_step(x,y,us,q,v,action,key,temporary=True);ema_update(self.model,self.teacher,.99);self.t+=1;self.retained+=1;assert self.sch.t==self.t;assert all(int(s['step'])==2000+self.t for s in self.opt.state.values());return info
    def branch(self,scene,action,repeat):
        j=scene['j'];key=f'{j}/{repeat}/{action}';path=self.folder/(key.replace('/','_')+'.private.json')
        if path.exists():return receipt(path)
        if self.meta['seed']!=261:receipt(ROOT/'TRANSFER_PREDICTION_LOCK.json')
        self.reset();steps=[];scores={}
        for h,item in enumerate(scene['items'],1):
            x,y=old.load_batch(self.L,item,torch.device('cuda:0'));us,q,v,z,_=self.context(item,x,y);info=self.advance(x,y,us,q,v,action if h==1 else 2,key+'/'+str(h));steps.append(dict(h=h,teacher_updated=True,adam=2000+h,scheduler=self.sch.t,info=info))
            if h in (1,5):scores[str(h)]={role:self.score(item[role],role) for role in ('online','audit')}
        self.reset();save(path,cell=self.task,j=j,action=action,repeat=repeat,horizons=scores,steps=steps,root_fingerprint=self.root_fp,root_restored=True,retained_after_discard=0,prefix_sha=self.meta['prefix_sha'],schedule_sha=self.meta['schedule_sha'])

    def extract(self,scene,repeat,guard):
        key=f'{self.task}/{scene["j"]}/{repeat}';path=self.folder/f'{scene["j"]}_{repeat}.private.json'
        if path.exists():return receipt(path)
        guard.bind(self,scene['items'][0]);item=scene['items'][0]
        def body():
            start=time.time();self.reset();before=self.snapshot();x,y=old.load_batch(self.L,item,torch.device('cuda:0'));us,q,v,z,_=self.context(item,x,y);self.model.train();parameters=tuple(self.model.parameters());names=[n for n,p in self.model.named_parameters()]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=self.model(x)
                with torch.autocast('cuda',enabled=False):ls=c.supervised_query_loss(out['class_logits'].float(),out['mask_logits'].float(),y)['total']
                pu=self.model(us)['semantic'];losses=[ls,ls+.5*u_loss(pu,q,v,1)[0],ls+.5*u_loss(pu,q,v,2)[0]]
            gradients=[]
            for a in range(3):gradients.append(self.ledger.call('vjp',key+'/action/'+str(a),lambda a=a:torch.autograd.grad(losses[a],parameters,retain_graph=a<2,allow_unused=True)))
            del losses,ls,pu,out
            self.model.eval();xx=torch.stack([self.L[i]['image'] for i in item['indices']]).cuda();yy=torch.stack([self.L[i]['label'] for i in item['indices']]).cuda()
            with torch.autocast('cuda',enabled=False):q_loss=-quality(self.model(xx)['semantic'],yy)
            gq=self.ledger.call('vjp',key+'/clean',lambda:torch.autograd.grad(q_loss,parameters,allow_unused=True));del q_loss
            previews=[]
            for a in range(3):previews.append(self.ledger.call('preview',key+'/'+str(a),lambda a=a:effect.preview(self.opt,parameters,gradients[a])[0]))
            values=z.tolist()+effect.features(parameters,names,gradients,gq,previews);assert len(values)==40 and np.isfinite(values).all();self.model.train(before['model_training'])
            # Feature forwards are eval/train operations with no parameter installation.
            # Restore transient modes/RNG/buffers and verify the complete state including .grad.
            raw=self.snapshot()
            for k in ('student','optimizer','teacher','scheduler','t','retained','grads'):assert same(raw[k],before[k]),'feature mutated '+k
            self.restore(before);assert same(self.snapshot(),before);assert fingerprint(self.root)==self.root_fp
            save(path,cell=self.task,j=scene['j'],repeat=repeat,features=values,root_fingerprint=self.root_fp,invariant=True,student_forwards=5,teacher_forwards=1,vjp_calls=4,virtual_previews=3,seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),prefix_sha=self.meta['prefix_sha'],schedule_sha=self.meta['schedule_sha'])
            del gradients,previews,gq
            return read(path)
        return self.ledger.call('extraction',key,body)

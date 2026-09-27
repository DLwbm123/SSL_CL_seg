"""One worker per frozen trajectory, with restorable disposable updates."""
import copy,csv,gc,json,os,random,time,uuid,traceback
from pathlib import Path
import numpy as np
import torch
from r1_12h import core as c,runner as old
from r1_6_readout_v1.runtime import seed_value,process_identity
from qprompt.state import make_reference,ema_update
from qprompt.metrics import image_dice
from .runtime import PLAN,TASKS,SHA,rpc,save
from .method import *
from .data import UImages,make_schedule,u_views
RUN=old.RUN;CODE=old.CODE;DEVICE=torch.device('cuda:0')


def authorize():
    a=json.loads(Path(__file__).with_name('EXECUTION_AUTHORIZATION.json').read_text())
    assert a['authorization']=='USER_REQUEST_EXECUTE_PACKAGE' and a['R3a'] and a['R3b'] and not a['R2'] and not a['R3c_training']
    old.provenance()


def setup(seed):
    old.setup();random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);c.seed=lambda *parts:seed_value(seed,*parts)


def cpu_copy(x):
    if torch.is_tensor(x):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu_copy(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_copy(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu_copy(v) for v in x)
    return copy.deepcopy(x)


def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])
def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['cpu']);torch.cuda.set_rng_state_all(s['cuda'])


def prepare():
    authorize();r16=Path(os.environ['EXEC_R16']);previous=json.loads((r16/'PREFIX_AUDIT.private.json').read_text());audit={};schedules={};folds={}
    for domain,nl,nu in [('RIM_ONE_r3',16,63),('Drishti_GS',10,41)]:
        ds=old.dataset(domain,'train_labeled');assert len(ds)==nl;u=UImages(os.environ['EXEC_U'],domain);assert len(u)==nu
        for i in range(len(ds)):ds[i]
        for i in range(len(u)):u[i]
        for seed in (261,262,263):
            doc=make_schedule(seed,domain,nl,nu);doc['l_rows']=ds.rows;doc['u_rows']=u.rows
            p=RUN/'schedules'/f'{seed}__{domain}.private.json';c.atomic(p,doc);schedules[f'{seed}__{domain}']=c.file_sha(p);folds[domain]=dict(grouping='image',patient_independence_claim=False,fold_sizes=list(map(len,doc['folds'])),L=nl,U=nu)
    original_audit={x['task']:x for x in json.loads((r16/'reports/STATE_AUDIT.json').read_text())}
    for spec in PLAN['external_prefixes']:
        seed=spec['seed'];backbone=spec['backbone'];domain=spec['domain'];task=f's{seed}__{backbone}__{domain}__QUERY_PREFIX'
        try:
            if seed==261:path=Path(previous[task]['path']);expected=previous[task]['sha256'];commit=previous[task]['source_commit']
            else:
                path=r16/'tasks'/task/'prefix.pt';commit=original_audit[task]['training_code_commit']
                endpoint=next(r16.joinpath('tasks').glob(f's{seed}__{backbone}__{domain}__B0/EVALUATION.json'));expected=json.loads(endpoint.read_text())['prefix_sha256']
            assert c.file_sha(path)==expected
            s=torch.load(path,map_location='cpu',weights_only=False);assert s['code_commit']==commit and s['arm']=='QUERY_PREFIX' and s['global_step']==s['data_cursor']==s['scheduler']['last_epoch']==2000
            assert s['backbone']==backbone and s['domain']==domain and all(torch.isfinite(v).all() for v in s['student'].values())
            assert all(int(v['step'])==2000 for v in s['optimizer']['state'].values())
            assert all(torch.isfinite(v).all() for row in s['optimizer']['state'].values() for v in row.values() if torch.is_tensor(v)) and s['bank'] is None and s['reference'] is None
            audit[spec['id']]=dict(valid=True,path=str(path),sha256=expected,source_commit=commit,entry_lr=[g['lr'] for g in s['optimizer']['param_groups']],step=2000);del s
        except Exception as e:audit[spec['id']]=dict(valid=False,error=repr(e))
    c.atomic(RUN/'PREFIX_BINDINGS.private.json',audit);c.atomic(RUN/'reports/PREFIX_BINDINGS.json',{k:{a:b for a,b in v.items() if a not in ('path','error')} for k,v in audit.items()})
    c.atomic(RUN/'DATA_BINDING.json',dict(manifest_sha=old.MANIFEST_SHA,split_sha=old.SPLIT_SHA,schedules=schedules,folds=folds,U_label_payloads=0))
    c.atomic(RUN/'reports/FOLD_AND_DATA_BINDING.json',json.loads((RUN/'DATA_BINDING.json').read_text()))
    c.atomic(RUN/'ENVIRONMENT.json',dict(python=__import__('sys').version,torch=torch.__version__,numpy=np.__version__))


class Worker:
    def __init__(self,spec,kind='r3b_student'):
        self.spec=spec;self.kind=kind;self.task=spec['id'];self.arm=spec.get('arm','R3A');self.folder=RUN/'tasks'/self.task;self.folder.mkdir(parents=True,exist_ok=True);self.token=uuid.uuid4().hex
        setup(spec['seed']);authorize();rpc(RUN,action='claim',task=self.task,token=self.token,pid=os.getpid(),identity=process_identity(os.getpid()))
        self.meta=dict(task=self.task,code_commit=CODE,config_sha=SHA,seed=spec['seed'],domain=spec['domain'],backbone=spec['backbone'],arm=self.arm,controller_version='bandit-v2-anchored-mixed-policy')
        p=RUN/'schedules'/f"{spec['seed']}__{spec['domain']}.private.json";self.meta['schedule_sha']=c.file_sha(p);assert self.meta['schedule_sha']==json.loads((RUN/'DATA_BINDING.json').read_text())['schedules'][f"{spec['seed']}__{spec['domain']}"];self.schedule=json.loads(p.read_text())
        ds=old.dataset(spec['domain'],'train_labeled');assert ds.rows==self.schedule['l_rows'];self.L=[ds[i] for i in range(len(ds))];self.U=None
        if self.arm!='SUP':
            u=UImages(os.environ['EXEC_U'],spec['domain']);assert u.rows==self.schedule['u_rows'];self.U=[u[i] for i in range(len(u))]
        binding=json.loads((RUN/'PREFIX_BINDINGS.private.json').read_text())[spec['dependencies'][0]];assert binding['valid'];self.meta.update(prefix_sha=binding['sha256'],prefix_source_commit=binding['source_commit'])
        self.model,_=c.build(spec['backbone'],False,old.ASSETS,old.SOURCE);self.model.cuda().train();self.opt,source_sch=c.optimizer_for(self.model,spec['backbone']);state=torch.load(binding['path'],map_location='cpu',weights_only=False);assert c.file_sha(binding['path'])==binding['sha256'];c.restore(state,self.model,self.opt,source_sch);self.source_scheduler=state['scheduler'];del state
        self.sch=LocalSchedule(self.opt);self.teacher=make_reference(self.model);self.t=0;self.retained=0;self.elapsed=0.
        self.policy=self.reference=self.popt=None
        if self.arm in ADAPTIVE:
            with torch.random.fork_rng(devices=[0]):torch.manual_seed(seed_value(spec['seed'],'bandit-policy',spec['backbone'],spec['domain']));self.policy=Controller(self.arm).cuda()
            self.reference=make_reference(self.policy);self.popt=torch.optim.Adam(self.policy.parameters(),lr=3e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
        if kind=='r3b_student' and (self.folder/'latest.pt').exists():
            s=torch.load(self.folder/'latest.pt',map_location='cpu',weights_only=False);assert s['code_commit']==CODE and s['config_sha']==SHA and s['prefix_sha']==self.meta['prefix_sha'] and s['schedule_sha']==self.meta['schedule_sha'];self.restore(s);del s
    def snapshot(self):
        return cpu_copy(dict(**self.meta,student=self.model.state_dict(),optimizer=self.opt.state_dict(),scheduler=self.sch.state_dict(),teacher=self.teacher.state_dict(),rng=rng(),policy=None if self.policy is None else self.policy.state_dict(),policy_reference=None if self.reference is None else self.reference.state_dict(),policy_optimizer=None if self.popt is None else self.popt.state_dict(),t=self.t,retained=self.retained,source_scheduler=self.source_scheduler,elapsed=self.elapsed,model_training=self.model.training,generators='stateless explicit seeds bound to schedule',precision='BF16 fit; FP32 feedback and KL; FP32 master',trainable=[n for n,p in self.model.named_parameters() if p.requires_grad]))
    def restore(self,s):
        self.model.load_state_dict(s['student']);self.opt.load_state_dict(s['optimizer']);self.sch.load_state_dict(s['scheduler']);self.teacher.load_state_dict(s['teacher']);self.t=s['t'];self.retained=s['retained'];self.elapsed=s['elapsed'];self.model.train(s['model_training'])
        if self.policy is not None:self.policy.load_state_dict(s['policy']);self.reference.load_state_dict(s['policy_reference']);self.popt.load_state_dict(s['policy_optimizer'])
        restore_rng(s['rng'])
    def charge_step(self,opt,kind,transaction):
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for g in opt.param_groups for p in g['params']):raise FloatingPointError('optimizer gradient nonfinite')
        grant=rpc(RUN,action='attempt',task=self.task,token=self.token,kind=kind,transaction=transaction,device='cuda');c.append(self.folder/'TRANSACTIONS.jsonl',dict(event='attempt',**grant))
        try:opt.step()
        except BaseException:c.append(self.folder/'TRANSACTIONS.jsonl',dict(event='optimizer_failed',**grant));raise
        c.append(self.folder/'TRANSACTIONS.jsonl',dict(event='optimizer_success',**grant))
    def student_step(self,x,y,us,q,v,a,transaction,temporary=False,extra=False):
        self.model.train();self.opt.zero_grad(set_to_none=True)
        c.append(self.folder/'COST_EVENTS.jsonl',dict(event='label_forward',purpose='extra_online_fit' if extra else 'fit',images=len(y),student_forwards=1+int(a!=0),teacher_forwards=0,transaction=transaction))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=self.model(x)
            with torch.autocast('cuda',enabled=False):ls=supervised_query_loss(out['class_logits'].float(),out['mask_logits'].float(),y)['total']
            if a:
                pu=self.model(us)['semantic'];lu,coverage=u_loss(pu,q,v,a)
            else:lu=ls*0;coverage={}
            total=ls+.5*lu
        if not torch.isfinite(total):raise FloatingPointError('student loss nonfinite')
        ug=None
        if a and (temporary or self.t%100==0):
            gg=torch.autograd.grad(.5*lu,tuple(self.model.parameters()),retain_graph=True,allow_unused=True);ug=float(torch.stack([g.float().square().sum() for g in gg if g is not None]).sum().sqrt());del gg
        total.backward();norm=torch.stack([p.grad.float().square().sum() for p in self.model.parameters() if p.grad is not None]).sum().sqrt()
        if not torch.isfinite(norm):raise FloatingPointError('student gradient nonfinite')
        self.charge_step(self.opt,self.kind,transaction)
        if not extra:self.sch.step()
        if not temporary:ema_update(self.model,self.teacher,.99);self.retained+=1
        c.append(self.folder/'TRANSACTIONS.jsonl',dict(event='student_commit',transaction=transaction,temporary=temporary,extra=extra,t=self.t,retained=self.retained))
        return dict(Lset=float(ls.detach()),LU=float(lu.detach()),grad_norm=float(norm),U_grad_norm=ug,coverage=coverage)
    def context(self,item,x,y):
        weak,us,v=u_views(self.U[item['u']],item,DEVICE);before=rng();mode=self.model.training;self.model.eval()
        c.append(self.folder/'COST_EVENTS.jsonl',dict(event='label_forward',purpose='context_fit',images=len(y),student_forwards=2,teacher_forwards=1,t=self.t))
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            q=self.teacher(weak)['semantic'].detach();p=self.model(weak)['semantic'];out=self.model(x)
        with torch.no_grad(),torch.autocast('cuda',enabled=False):
            ls=supervised_query_loss(out['class_logits'].float(),out['mask_logits'].float(),y)['total'];z,flags=state_vector(q,p,v,ls,self.t,self.opt.param_groups[0]['lr']/self.sch.entry[0])
        self.model.train(mode);restore_rng(before);return us,q,v,z,flags
    def score(self,indices,purpose):
        before=rng();mode=self.model.training;self.model.eval()
        c.append(self.folder/'COST_EVENTS.jsonl',dict(event='label_forward',purpose=purpose,images=len(indices),student_forwards=1,teacher_forwards=0,t=self.t))
        with torch.inference_mode(),torch.autocast('cuda',enabled=False):
            x=torch.stack([self.L[i]['image'] for i in indices]).cuda();y=torch.stack([self.L[i]['label'] for i in indices]).cuda()
            value=quality(self.model(x)['semantic'],y);result=None if value is None else float(value)
        self.model.train(mode);restore_rng(before);return result
    def candidate(self,state,item,x,y,us,q,v,a,transaction):
        self.restore(state);info=self.student_step(x,y,us,q,v,a,transaction,temporary=True);online=self.score(item['online'],'online_reward');audit=self.score(item['audit'],'audit_only');self.restore(state)
        return dict(action=a,online=online,audit=audit,info=info)
    def choose(self,z,item):
        if self.arm=='SUP':return 0,None
        if self.arm in ('FIX_FINE','FIX_FINE_EXTRA_L'):return 2,None
        if self.arm=='FIX_COARSE':return 1,None
        if self.arm=='RULE':return (0 if float(z[9])>.5 else 1 if float(z[10])>.5 else 2),None
        prob=torch.ones(3,device=DEVICE)/3 if self.arm=='RANDOM' else self.policy.distribution(z).detach().clone()
        return draw(prob,item['actual_seed'])[0],prob
    def run(self):
        start=time.monotonic();deadline=json.loads((RUN/'SESSION.json').read_text())['deadline'];save(self.folder,self.snapshot())
        while self.t<1200 and time.time()<deadline:
            if (RUN/'ENGINEERING_PAUSE.json').exists():break
            item=self.schedule['steps'][self.t];decision=self.t%20==0
            if decision:save(self.folder,self.snapshot())
            x,y=old.load_batch(self.L,item,DEVICE)
            us=q=v=z=None;flags={}
            if self.arm!='SUP':us,q,v,z,flags=self.context(item,x,y)
            action,behavior=self.choose(z,item);record=dict(meta=dict(t=self.t,arm=self.arm),runtime={},losses={},state=None if z is None else z.tolist(),flags=flags,action=action)
            if decision:
                # Pre-probe durable seal survives restart; only future actions see this reward.
                seal=self.folder/'sealed'/f'{self.t}.json'
                if seal.exists():
                    sealed=json.loads(seal.read_text());assert sealed['arm']==self.arm and sealed['schedule_sha']==self.meta['schedule_sha'];action=sealed['action'];record['action']=action
                    if sealed['behavior'] is not None:behavior=torch.tensor(sealed['behavior'],device=DEVICE)
                    if sealed.get('state') is not None:z=torch.tensor(sealed['state'],device=DEVICE);record['state']=sealed['state']
                else:c.atomic(seal,dict(t=self.t,arm=self.arm,action=action,state=None if z is None else z.tolist(),behavior=None if behavior is None else behavior.tolist(),schedule_sha=self.meta['schedule_sha']))
            if decision and self.arm in ADAPTIVE:
                state=self.snapshot();anchor=self.candidate(state,item,x,y,us,q,v,0,f'{self.t}/anchor');actions=draw(behavior,item['probe_seed'],4);candidates=[self.candidate(state,item,x,y,us,q,v,a,f'{self.t}/probe/{j}') for j,a in enumerate(actions)]
                assert self.t==state['t'] and self.sch.state_dict()==state['scheduler'] and c.tensor_sha(self.model.state_dict())==c.tensor_sha(state['student'])
                scale=json.loads((RUN/'tasks'/self.spec['dependencies'][1]/'R3A_DONE.json').read_text())['scale'];valid=anchor['online'] is not None and all(p['online'] is not None for p in candidates)
                rewards=[p['online']-anchor['online'] for p in candidates] if valid else None;stats=[]
                if valid and (self.arm=='REG' or any(r!=0 for r in rewards)):
                    for j in range(4):
                        self.popt.zero_grad(set_to_none=True);loss,diag=controller_loss(self.policy,self.reference,z,torch.tensor(actions,device=DEVICE),torch.tensor(rewards,device=DEVICE),behavior,scale);assert torch.isfinite(loss);loss.backward();self.charge_step(self.popt,'controller',f'{self.t}/controller/{j}');stats.append(dict(loss=float(loss.detach()),**diag))
                    ema_update(self.policy,self.reference,.99)
                record['feedback']=dict(anchor=anchor,candidates=candidates,raw_rewards=rewards,scale=scale,actions=actions,controller=stats,behavior=behavior.tolist(),future_distribution=self.policy.distribution(z).detach().tolist(),online_images=len(item['online']),audit_images=len(item['audit']))
                del state
            if decision:c.atomic(self.folder/'decisions'/f'{self.t}.private.json',record)
            info=self.student_step(x,y,us,q,v,action,f'{self.t}/actual');record['losses']=info
            if self.arm=='FIX_FINE_EXTRA_L' and decision:
                # Extras follow ordinary update, at that decision's pre-update LR; ordinary clock advances once.
                next_t=self.sch.t;self.sch.t=self.t;self.sch.apply()
                for j,extra in enumerate(item['extra']):
                    ex,ey=old.load_batch(self.L,extra,DEVICE);self.student_step(ex,ey,None,None,None,0,f'{self.t}/extra/{j}',extra=True)
                self.sch.t=next_t;self.sch.apply()
            c.append(self.folder/'ACTION_LOG.jsonl',dict(t=self.t,action=action,distribution=None if behavior is None else behavior.tolist()))
            self.t+=1;record['runtime']=dict(ordinary=self.t,retained=self.retained,lr=[g['lr'] for g in self.opt.param_groups],elapsed=self.elapsed+time.monotonic()-start)
            if decision or self.t%100==0 or self.t==1200:
                self.elapsed+=time.monotonic()-start;start=time.monotonic();save(self.folder,self.snapshot(),final=self.t==1200)
                c.append(self.folder/'DIAGNOSTICS.private.jsonl',record)
        if self.t==1200:c.atomic(self.folder/'TRAIN_DONE.json',dict(**self.meta,ordinary=self.t,retained=self.retained,cumulative=2000+self.retained,elapsed=self.elapsed))
        else:save(self.folder,self.snapshot());c.atomic(self.folder/('PAUSED.json' if (RUN/'ENGINEERING_PAUSE.json').exists() else 'TIME_LIMIT.json'),dict(t=self.t,retained=self.retained))
    def audit(self):
        original=self.snapshot();rows=[]
        for j in range(16):
            if (RUN/'ENGINEERING_PAUSE.json').exists():raise RuntimeError('engineering pause at audit boundary')
            if (self.folder/'contexts'/f'{j}.json').exists():rows.append(json.loads((self.folder/'contexts'/f'{j}.json').read_text()));continue
            self.restore(original);item=self.schedule['steps'][20*(j*60//16)];x,y=old.load_batch(self.L,item,DEVICE);us,q,v,z,flags=self.context(item,x,y)
            candidates=[self.candidate(original,item,x,y,us,q,v,a,f'audit/{j}/{k}') for k,a in enumerate((0,1,2,0,2))]
            row=dict(context=j,schedule_t=item['t'],state=z.tolist(),flags=flags,candidates=candidates,online_rewards=[p['online']-candidates[0]['online'] if p['online'] is not None and candidates[0]['online'] is not None else None for p in candidates],audit_rewards=[p['audit']-candidates[0]['audit'] if p['audit'] is not None and candidates[0]['audit'] is not None else None for p in candidates]);c.atomic(self.folder/'contexts'/f'{j}.json',row);rows.append(row)
        rewards=[r['online_rewards'][a] for r in rows for a in (1,2) if r['online_rewards'][a] is not None];assert len(rewards)==32,'invalid R3a feedback';scale=max(1e-4,float(np.sqrt(np.mean(np.square(rewards)))))
        assert c.tensor_sha(self.model.state_dict())==c.tensor_sha(original['student']) and self.retained==0
        c.atomic(self.folder/'R3A_DONE.json',dict(**self.meta,contexts=16,student_calls=80,scale=scale,retained=0));c.atomic(self.folder/'R3A.private.json',rows)


def evaluate():
    spec=TASKS[os.environ['EXEC_TASK']];setup(spec['seed']);folder=RUN/'tasks'/spec['id'];s=torch.load(folder/'final.pt',map_location='cpu',weights_only=False);assert s['t']==1200 and s['retained']==spec['retained_new_student_updates']
    assert s['code_commit']==CODE and s['config_sha']==SHA and s['scheduler']['t']==1200
    binding=json.loads((RUN/'PREFIX_BINDINGS.private.json').read_text())[spec['dependencies'][0]];assert s['prefix_sha']==binding['sha256'] and s['prefix_source_commit']==binding['source_commit']
    assert s['schedule_sha']==json.loads((RUN/'DATA_BINDING.json').read_text())['schedules'][f"{spec['seed']}__{spec['domain']}"]
    assert all(int(v['step'])==2000+s['retained'] for v in s['optimizer']['state'].values()) and all(g['lr']==0 for g in s['optimizer']['param_groups'])
    assert all(torch.isfinite(v).all() for key in ('student','teacher') for v in s[key].values())
    assert all(torch.isfinite(v).all() for row in s['optimizer']['state'].values() for v in row.values() if torch.is_tensor(v))
    assert set(s['rng'])=={'python','numpy','cpu','cuda'} and len(s['rng']['cuda'])==1
    if spec['arm'] in ADAPTIVE:assert s['policy'] is not None and s['policy_reference'] is not None and s['policy_optimizer'] is not None
    else:assert s['policy'] is None and s['policy_reference'] is None and s['policy_optimizer'] is None
    c.atomic(folder/'STATE_AUDIT.json',dict(task=spec['id'],passed=True,ordinary=s['t'],retained=s['retained'],optimizer_step=2000+s['retained'],code_commit=CODE,prefix_sha=s['prefix_sha'],schedule_sha=s['schedule_sha'],rng_present=True,teacher_finite=True,terminal_lr=0))
    model,_=c.build(spec['backbone'],False,old.ASSETS,old.SOURCE);model.load_state_dict(s['student']);model.cuda().eval();ds=old.dataset(spec['domain'],'val');values=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for i in range(len(ds)):
            sample=ds[i];out=model(sample['image'][None].cuda())['semantic'];values.append(image_dice(out[0].argmax(0).cpu(),sample['label']))
    scores={k:float(np.mean([v[k] for v in values if v[k] is not None])) if any(v[k] is not None for v in values) else None for k in ('rim','cup','macro','disc_union')}
    c.atomic(folder/'EVALUATION.json',dict(**{k:s[k] for k in ('task','seed','domain','backbone','arm','code_commit','config_sha','prefix_sha','schedule_sha','controller_version')},**scores,support=len(values),rim_support=sum(v['rim'] is not None for v in values),cup_support=sum(v['cup'] is not None for v in values),ordinary=1200,retained=s['retained'],checkpoint_sha=c.file_sha(folder/'final.pt'),complete=True))


def main():
    mode=os.environ['EXEC_MODE']
    if mode=='prepare':prepare()
    elif mode=='evaluate':evaluate()
    elif mode in ('train','audit'):
        w=Worker(TASKS[os.environ['EXEC_TASK']],kind='r3a_student' if mode=='audit' else 'r3b_student')
        try:w.audit() if mode=='audit' else w.run()
        except BaseException as e:c.atomic(w.folder/f'FAILURE_{time.time_ns()}.private.json',dict(error=repr(e),traceback=traceback.format_exc(),t=w.t,retained=w.retained));raise
        finally:rpc(RUN,action='release',task=w.task,token=w.token)
    elif mode=='supervise':
        from .supervisor import supervise
        supervise()
    elif mode=='qualify':
        from .qualification import qualify
        qualify()
    else:raise ValueError(mode)

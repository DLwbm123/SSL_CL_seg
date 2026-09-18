"""Generated native suite helpers; future CUDA/smoke require fresh authority."""
import copy
import random
from pathlib import Path
import torch
from .protocol import STUDY, DOC, CAPS, read, write, digest, canonical_plan, execution_plan, smoke_nodes
from .authority import preflight, synthetic_capability, Capability, _SEAL, baseline_environment
from .trainer import AGMSTrainer
from .model import AGMSModel
from ..five_frameworks_v1.recipes import SyntheticCurrentDomain
from ..five_frameworks_v1.native_parent import build, NativeLRParent
from ..five_frameworks_v1.model import Model
from ..five_frameworks_v1.train_stage import StageTrainer
from ..five_frameworks_v1.native_runner import Counter
from ..f5_confirmation_v1.assurance import environment, bind_environment, cost_session, session_totals
from .execution import owned_root, choose_gpu, ledger_count, accept_prefix, construct


class Generated(SyntheticCurrentDomain):
    def __init__(self,*a,device='cpu',**kw):
        super().__init__(*a,**kw);self.device=torch.device(device)
    def labeled(self,*a,**kw):
        x,y,ids=super().labeled(*a,**kw)
        # Fixed synthetic geometry contains all labels and an ignore hole.
        y=(torch.arange(self.size)[None,None,:].expand(2,self.size,-1)%3).long().clone()
        y[:,self.size//2,self.size//2]=255
        return x.to(self.device),y.to(self.device),ids
    def unlabeled(self,*a,**kw):
        x,g,ids=super().unlabeled(*a,**kw)
        return x.to(self.device),g.to(self.device),ids


def generated_trainer(reference,arm,device,permit,size=32,baseline=False):
    from ..f5_confirmation_v1.native_qualification import foreground_fixture
    provider=Generated(seed=163,order=1,stage=2,size=size,device=device,
                       stage_source={'kind':'generated','domain':'RIM_ONE_r3','prefix_binding_sha256':'generated_only'})
    native=build(reference,device,163);foreground_fixture(native)
    opts={**canonical_plan()['options']['Drishti_GS'],'total_steps':5,'PAS_confidence':.6,'PAS_cosine':.7}
    parent=NativeLRParent(native,163,provider.stage_source)
    if baseline:return StageTrainer(Model(parent,'B2_PARENT_PAS_KL').to(device),provider,opts,execution=permit)
    return AGMSTrainer(AGMSModel(parent,arm,163,1,2).to(device),provider,opts,arm,permit)


def exact(a,b):
    if isinstance(a,torch.Tensor):
        if not torch.equal(a,b):raise AssertionError('native state tensor differs')
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise AssertionError('state keys differ')
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b,(a,b)


def snapshot(t,common=False):
    v=dict(student=copy.deepcopy(t.model.state_dict()),ema=copy.deepcopy(t.ema.state_dict()),
           optimizer=copy.deepcopy(t.optimizer.state_dict()),scheduler=copy.deepcopy(t.scheduler.state_dict()),
           prototypes=t.prototypes.values.clone(),support=t.prototypes.support.clone(),step=t.step,cursor=t.cursor,
           reads=[t.provider.l_reads,t.provider.u_reads],rng=torch.get_rng_state(),python=random.getstate(),
           telemetry=copy.deepcopy(t.telemetry))
    if not common:v.update(risk=t.risk.clone(),risk_updates=t.risk_updates,diagnostics=copy.deepcopy(t.diagnostics),
                           support_summary=copy.deepcopy(t.support_summary),extra_cost=copy.deepcopy(t.extra_cost))
    return v


def baseline_equivalence(reference,device,permit,counter,size):
    values=[]
    for old in (True,False):
        torch.manual_seed(381);random.seed(381)
        t=generated_trainer(reference,'A0',device,permit,size,baseline=old);counter.wrap(t.optimizer)
        for _ in range(2):t.update()
        values.append(snapshot(t,common=True));del t
    exact(*values)
    return dict(main_EMA_optimizer_prototypes_reads_RNG_exact=True,calls=4,status='PASS')


def generated_identity(t):
    return dict(study_id=STUDY,arm=t.arm,family='B2_PARENT_PAS_KL',seed=163,order=1,stage=2,node_id='GENERATED_'+t.arm)


def continuation(reference,arm,device,permit,counter,root,size):
    from .state import save,resume
    t=generated_trainer(reference,arm,device,permit,size);counter.wrap(t.optimizer);t.update()
    assert t.provider.u_reads==0
    path=Path(root)/(arm+'.pt');ident=generated_identity(t);save(t,path,ident)
    for _ in range(2):t.update()
    expected=snapshot(t)
    p=Generated(seed=163,order=1,stage=2,size=size,device=device,stage_source=t.provider.stage_source)
    r=resume(path,reference,device,p,t.options,ident,permit,arm);counter.wrap(r.optimizer)
    for _ in range(2):r.update()
    exact(expected,snapshot(r))
    return r,path


def failures(reference,device,permit,counter,size):
    for arm in ('A1','A5'):
        t=generated_trainer(reference,arm,device,permit,size)
        # Exercise active branch without advancing any scientific or optimizer state.
        t.step=t.cursor=1;counter.wrap(t.optimizer)
        risk=t.risk.clone();diag=copy.deepcopy(t.diagnostics);before=counter.count
        try:t.update(fault='after_optimizer')
        except RuntimeError as e:assert 'injected after_optimizer' in str(e)
        else:raise AssertionError('failure injection absent')
        assert t.step==1 and counter.count==before+1 and t.requires_restore
        assert torch.equal(risk,t.risk) and diag==t.diagnostics
        try:t.update()
        except RuntimeError:pass
        else:raise AssertionError('uncommitted failed state reused')


def validate_receipt(r,mode,config,plan,root):
    specs=execution_plan()[mode];ledger='cuda_physical.jsonl' if mode=='CUDA' else 'smoke_physical.jsonl'
    env=read(root/'RUNTIME_ENVIRONMENT.json');baseline_environment(env['fingerprint'])
    if (r['status']!='PASS' or r['execution_commit']!=config['execution_commit'] or r['plan_sha256']!=plan['plan_sha256']
            or r['execution_sha256']!=digest(execution_plan()) or r['environment_sha256']!=digest(env['fingerprint'])
            or env['sha256']!=r['environment_sha256'] or r['physical_calls']!=specs['planned_calls']
            or ledger_count(root/ledger)!=r['physical_calls'] or len(r['rows'])!=len(specs['cases'])):
        raise PermissionError('qualification evidence mismatch')
    pos=0
    for row,spec in zip(r['rows'],specs['cases']):
        if any(row.get(k)!=v for k,v in spec.items()) or row['passed'] is not True or row['start']!=pos+1 or row['end']!=pos+spec['calls']:
            raise PermissionError('qualification case/range mismatch')
        pos=row['end']


def qualify(config,mode):
    if mode not in ('CUDA','smoke'):raise ValueError('finite mode required')
    plan,production=preflight(config);choose_gpu()
    from ..single_teacher_scd_v0_1.engine import precision
    precision();device=torch.device('cuda:0');torch.cuda.set_device(device)
    name='CUDA_QUALIFICATION' if mode=='CUDA' else 'SMOKE'
    with owned_root(config,plan) as root:
        env=bind_environment(root,baseline_environment(environment()));path=root/(name+'.json')
        ledger=root/('cuda_physical.jsonl' if mode=='CUDA' else 'smoke_physical.jsonl')
        if path.exists():validate_receipt(read(path),mode,config,plan,root);session_totals(root/'costs'/name);return read(path)
        if ledger_count(ledger) or list((root/'costs'/name).glob('*/session.json')):raise RuntimeError('partial qualification; no automatic retry')
        if mode=='smoke':validate_receipt(read(root/'CUDA_QUALIFICATION.json'),'CUDA',config,plan,root);session_totals(root/'costs'/'CUDA_QUALIFICATION')
        permit=synthetic_capability('synthetic',production) if mode=='CUDA' else Capability({**production.bindings,'execution_scope':'smoke'},CAPS,_SEAL)
        counter=Counter(ledger,execution_plan()[mode]['cap']);rows=[]
        try:
            prefixes={}
            if mode=='smoke':
                # Verify both order prefixes, even though six smoke arms use O1 L.
                for order in (1,2):
                    n=next(n for n in plan['nodes'] if n['order']==order)
                    prefixes[order]=accept_prefix(n,config,plan,permit,root,device)
            with cost_session(root/'costs'/name,name):
                for spec in execution_plan()[mode]['cases']:
                    start=counter.count
                    if mode=='CUDA':
                        if spec['kind']=='continuation':
                            t,_=continuation(config['reference'],spec['arm'],device,permit,counter,root,384);del t
                        elif spec['kind']=='baseline':baseline_equivalence(config['reference'],device,permit,counter,384)
                        else:failures(config['reference'],device,permit,counter,384)
                    else:
                        n=next(n for n in smoke_nodes() if n['arm']==spec['arm'] and n['order']==spec['order'])
                        t=construct(n,config,plan,permit,*prefixes[n['order']],device,smoke=True);counter.wrap(t.optimizer)
                        for _ in range(4):t.update()
                        assert t.provider.u_reads==0 and not t.diagnostics;del t
                    rows.append({**spec,'start':start+1,'end':counter.count,'passed':True})
                    torch.cuda.empty_cache()
            result=dict(status='PASS',execution_commit=config['execution_commit'],plan_sha256=plan['plan_sha256'],
                        execution_sha256=digest(execution_plan()),environment_sha256=env,physical_calls=counter.count,rows=rows)
            write(path,result);validate_receipt(result,mode,config,plan,root);return result
        except BaseException as e:
            write(path,dict(status='ENGINEERING_STOP',error=repr(e),physical_calls=counter.count));raise

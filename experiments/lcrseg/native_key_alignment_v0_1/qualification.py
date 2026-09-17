"""Finite native CUDA and L-only smoke definitions; both require fresh authority."""
import copy
import random
from pathlib import Path
from .protocol import read,write,digest
from .authority import preflight,synthetic_capability,Capability,_SEAL,execution_plan
from .execution import owned_root,choose_gpu,ledger_count,accept_prefix,construct,identity
from ..f5_confirmation_v1.assurance import bind_environment,environment,cost_session,session_totals


def equal(a,b):
    import torch
    if isinstance(a,torch.Tensor):
        assert a.shape==b.shape and a.dtype==b.dtype
        assert torch.allclose(a.cpu(),b.cpu(),rtol=1e-5,atol=2e-6) if a.is_floating_point() else torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def snapshot(t):
    import torch
    return copy.deepcopy(dict(model=t.model.state_dict(),ema=t.ema.state_dict(),optimizer=t.optimizer.state_dict(),
        scheduler=t.scheduler.state_dict(),prototypes=t.prototypes.values,support=t.prototypes.support,
        basis=t.align_basis,physical_optimizer_updates=t.physical_optimizer_updates,step=t.step,cursor=t.cursor,reads=(t.provider.l_reads,t.provider.u_reads),
        diagnostics=t.diagnostics,alignment_cost=t.alignment_cost,support_summary=t.support_summary,
        torch_rng=torch.get_rng_state(),python_rng=random.getstate(),
        cuda_rng=torch.cuda.get_rng_state_all() if next(t.model.parameters()).is_cuda else []))


def generated_trainer(reference,arm,device,permit,size=384):
    import torch
    from .tests import Generated
    from ..five_frameworks_v1.native_parent import build,NativeLRParent
    from ..five_frameworks_v1.model import Model
    from ..f5_confirmation_v1.native_qualification import foreground_fixture
    from .trainer import KeyAlignmentTrainer
    from .protocol import canonical_plan
    class DeviceGenerated(Generated):
        def labeled(self,*a,**kw):
            x,y,ids=super().labeled(*a,**kw);return x.to(device),y.to(device),ids
        def unlabeled(self,*a,**kw):
            x,g,ids=super().unlabeled(*a,**kw);return x.to(device),g.to(device),ids
    provider=DeviceGenerated(seed=163,order=1,stage=2,size=size,stage_source={'kind':'generated','domain':'RIM_ONE_r3','prefix_binding_sha256':'generated_only'})
    native=build(reference,device,163);foreground_fixture(native)
    options={**canonical_plan()['options']['Drishti_GS'],'total_steps':5,'PAS_confidence':0.,'PAS_cosine':-1.}
    return KeyAlignmentTrainer(Model(NativeLRParent(native,163,provider.stage_source),'B2_PARENT_PAS_KL').to(device),
                                provider,options,arm,permit)


def cuda_cases(config,plan,permit,root,device,counter):
    import torch
    from .state import save,resume
    rows=[]
    for arm in ('C0','C1','C2','C3'):
        t=generated_trainer(config['reference'],arm,device,permit);counter.wrap(t.optimizer)
        ident=dict(study_id=plan['study_id'],arm=arm,family='B2_PARENT_PAS_KL',seed=163,order=1,stage=2,node_id='SYNTHETIC_'+arm)
        start=counter.count;t.update()
        assert t.native and t.provider.u_reads==0 and __import__('math').isfinite(t.last['labeled_loss'])
        rows.append(dict(arm=arm,kind='warmup',calls=1,start=start+1,end=counter.count,passed=True))
        path=root/(arm+'_generated.pt');save(t,path,ident)
        start=counter.count
        for _ in range(2):t.update()
        assert t.last['active_U'] and (arm=='C0' or t.last['alignment']['valid_classes']>0)
        t.model.parent.apply_constraints();expected=snapshot(t)
        # Create a fresh generated provider only, no extra trainer initialization.
        p=type(t.provider)(seed=163,order=1,stage=2,size=384,stage_source=t.provider.stage_source)
        r=resume(path,config['reference'],device,p,t.options,ident,permit,arm);counter.wrap(r.optimizer)
        for _ in range(2):r.update()
        equal(expected,snapshot(r))
        assert torch.equal(expected['torch_rng'],torch.get_rng_state())
        assert all(torch.equal(a,b) for a,b in zip(expected['cuda_rng'],torch.cuda.get_rng_state_all()))
        rows.append(dict(arm=arm,kind='resume',calls=4,start=start+1,end=counter.count,passed=True))
        start=counter.count;step=r.step;before=copy.deepcopy(r.diagnostics)
        try:r.update(fault='after_optimizer')
        except RuntimeError as e:assert 'injected after_optimizer' in str(e)
        else:raise AssertionError('injection absent')
        assert r.step==step and r.requires_restore and r.diagnostics==before and counter.count==start+1
        try:r.update()
        except RuntimeError:pass
        else:raise AssertionError('failed trainer continued')
        rows.append(dict(arm=arm,kind='failure',calls=1,start=start+1,end=counter.count,passed=True))
        del t,r;torch.cuda.empty_cache()
    return rows


def validate_receipt(r,mode,config,plan,root):
    specs=execution_plan()[mode];ledger='cuda_physical.jsonl' if mode=='CUDA' else 'smoke_physical.jsonl'
    env=read(root/'RUNTIME_ENVIRONMENT.json')
    if (r['status']!='PASS' or r['execution_commit']!=config['execution_commit'] or r['plan_sha256']!=plan['plan_sha256']
        or r['execution_sha256']!=digest(execution_plan()) or r['environment_sha256']!=digest(env['fingerprint'])
        or env['sha256']!=r['environment_sha256'] or r['physical_calls']!=specs['planned_calls']
        or ledger_count(root/ledger)!=r['physical_calls'] or len(r['rows'])!=len(specs['cases'])):
        raise PermissionError('qualification evidence mismatch')
    pos=0
    for row,spec in zip(r['rows'],specs['cases']):
        if any(row.get(k)!=v for k,v in spec.items()) or row['passed'] is not True or row['start']!=pos+1 or row['end']!=pos+spec['calls']:
            raise PermissionError('case coverage/count mismatch')
        pos=row['end']


def qualify(config,mode):
    if mode not in ('CUDA','smoke'):raise ValueError('finite mode required')
    plan,production=preflight(config);choose_gpu()
    import torch
    from ..single_teacher_scd_v0_1.engine import precision
    from ..five_frameworks_v1.native_runner import Counter
    precision();device=torch.device('cuda:0');torch.cuda.set_device(device)
    name='CUDA_QUALIFICATION' if mode=='CUDA' else 'SMOKE'
    with owned_root(config,plan) as root:
        env=bind_environment(root,environment());path=root/(name+'.json')
        ledger=root/('cuda_physical.jsonl' if mode=='CUDA' else 'smoke_physical.jsonl')
        if path.exists():
            validate_receipt(read(path),mode,config,plan,root);session_totals(root/'costs'/name)
            return read(path)
        if list((root/'costs'/name).glob('*/session.json')):
            raise RuntimeError('previous qualification attempt without receipt; no automatic retry')
        if ledger_count(ledger):raise RuntimeError('partial qualification; no automatic retry')
        if mode=='smoke':
            validate_receipt(read(root/'CUDA_QUALIFICATION.json'),'CUDA',config,plan,root)
            session_totals(root/'costs'/'CUDA_QUALIFICATION')
        counter=Counter(ledger,execution_plan()[mode]['cap'])
        permit=synthetic_capability('synthetic',production) if mode=='CUDA' else Capability({**production.bindings,'execution_scope':'smoke'},production.budget,_SEAL)
        try:
            # Prefix acceptance uses separate cost sessions, before smoke instrumentation.
            prefixes={}
            if mode=='smoke':
                for n in plan['nodes']:
                    if n['seed']==163 and n['order']==1:prefixes[n['arm']]=(n,*accept_prefix(n,config,plan,permit,root,device))
            with cost_session(root/'costs'/name,name):
                if mode=='CUDA':rows=cuda_cases(config,plan,permit,root,device,counter)
                else:
                    rows=[]
                    for arm in ('C0','C1','C2','C3'):
                        n,old,receipt=prefixes[arm];t=construct(n,config,plan,permit,old,receipt,device,smoke=True)
                        counter.wrap(t.optimizer);start=counter.count
                        for _ in range(8):t.update()
                        assert t.provider.u_reads==0 and not t.diagnostics
                        rows.append(dict(arm=arm,calls=8,start=start+1,end=counter.count,passed=True,L_only=True,discarded=True))
                        del t;torch.cuda.empty_cache()
            result=dict(status='PASS',execution_commit=config['execution_commit'],plan_sha256=plan['plan_sha256'],
                execution_sha256=digest(execution_plan()),environment_sha256=env,physical_calls=counter.count,rows=rows)
            write(path,result);validate_receipt(result,mode,config,plan,root);return result
        except BaseException as e:
            write(path,dict(status='ENGINEERING_STOP',error=repr(e),physical_calls=counter.count));raise

"""Generated native qualification and L-only smoke, behind new dose authority."""
import copy,random
from pathlib import Path
import torch
from .protocol import read,write,digest,canonical_plan,DOSES,STUDY
from .authority import preflight,synthetic_capability,Capability,_SEAL,execution_plan,baseline_environment
from .execution import owned_root,choose_gpu,ledger_count,accept_prefix,construct,identity
from ..f5_confirmation_v1.assurance import bind_environment,environment,cost_session,session_totals
from ..native_key_alignment_v0_1.qualification import equal
from .trainer import DoseTrainer


def generated_trainer(reference,arm,device,permit,weight,size=384,meter=True):
    from ..native_key_alignment_v0_1.tests import Generated
    from ..five_frameworks_v1.native_parent import build,NativeLRParent
    from ..five_frameworks_v1.model import Model
    from ..f5_confirmation_v1.native_qualification import foreground_fixture
    class DeviceGenerated(Generated):
        def labeled(self,*a,**kw):
            x,y,ids=super().labeled(*a,**kw);return x.to(device),y.to(device),ids
        def unlabeled(self,*a,**kw):
            x,g,ids=super().unlabeled(*a,**kw);return x.to(device),g.to(device),ids
    provider=DeviceGenerated(seed=163,order=1,stage=2,size=size,stage_source={'kind':'generated','domain':'RIM_ONE_r3','prefix_binding_sha256':'generated_only'})
    native=build(reference,device,163);foreground_fixture(native)
    options={**canonical_plan()['options']['Drishti_GS'],'total_steps':5,'PAS_confidence':0.,'PAS_cosine':-1.}
    return DoseTrainer(Model(NativeLRParent(native,163,provider.stage_source),'B2_PARENT_PAS_KL').to(device),provider,options,arm,permit,weight,meter=meter)


def without_time(v):
    if isinstance(v,dict):return {k:without_time(x) for k,x in v.items() if not (isinstance(k,str) and k.endswith('seconds'))}
    if isinstance(v,list):return [without_time(x) for x in v]
    return v


def snapshot(t):
    from ..native_key_alignment_v0_1.qualification import snapshot as old_snapshot
    return without_time(old_snapshot(t))


def common(t):
    v=snapshot(t)
    for k in ('diagnostics','alignment_cost','support_summary'):v.pop(k)
    return v


def exact(a,b):
    if isinstance(a,torch.Tensor):
        if not torch.equal(a,b):raise AssertionError('meter changed tensor')
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b


def low_equivalence(reference,arm,device,permit,counter,size):
    from ..native_key_alignment_v0_1 import authority as old_authority
    from ..native_key_alignment_v0_1.qualification import generated_trainer as old_make
    # A generated-only old capability is NOT a production approval or payload permit.
    old_permit=old_authority.synthetic_capability() if device.type=='cpu' else old_authority.Capability(
        {'study_id':'NATIVE_KEY_ALIGNMENT_V0_1','execution_scope':'synthetic','purpose':'generated low-dose comparison only'},
        {'generated_optimizer_calls':2},old_authority._SEAL)
    snaps=[]
    for kind in ('old','on','off'):
        torch.manual_seed(311);random.seed(311)
        t=old_make(reference,arm,device,old_permit,size=size) if kind=='old' else generated_trainer(reference,arm,device,permit,.05,size,meter=kind=='on')
        counter.wrap(t.optimizer)
        for _ in range(2):t.update()
        snaps.append(common(t));del t
    exact(snaps[0],snaps[1]);exact(snaps[1],snaps[2])
    return dict(arm=arm,old_new_low_dose_exact=True,meter_on_off_exact=True,reads_RNG_state_exact=True)


def generated_identity(t):
    return dict(study_id=STUDY,arm=t.arm,dose_id=t.dose_id,lambda_align=t.align_weight,family='B2_PARENT_PAS_KL',seed=163,order=1,stage=2,node_id='GENERATED_'+t.arm+'_'+t.dose_id)


def continuation(reference,arm,weight,device,permit,counter,root,size,failure=True):
    from .state import save,resume
    from unittest.mock import patch
    t=generated_trainer(reference,arm,device,permit,weight,size);counter.wrap(t.optimizer);start=counter.count;t.update()
    assert t.provider.u_reads==0
    rows=[dict(arm=arm,dose_id=t.dose_id,lambda_align=weight,kind='warmup',calls=1,start=start+1,end=counter.count,passed=True)]
    path=Path(root)/(t.arm+t.dose_id+'.pt');save(t,path,generated_identity(t));start=counter.count
    for _ in range(2):t.update()
    assert t.last['alignment']['valid_classes']>0 and t.last['alignment']['weighted_loss']>0
    expected=snapshot(t)
    p=type(t.provider)(seed=163,order=1,stage=2,size=size,stage_source=t.provider.stage_source)
    with patch('experiments.lcrseg.nka_dose_v0_1.trainer.coordinate_basis',side_effect=AssertionError('resampling forbidden')):
        r=resume(path,reference,device,p,t.options,generated_identity(t),permit,arm,alignment_weight=weight)
    counter.wrap(r.optimizer)
    for _ in range(2):r.update()
    equal(expected,snapshot(r));assert type(r) is DoseTrainer
    rows.append(dict(arm=arm,dose_id=t.dose_id,lambda_align=weight,kind='resume',calls=4,start=start+1,end=counter.count,passed=True))
    if failure:
        start=counter.count;step=r.step;before=copy.deepcopy(r.diagnostics)
        try:r.update(fault='after_optimizer')
        except RuntimeError as e:assert 'injected after_optimizer' in str(e)
        else:raise AssertionError('no failure')
        assert r.step==step and r.requires_restore and r.diagnostics==before and counter.count==start+1
        try:r.update()
        except RuntimeError:pass
        else:raise AssertionError('continued failed trainer')
        rows.append(dict(arm=arm,dose_id=t.dose_id,lambda_align=weight,kind='failure',calls=1,start=start+1,end=counter.count,passed=True))
    return rows,path,t


def oracle(device,counter):
    from .meter import candidate,verify_prediction
    evidence=[]
    for i,(lr,foreach) in enumerate([(.0005,None),(0.,False),(1e-12,True),(.001,None)]):
        p=[torch.nn.Parameter(torch.linspace(-.4,.7,9,device=device).reshape(3,3)) for _ in range(4)]
        opt=torch.optim.Adam([{'params':p[:2],'lr':lr,'name':'A'},{'params':p[2:],'lr':lr*.5,'name':'B'}],weight_decay=4e-5,foreach=foreach)
        for j,x in enumerate(p):
            if j%2:opt.state[x]=dict(step=torch.tensor(float(j*7)),exp_avg=torch.full_like(x,.01),exp_avg_sq=torch.full_like(x,.02))
            x.grad=None if j==0 or (i==2 and j==3) else (torch.zeros_like(x) if j==1 else torch.full_like(x,.03*(j+1)))
        named={str(j):x for j,x in enumerate(p)};names={id(x):n for n,x in named.items()};before={n:x.detach().clone() for n,x in named.items()}
        old_state=copy.deepcopy(opt.state_dict());old_grads=[None if x.grad is None else x.grad.clone() for x in p]
        expected,st,cost=candidate(opt,{id(x):x.grad for x in p},names)
        exact(before,{n:x.detach() for n,x in named.items()});exact(old_state,opt.state_dict());exact(old_grads,[x.grad for x in p])
        counter.wrap(opt);opt.step();check=verify_prediction((before,expected,st,named),opt)
        assert not opt.state.get(p[0],{}) and torch.equal(p[0],before['0'])
        if i==2:exact(old_state['state'][3],opt.state_dict()['state'][3]);assert torch.equal(p[3],before['3'])
        evidence.append(dict(case=i,lr=lr,foreach=foreach,prediction=check,cost=cost,None_is_not_zero=True))
    return evidence


def validate_receipt(r,mode,config,plan,root):
    specs=execution_plan()[mode];ledger='cuda_physical.jsonl' if mode=='CUDA' else 'smoke_physical.jsonl';env=read(root/'RUNTIME_ENVIRONMENT.json')
    baseline_environment(env['fingerprint'])
    if (r['status']!='PASS' or r['execution_commit']!=config['execution_commit'] or r['plan_sha256']!=plan['plan_sha256']
        or r['execution_sha256']!=digest(execution_plan()) or r['environment_sha256']!=digest(env['fingerprint'])
        or env['sha256']!=r['environment_sha256'] or r['physical_calls']!=specs['planned_calls']
        or ledger_count(root/ledger)!=r['physical_calls'] or len(r['rows'])!=len(specs['cases'])):raise PermissionError('qualification evidence mismatch')
    pos=0
    for row,spec in zip(r['rows'],specs['cases']):
        if any(row.get(k)!=v for k,v in spec.items()) or row['passed'] is not True or row['start']!=pos+1 or row['end']!=pos+spec['calls']:raise PermissionError('case coverage/count mismatch')
        pos=row['end']


def qualify(config,mode):
    if mode not in ('CUDA','smoke'):raise ValueError('finite mode required')
    plan,production=preflight(config);choose_gpu()
    from ..single_teacher_scd_v0_1.engine import precision
    from ..five_frameworks_v1.native_runner import Counter
    precision();device=torch.device('cuda:0');torch.cuda.set_device(device)
    name='CUDA_QUALIFICATION' if mode=='CUDA' else 'SMOKE'
    with owned_root(config,plan) as root:
        env=bind_environment(root,baseline_environment(environment()));path=root/(name+'.json')
        ledger=root/('cuda_physical.jsonl' if mode=='CUDA' else 'smoke_physical.jsonl')
        if path.exists():validate_receipt(read(path),mode,config,plan,root);session_totals(root/'costs'/name);return read(path)
        if ledger_count(ledger) or list((root/'costs'/name).glob('*/session.json')):raise RuntimeError('partial qualification; no automatic retry')
        if mode=='smoke':validate_receipt(read(root/'CUDA_QUALIFICATION.json'),'CUDA',config,plan,root);session_totals(root/'costs'/'CUDA_QUALIFICATION')
        counter=Counter(ledger,execution_plan()[mode]['cap'])
        permit=synthetic_capability('synthetic',production) if mode=='CUDA' else Capability({**production.bindings,'execution_scope':'smoke'},production.budget,_SEAL)
        try:
            prefixes={}
            if mode=='smoke':
                for n in plan['nodes']:
                    if n['seed']==163 and n['order']==1:prefixes[n['arm'],n['dose_id']]=(n,*accept_prefix(n,config,plan,permit,root,device))
            with cost_session(root/'costs'/name,name):
                rows=[];evidence={}
                if mode=='CUDA':
                    for arm in ('C2','C3'):
                        for dose,w in DOSES.items():
                            out,_,t=continuation(config['reference'],arm,w,device,permit,counter,root,384);rows+=out;del t;torch.cuda.empty_cache()
                    for arm in ('C2','C3'):
                        start=counter.count;evidence[arm]=low_equivalence(config['reference'],arm,device,permit,counter,384)
                        rows.append(dict(arm=arm,kind='low_dose_equivalence',calls=6,start=start+1,end=counter.count,passed=True))
                    start=counter.count;evidence['oracle']=oracle(device,counter)
                    rows += [dict(kind='adam_oracle',case=i,calls=1,start=start+i+1,end=start+i+1,passed=True) for i in range(4)]
                else:
                    for (arm,dose),(n,old,receipt) in prefixes.items():
                        t=construct(n,config,plan,permit,old,receipt,device,smoke=True);counter.wrap(t.optimizer);start=counter.count
                        for _ in range(8):t.update()
                        assert t.provider.u_reads==0 and not t.diagnostics
                        rows.append(dict(arm=arm,dose_id=dose,lambda_align=n['lambda_align'],calls=8,start=start+1,end=counter.count,passed=True,L_only=True,discarded=True));del t;torch.cuda.empty_cache()
            result=dict(status='PASS',execution_commit=config['execution_commit'],plan_sha256=plan['plan_sha256'],execution_sha256=digest(execution_plan()),environment_sha256=env,physical_calls=counter.count,rows=rows,evidence=evidence)
            write(path,result);validate_receipt(result,mode,config,plan,root);return result
        except BaseException as e:write(path,dict(status='ENGINEERING_STOP',error=repr(e),physical_calls=counter.count));raise

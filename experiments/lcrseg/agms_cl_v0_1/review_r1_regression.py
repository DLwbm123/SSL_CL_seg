"""At most two isolated zero-optimizer R1 regressions. No production capability.

Run with an external evidence directory. Real P0 orchestration and counter classes
are exercised with generated payloads/providers and stub model/environment/access.
This is not a native model, prefix acceptance, P0, or CUDA qualification.
"""
import contextlib
import copy
import functools
import json
import random
import platform
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from . import p0, authority, reporting
from .protocol import ROOT,DOC,read,write,digest,canonical_plan,code_manifest
from .revalidation import CHECKS,BASE_COMMIT,binding,validate_composite
from ..f5_confirmation_v1.assurance import cost_session,session_totals
from ..ssl_anchored_mix_v0_1 import telemetry
from ..five_frameworks_v1.semantics import tensor_fingerprint


def refuse(call,kind=(PermissionError,ValueError,RuntimeError,FileNotFoundError)):
    try:call()
    except kind as error:return str(error)
    raise AssertionError('expected refusal')


class GeneratedModel(torch.nn.Module):
    def __init__(self,*args):super().__init__()
    def forward(self,x,**kw):
        return torch.tensor([.04,.48,.48]).log().reshape(1,3,1,1).expand(len(x),3,*x.shape[-2:])


class CPUProxy:
    # Only the tested P0 module sees this proxy; never mutate torch.device globally.
    cuda=SimpleNamespace(set_device=lambda device:None)
    def device(self,*args):return torch.device('cpu')
    def __getattr__(self,name):return getattr(torch,name)


def run(evidence):
    evidence=Path(evidence).resolve()
    if evidence==ROOT or ROOT in evidence.parents:raise ValueError('generated evidence must be outside checkout')
    evidence.mkdir(parents=True,exist_ok=True)
    public=DOC/'REVIEW_R1_REGRESSION'
    attempts=read(evidence/'ATTEMPTS.json') if (evidence/'ATTEMPTS.json').exists() else []
    published=read(public/'ATTEMPTS.json') if (public/'ATTEMPTS.json').exists() else []
    if attempts[:len(published)]!=published:raise ValueError('regression history cannot reset')
    if len(attempts)>=2 or any(x['status']=='STARTED' for x in attempts):raise RuntimeError('R1 regression attempt limit')
    attempt=len(attempts)+1
    row=dict(attempt=attempt,status='STARTED',planned_optimizer_calls=0)
    attempts.append(row);write(evidence/'ATTEMPTS.json',attempts)
    work=evidence/f'attempt_{attempt}';work.mkdir()
    plan=canonical_plan();tree=code_manifest()['code_tree_sha256'];checks={};optimizer_attempts=[]
    def forbidden_step(*a,**k):
        optimizer_attempts.append(True);raise AssertionError('optimizer.step forbidden')
    def checked(name):checks[name]='PASS'
    results={}
    report=dict(status='FAIL',attempt=attempt,code_tree_sha256=tree,checks=checks,
                optimizer_calls=0,optimizer_attempts=0,real_prefix_reads=0,patient_reads=0,
                native_model_runs=0,CUDA_runs=0,production_permits_created=0,
                scope='generated CPU orchestration/report checks; real counters; stub native model/provider/environment',
                environment=dict(python=platform.python_version(),torch=torch.__version__,device='cpu'))
    try:
        with contextlib.ExitStack() as stack:
            for cls in (torch.optim.Adam,torch.optim.SGD):stack.enter_context(patch.object(cls,'step',forbidden_step))
            # Sentinel state is never stepped or forwarded. No oracle invokes an optimizer.
            sentinel=torch.nn.Linear(2,1);sentinel.weight.grad=torch.ones_like(sentinel.weight)
            risk=torch.tensor([.25,.25,.25],requires_grad=True)
            optimizer=torch.optim.Adam(sentinel.parameters())
            model_before=copy.deepcopy(sentinel.state_dict());grad_before=sentinel.weight.grad.clone()
            risk_before=risk.detach().clone();opt_before=copy.deepcopy(optimizer.state_dict())
            rng=torch.get_rng_state().clone();py_rng=random.getstate();np_rng=np.random.get_state()
            original_step=torch.optim.Adam.step
            inner=[]
            def nested():
                with cost_session(work/'old'/'P0','P0_read_only',cuda=False):
                    with cost_session(work/'old'/'PREFIX','prefix_integrity',cuda=False):inner.append(True)
            assert 'nested counter scope' in refuse(nested)
            assert not inner and telemetry.ACTIVE is None and torch.optim.Adam.step is original_step
            assert all(session_totals(work/'old'/n)['failed_sessions']==1 for n in ('P0','PREFIX'))
            checked('old_nested_failure_and_cleanup')
            # All payload files below are generated in this regression, never real prefixes.
            fixtures={}
            for order in (1,2):
                folder=work/'generated'/str(order);folder.mkdir(parents=True)
                student={'w':torch.tensor([float(order)])};identity={'domain':'generated','order':order}
                receipt=dict(identity=identity,student_hash=tensor_fingerprint(student),node_id=f'generated_{order}',transform_hash='generated')
                torch.save(dict(identity=identity,student=student),folder/'student.pt');fixtures[order]=(folder,receipt)
            events=[];reads=[];failure=[None];tamper=[None]
            def prefix(node,config,plan,permit,root,device):
                order=node['order'];assert telemetry.ACTIVE is None
                events.append(f'prefix{order}_enter')
                with cost_session(root/'costs'/f'PREFIX_{order}','prefix_integrity',cuda=False):
                    if failure[0]==order:raise ValueError('generated prefix rejection')
                events.append(f'prefix{order}_exit')
                path,receipt=fixtures[order];receipt=copy.deepcopy(receipt)
                if tamper[0]=='identity':receipt['identity']={'wrong':True}
                if tamper[0]=='hash':receipt['student_hash']='0'*64
                return path,receipt
            @contextlib.contextmanager
            def counted(root,scope):
                assert telemetry.ACTIVE is None
                assert events[:4]==['prefix1_enter','prefix1_exit','prefix2_enter','prefix2_exit']
                events.append('P0_enter')
                with cost_session(root,scope,cuda=False) as counter:yield counter
                events.append('P0_exit')
            def provider(data,seed,order,stage,sid,device,permit,allow_u):
                assert allow_u is False and telemetry.ACTIVE is not None
                reads.append(order)
                item={'image':torch.zeros(3,4,4),'geometry':torch.ones(4,4,dtype=torch.bool),'label':torch.ones(4,4,dtype=torch.long)}
                return SimpleNamespace(_l=[item for _ in range(10 if order==1 else 16)])
            @contextlib.contextmanager
            def owned(config,plan):
                root=Path(config['run_root']);root.mkdir(exist_ok=True);yield root
            def execute(root):
                config=dict(run_root=str(root),execution_commit='generated',reference='generated',data='generated')
                with contextlib.ExitStack() as mock:
                    replacements=dict(preflight=lambda c:(plan,None),choose_gpu=lambda:None,torch=CPUProxy(),
                        owned_root=owned,environment=lambda:{'generated':True},baseline_environment=lambda e:e,
                        accept_prefix=prefix,cost_session=counted,build=lambda *a:SimpleNamespace(load_state_dict=lambda s:None),
                        NativeLRParent=lambda *a,**k:None,Model=GeneratedModel,NativeCurrentDomain=provider)
                    for k,v in replacements.items():mock.enter_context(patch.object(p0,k,v))
                    mock.enter_context(patch('experiments.lcrseg.single_teacher_scd_v0_1.engine.precision',lambda:None))
                    return p0.run(config)
            root=work/'success';result=execute(root)
            assert result['images']==26 and reads==[1,2] and events[-1]=='P0_exit'
            assert telemetry.ACTIVE is None
            assert all(session_totals(root/'costs'/n)['failed_sessions']==0 for n in ('PREFIX_1','PREFIX_2','P0'))
            results['success_events']=list(events);checked('prefixes_exit_before_P0')
            previous=(list(events),list(reads),list(root.rglob('session.json')))
            assert execute(root)==result and previous==(events,reads,list(root.rglob('session.json')))
            checked('sealed_idempotency')
            events.clear();reads.clear();failure[0]=2
            root=work/'failed_prefix';assert 'generated prefix rejection' in refuse(lambda:execute(root))
            assert reads==[] and not (root/'P0_REPORT.json').exists() and not (root/'costs'/'P0').exists()
            assert session_totals(root/'costs'/'PREFIX_2')['failed_sessions']==1
            results['failed_prefix_events']=list(events);failure[0]=None;checked('prefix_failure_no_L_or_P0')
            events.clear();root=work/'partial';write(root/'costs/P0/0001/session.json',{'status':'FAILED'})
            assert 'partial P0' in refuse(lambda:execute(root)) and not events and not reads
            checked('partial_P0_refused')
            for mode in ('identity','hash'):
                events.clear();reads.clear();tamper[0]=mode;root=work/f'bad_{mode}'
                assert 'prefix changed' in refuse(lambda:execute(root)) and reads==[] and not (root/'P0_REPORT.json').exists()
                assert session_totals(root/'costs'/'P0')['failed_sessions']==1
            tamper[0]=None;checked('payload_identity_hash_rechecked')
            from .core import coverage
            coverage_rows=[]
            for g,f,c in ((4,0,0),(4,4,0),(4,0,4),(3,1,1),(0,0,0)):
                geom=(torch.arange(4)<g).reshape(1,2,2);fine=(torch.arange(4)<f).reshape(1,2,2)
                coarse=((torch.arange(4)>=f)&(torch.arange(4)<f+c)).reshape(1,2,2)
                stats={'disc':torch.ones(1,1,2,2),'variance':torch.zeros(1,2,2)}
                raw=coverage(fine,coarse,geom,stats)[0];saved=copy.deepcopy(raw);out=reporting.normalize_coverage(raw)
                assert raw==saved and out['ignore']==4-g and out['total_pixels']==4
                assert out['fine']+out['coarse']+out['unselected_valid_pixels']==g
                assert out['geometry']+out['invalid_geometry_pixels']==4
                assert out['invalid_geometry_fraction']==(4-g)/4
                assert out['unselected_valid_fraction']==((g-f-c)/g if g else None)
                if not g:assert out['fine_fraction'] is None and out['coarse_fraction'] is None
                coverage_rows.append(raw)
            refuse(lambda:reporting.normalize_coverage(dict(geometry=1,fine=1,coarse=1,ignore=0)))
            # Exercise the actual export path without training, old suite, or invented scientific results.
            rows=[]
            for arm in ('A0','A1','A2','A3','A4','A5'):
                for order in (1,2):
                    d=dict(coverage=coverage_rows,risk=[.25]*3,alpha=[1/3]*3,shadow_L=[],L_errors=[],uniform_risk_disagreement=0)
                    rows.append(dict(node_id=f'generated_{arm}_{order}',arm=arm,identity={'order':order},
                        origin='historical_import' if arm=='A0' else 'new_execution',support={},
                        scores={n:dict(rim=0,cup=0,disc_union=0,macro_Dice=0) for n in ('REFUGE','RIM_ONE_r3','Drishti_GS')},
                        diagnostics={str(i):copy.deepcopy(d) for i in range(4)}))
            raw_rows=copy.deepcopy(rows)
            with patch.object(reporting,'metrics',lambda r:dict(Final=0,Old=0,Incoming=0,Forget=0)):
                reporting.export(work/'export',dict(rows=rows,pairs=[{'generated':True}]*27,status='GENERATED_FIXTURE',costs={},gates=None))
            assert rows==raw_rows
            exported=read(work/'export/COVERAGE_AND_RISK.json')
            assert exported[2]['points'][0]['coverage'][0]['unselected_valid_pixels']==4
            assert read(work/'export/PUBLIC_RESULTS.json')['rows']==raw_rows
            assert 'unselected_valid_pixels' not in read(work/'export/GRADIENT_DIAGNOSTICS.json')[0]['coverage'][0]
            checked('coverage_partitions_and_export')
            # Exercise real preflight with only Git/clean-tree responses mocked, no valid approval/permit.
            from .execution import accept_prefix
            for permit in (None,SimpleNamespace(study_id='NKA_DOSE'),authority.synthetic_capability()):
                refuse(lambda:accept_prefix(plan['nodes'][0],{},plan,permit,work,torch.device('cpu')),PermissionError)
            for label,review in [('missing',None),('old',{'study_id':'NKA_DOSE_V0_1'}),
                                 ('changes_requested',read(DOC/'REVIEW_R1/REVIEW_DECISION.json'))]:
                cfg=dict(review=str(work/f'{label}_review.json'),launch_confirmation=str(work/f'{label}_launch.json'))
                if review is not None:write(Path(cfg['review']),review);write(Path(cfg['launch_confirmation']),{})
                with patch.object(authority.subprocess,'check_output',side_effect=[BASE_COMMIT+'\n','']), \
                     patch.object(p0,'choose_gpu',side_effect=AssertionError('GPU access before authority')), \
                     patch.object(p0,'accept_prefix',side_effect=AssertionError('prefix before authority')):
                    refuse(lambda:p0.run(cfg),(PermissionError,FileNotFoundError))
            checked('missing_and_old_authority_refused')
            assert all(torch.equal(v,model_before[k]) for k,v in sentinel.state_dict().items())
            assert torch.equal(sentinel.weight.grad,grad_before) and torch.equal(risk,risk_before) and risk.grad is None
            assert optimizer.state_dict()==opt_before and torch.equal(torch.get_rng_state(),rng)
            assert random.getstate()==py_rng
            now=np.random.get_state();assert now[0]==np_rng[0] and np.array_equal(now[1],np_rng[1]) and now[2:]==np_rng[2:]
            checked('readonly_state_and_RNG')
            assert not optimizer_attempts and telemetry.ACTIVE is None and torch.optim.Adam.step is original_step
            assert all(read(p).get('operation_counts',{}).get('optimizer_steps_attempts',0)==0 for p in work.rglob('session.json'))
            checked('zero_optimizer_and_real_IO')
            # Negative binding checks use in-memory copies; no approval or positive production receipt.
            current=code_manifest();proof=binding(plan,current)
            bad=copy.deepcopy(current);key='experiments/lcrseg/agms_cl_v0_1/core.py';bad['files'][key]='0'*64;bad['code_tree_sha256']=digest(bad['files'])
            refuse(lambda:binding(plan,bad),PermissionError)
            bad_plan=copy.deepcopy(plan);bad_plan['plan_sha256']='0'*64;refuse(lambda:binding(bad_plan),PermissionError)
            checked('composite_binding_rejects_drift')
            report.update(status='PASS',checks=checks,details=results)
            assert set(checks)==set(CHECKS)
            composite=dict(proof,status='PASS',regression_sha256=digest(report))
            validate_composite(plan,composite,report,current)
            bad_report=copy.deepcopy(report);bad_report['code_tree_sha256']='0'*64
            refuse(lambda:validate_composite(plan,composite,bad_report,current),PermissionError)
            bad_proof=copy.deepcopy(composite);bad_proof['protected_files']={}
            refuse(lambda:validate_composite(plan,bad_proof,report,current),PermissionError)
            write(work/'COMPOSITE_REPORT.json',composite)
    except BaseException as error:
        report.update(status='FAIL',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['optimizer_attempts']=len(optimizer_attempts)
        row.update(status=report['status'],optimizer_calls=0,optimizer_attempts=len(optimizer_attempts))
        write(work/'REPORT.json',report);write(evidence/'ATTEMPTS.json',attempts)
        write(public/'ATTEMPTS.json',attempts)
        write(public/f'ATTEMPT_{attempt}_REPORT.json',report);write(public/'REPORT.json',report)
        # Publish bounded aggregate counter evidence, not generated tensor fixtures/private paths.
        costs={str(p.parent.relative_to(work)):read(p) for p in sorted(work.rglob('session.json'))}
        write(public/f'ATTEMPT_{attempt}_COSTS.json',costs)
        if report['status']=='PASS':write(public/'COMPOSITE_REPORT.json',read(work/'COMPOSITE_REPORT.json'))
    return report

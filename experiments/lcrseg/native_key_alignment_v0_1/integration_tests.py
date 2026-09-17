"""CPU_INTEGRATION_R1: separately authorized supplement, never resets D0."""
import copy
import json
import random
import tempfile
import time
import traceback
from pathlib import Path
from unittest.mock import patch
import torch
from .protocol import ROOT,DOC,read,write,digest,canonical_plan,validate_plan,code_manifest,ANCHORS
from .authority import synthetic_capability,authorize,execution_plan
from .qualification import generated_trainer,snapshot,equal
from .state import save,resume,validate_payload,metadata_path
from ..five_frameworks_v1.native_runner import Counter
from ..five_frameworks_v1.train_stage import StageTrainer
from ..five_frameworks_v1.native_operations import NativeOperations

CASES={'canonical_rehash_and_runtime':0,'zero_equivalence':6,'C2_C3_continuation_and_negative_restore':10,
       'failure_ledger':1,'integrity_and_incomplete_cost':0,'reports_and_diagnostics':0}
LEDGER=DOC/'CPU_INTEGRATION_R1'


def refuse(fn):
    try:fn()
    except (ValueError,PermissionError,RuntimeError,KeyError):return
    raise AssertionError('invalid input accepted')


def run_tests(reference):
    import os,sys
    if torch.cuda.is_initialized() or sys.flags.optimize or 'PYTHONOPTIMIZE' in os.environ:raise RuntimeError('CPU unoptimized tests only')
    attempts=read(LEDGER/'ATTEMPTS.json') if (LEDGER/'ATTEMPTS.json').exists() else []
    if len(attempts)>=2:raise RuntimeError('CPU_INTEGRATION_R1 invocation cap')
    assert sum(CASES.values())==17<=24
    # Durable case/call declaration precedes all native construction and updates.
    current=dict(invocation=len(attempts)+1,status='STARTED',cases=CASES,planned_optimizer_calls=17,started=time.time())
    attempts.append(current);write(LEDGER/'ATTEMPTS.json',attempts)
    counter=Counter(LEDGER/'PHYSICAL.jsonl',48);begin=counter.count
    if begin+17>48:raise RuntimeError('supplemental physical budget insufficient')
    torch.set_num_threads(2);device=torch.device('cpu');permit=synthetic_capability();results=[];evidence={}
    diagnostic_vjps=0
    def make(arm):
        t=generated_trainer(reference,arm,device,permit,size=32);return t
    def ident(arm):return dict(study_id='NATIVE_KEY_ALIGNMENT_V0_1',family='B2_PARENT_PAS_KL',arm=arm,seed=163,order=1,stage=2,node_id='GENERATED_'+arm)
    def apply(t):
        counter.wrap(t.optimizer);return t

    def canonical_rehash_and_runtime():
        from .protocol import validate_prefix
        original=canonical_plan();assert original==read(DOC/'D1_MATRIX.json');accepted=[]
        def rehash(p):
            for n in p['nodes']:
                n['options_sha256']=digest(p['options'][n['domain']])
            p['plan_sha256']=digest({k:v for k,v in p.items() if k!='plan_sha256'});return p
        def changes(p,name):
            if name=='lambda_U':p['options']['Drishti_GS']['lambda_U']=.25
            elif name=='swap_domains':
                for n in p['nodes']:
                    n['domain']='RIM_ONE_r3' if n['order']==1 else 'Drishti_GS';n['updates']=p['options'][n['domain']]['total_steps']
            elif name=='move_step':p['nodes'][0]['updates']-=1;p['nodes'][1]['updates']+=1
            elif name=='C3_coordinate_weight':p['arms']['C3'].update(coordinate='fixed_random_key',lambda_align=.2)
            elif name=='cross_seed_prefix':p['nodes'][0]['prefix_node']=p['prefixes'][2]['node_id'];p['nodes'][0]['prefix_binding_sha256']=p['prefixes'][2]['binding_sha256']
            elif name=='PAS':p['options']['Drishti_GS']['PAS_confidence']=.7
            elif name=='ramp':p['options']['Drishti_GS']['U_ramp_fraction']=.1
            elif name=='projection':p['auxiliary']['key_dimension']=16
            elif name=='analysis':p['analysis']['gate']['mean_Final_ge']=0
            elif name=='diagnostic':p['auxiliary']['diagnostics']='optional'
            elif name=='prefix_identity':p['prefixes'][0]['identity']['family']='C3'
            elif name=='id':p['nodes'][0]['id']='OTHER'
        for name in ('lambda_U','swap_domains','move_step','C3_coordinate_weight','cross_seed_prefix','PAS','ramp','projection','analysis','diagnostic','prefix_identity','id'):
            p=copy.deepcopy(original);changes(p,name);refuse(lambda:validate_plan(rehash(p)));accepted.append(name)
        original_read=Path.read_bytes
        for source in ANCHORS:
            def changed(path,source=source):return original_read(path)+(b' ' if path==ROOT/source else b'')
            with patch.object(Path,'read_bytes',changed):refuse(canonical_plan)
        review=read(DOC/'external_review_R1/REVIEW_DECISION.json');refuse(lambda:authorize(review,{},{}))
        t=make('C3');refuse(lambda:__import__('experiments.lcrseg.native_key_alignment_v0_1.execution',fromlist=['construct']).construct(
            original['nodes'][0],{},original,permit,Path('/not-opened'),{},device))
        t.align_basis[0,0]+=.1;refuse(t.semantic_record)
        # Preserve the dedicated narrow prefix check independently of plan validation.
        n=copy.deepcopy(original['nodes'][0]);p=copy.deepcopy(original['prefixes'][0])
        receipt=dict(identity=p['identity'],status='SEALED',student_hash=p['student_sha256'])
        p['receipt_sha256']=digest(receipt);p['binding_sha256']=digest({k:v for k,v in p.items() if k!='binding_sha256'});n['prefix_binding_sha256']=p['binding_sha256']
        validate_prefix(n,p,receipt)
        for key,value in [('seed',164),('phase','D2'),('stage',1),('arm','F5')]:
            refuse(lambda key=key,value=value:validate_prefix({**n,key:value},p,receipt))
        assert len(original['nodes'])==16 and sum(n['updates'] for n in original['nodes'])==42400
        from .authority import validate_runtime
        bad_options=dict(t.options,lambda_U=float('nan'))
        from ..five_frameworks_v1.semantics import resolve_options
        refuse(lambda:resolve_options(bad_options))
        refuse(lambda:validate_runtime(t.model,t.provider,t.options,'C3',permit,None,.2))
        refuse(lambda:validate_runtime(t.model,object(),t.options,'C3',permit,None,.05))
        t.model.family='B0_PARENT_LCTX'
        refuse(lambda:validate_runtime(t.model,t.provider,t.options,'C3',permit,None,.05))
        evidence['canonical']=dict(rehashed_counterexamples_rejected=accepted,changed_anchor_files_rejected=2,
            original_plan_unchanged=True,real_reads=0,production_approvals_created=0)

    def zero_equivalence():
        snapshots=[];rngs=[]
        for arm in ('baseline','C0','C3'):
            torch.manual_seed(211);random.seed(211);t=make('C0' if arm=='baseline' else arm)
            if arm=='baseline':
                p=type(t.provider)(seed=163,order=1,stage=2,size=32,stage_source=t.provider.stage_source)
                t=StageTrainer(t.model,p,t.options,execution=permit)
            elif arm=='C3':t.align_weight=0.
            apply(t)
            for _ in range(2):t.update()
            snapshots.append(copy.deepcopy(dict(model=t.model.state_dict(),ema=t.ema.state_dict(),optimizer=t.optimizer.state_dict(),
                scheduler=t.scheduler.state_dict(),prototypes=t.prototypes.values,support=t.prototypes.support,reads=(t.provider.l_reads,t.provider.u_reads))))
            rngs.append((torch.get_rng_state().clone(),random.getstate()))
        equal(snapshots[0],snapshots[1]);equal(snapshots[0],snapshots[2]);equal(rngs[0],rngs[1]);equal(rngs[0],rngs[2])
        evidence['zero_equivalence']='native B2/C0/C3-zero full state and RNG'

    def C2_C3_continuation_and_negative_restore():
        with tempfile.TemporaryDirectory(prefix='nka_resume_') as tmp:
            for arm in ('C2','C3'):
                t=apply(make(arm));t.update();path=Path(tmp)/(arm+'.pt');save(t,path,ident(arm))
                for _ in range(2):t.update()
                assert t.last['alignment']['valid_classes']>0
                assert t.diagnostics and all('alignment' in d and 'positions' not in d['alignment'] for d in t.diagnostics.values())
                expected=snapshot(t)
                p=type(t.provider)(seed=163,order=1,stage=2,size=32,stage_source=t.provider.stage_source)
                # C2 must restore the saved basis, not draw/QR a replacement.
                with patch('experiments.lcrseg.native_key_alignment_v0_1.trainer.coordinate_basis',side_effect=AssertionError('basis resampled')):
                    r=resume(path,reference,device,p,t.options,ident(arm),permit,arm)
                assert type(r) is type(t);apply(r)
                for _ in range(2):r.update()
                equal(expected,snapshot(r));assert torch.equal(expected['torch_rng'],torch.get_rng_state())
                v=torch.load(path,weights_only=False);meta=read(metadata_path(path))
                for field in ('arm','prefix','basis','lambda','diagnostics'):
                    bad=copy.deepcopy(v)
                    if field=='arm':bad['nka']['arm']='C1'
                    elif field=='prefix':bad['nka']['prefix']={'wrong':True}
                    elif field=='basis':bad['basis'][0,0]+=.1
                    elif field=='lambda':bad['nka']['lambda_align']=.2
                    else:bad['diagnostics']['999']={}
                    refuse(lambda:validate_payload(bad,meta,ident(arm),arm,t.options,p,.05))
                evidence[arm]=dict(continuous_resume='PASS',trainer=type(r).__name__,basis_exact=True,
                    diagnostics_exact=True,nonzero_active_auxiliary=t.last['alignment']['weighted_loss']>0)

    def failure_ledger():
        from .execution import stage_state
        t=apply(make('C1'));t.step=t.cursor=1;before=counter.count
        try:t.update(fault='after_optimizer')
        except RuntimeError as e:assert 'injected after_optimizer' in str(e)
        else:raise AssertionError('missing fault')
        assert counter.count==before+1 and t.step==1 and t.requires_restore and not t.diagnostics
        refuse(lambda:t.update());assert counter.count==before+1
        with tempfile.TemporaryDirectory() as tmp:
            refuse(lambda:save(t,Path(tmp)/'failed.pt',ident('C1')))
            node=canonical_plan()['nodes'][0];nr=Path(tmp)/node['id'];nr.mkdir()
            write(nr/'failure.json',{'status':'ENGINEERING_STOP'});refuse(lambda:stage_state(node,Path(tmp),{'execution_commit':'generated'}))
        evidence['failed_physical_calls']=1

    def integrity_and_incomplete_cost():
        from ..f5_confirmation_v1.review_tests import integrity_faults,cumulative_costs_rng
        integrity_faults();cumulative_costs_rng()
        evidence['integrity']='generated corruption/schema/identity/F/hash/stale-proof and unclosed-cost regressions PASS'

    def reports_and_diagnostics():
        from .reporting import summarize,export
        plan=canonical_plan();rows=[]
        def scores(b):return {d:dict(rim=b,cup=b+.02,disc_union=b+.04,macro_Dice=b+.01) for d in ('REFUGE','RIM_ONE_r3','Drishti_GS')}
        for n in plan['nodes']:
            rows.append(dict(node_id=n['id'],identity=n,scores=scores(.7+(.02 if n['arm']=='C3' else 0)),
                prefix_scores=scores(.8),prefix_timeline={'0':scores(.9)},diagnostics={str(x):dict(norms=dict(supervised=1,KL=1,alignment=.1),extra_vjps=3,alignment={'classes':{},'generated_fixture':True}) for x in execution_plan()['diagnostics'][n['domain']]},
                support={},alignment_cost={}))
        summary=summarize(rows);assert summary['gate']['D1_performance']
        bad=copy.deepcopy(rows);bad[0]['diagnostics'].pop(next(iter(bad[0]['diagnostics'])));refuse(lambda:summarize(bad))
        refuse(lambda:summarize(rows[:-1]))
        with tempfile.TemporaryDirectory() as tmp:export(tmp,dict(status='GENERATED_REPORT_FIXTURE',rows=rows,costs={'optimizer_calls':0},**summary))
        evidence['reports']=dict(rows=16,domain_rows=48,paired_rows=21,common_prefix_identity=True)

    from collections import Counter as Counts
    from .trainer import KeyAlignmentTrainer
    original_diagnostics=KeyAlignmentTrainer.gradient_diagnostics
    def counted_diagnostics(t,*args,**kwargs):
        nonlocal diagnostic_vjps
        before=t.alignment_cost['diagnostic_vjps']
        try:return original_diagnostics(t,*args,**kwargs)
        finally:diagnostic_vjps+=t.alignment_cost['diagnostic_vjps']-before
    counts=Counts();checks=locals()
    with patch.object(KeyAlignmentTrainer,'gradient_diagnostics',counted_diagnostics):
        for name,expected_calls in CASES.items():
            start=counter.count
            try:
                if name=='integrity_and_incomplete_cost':
                    # This shared test owns its cost sessions; never nest recorders.
                    checks[name]();counts['backward']+=3;counts['backward_attempts']+=3
                else:
                    with NativeOperations(LEDGER/'operations'/str(len(attempts))/name) as operations:checks[name]()
                    counts.update(operations.counts)
                assert counter.count-start==expected_calls
                results.append(dict(test=name,status='PASS',optimizer_calls=counter.count-start))
            except Exception:
                if name!='integrity_and_incomplete_cost':counts.update(operations.counts)
                results.append(dict(test=name,status='FAIL',optimizer_calls=counter.count-start,traceback=traceback.format_exc().replace(str(ROOT),'<repo>')))
    current.update(status='PASS' if all(r['status']=='PASS' for r in results) else 'FAIL',results=results,calls=counter.count-begin,
                   cumulative_calls=counter.count,seconds=time.time()-current['started'])
    write(LEDGER/'ATTEMPTS.json',attempts)
    result=dict(status=current['status'],tests=results,evidence=evidence,code_tree_sha256=code_manifest()['code_tree_sha256'],
        science_sha256=canonical_plan()['plan_sha256'],execution_sha256=digest(execution_plan()),
        cost=dict(original_D0_calls=30,original_D0_attempts=2,supplemental_calls=counter.count,supplemental_cap=48,
            supplemental_attempts=len(attempts),attempt_cap=2,combined_calls=30+counter.count,combined_cap=78,
            this_attempt_calls=counter.count-begin,operation_counts=counts,this_attempt_diagnostic_VJPs=diagnostic_vjps,
            seconds=current['seconds']),environment=dict(python=sys.version.split()[0],torch=torch.__version__,device='cpu',optimize=sys.flags.optimize),
        real_tensor_reads=0,patient_payload_reads=0,production_approvals_created=0,CUDA_calls=0,
        pending=['real prefix acceptance','native CUDA','real smoke','formal D1'])
    write(LEDGER/('ATTEMPT_'+str(len(attempts))+'_REPORT.json'),result);write(LEDGER/'TEST_REPORT.json',result)
    if result['status']!='PASS':raise RuntimeError('integration test failure; retain evidence; no automatic retry')
    return result

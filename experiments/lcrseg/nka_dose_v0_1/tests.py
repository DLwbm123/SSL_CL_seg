"""New bounded CPU generated qualification: 29 calls per declared attempt."""
import copy,os,sys,time,traceback,csv
from collections import Counter as Counts
from pathlib import Path
from unittest.mock import patch
import torch
from .protocol import ROOT,DOC,STUDY,read,write,digest,canonical_plan,validate_plan,code_manifest,ANCHORS
from .authority import synthetic_capability,execution_plan,authorize,baseline_environment,validate_runtime
from .qualification import generated_trainer,low_equivalence,continuation,oracle,generated_identity
from ..five_frameworks_v1.native_runner import Counter
from ..five_frameworks_v1.native_operations import NativeOperations

CASES={'canonical_binding':0,'adam_oracle':4,'low_dose_equivalence':12,'continuation_restore':10,
       'other_cells_gradients':2,'failure_and_tail':1,'effective_weights':0,
       'integrity_and_costs':0,'report_schema':0}


def refuse(fn):
    try:fn()
    except (ValueError,PermissionError,RuntimeError,KeyError):return
    raise AssertionError('invalid request accepted')


def run_tests(reference,evidence_root):
    if sys.flags.optimize or 'PYTHONOPTIMIZE' in os.environ or torch.cuda.is_initialized():raise RuntimeError('unoptimized CPU only')
    root=Path(evidence_root).resolve();root.mkdir(parents=True,exist_ok=True)
    if ROOT==root or ROOT in root.parents:raise RuntimeError('generated payloads must be outside checkout')
    attempts=read(root/'ATTEMPTS.json') if (root/'ATTEMPTS.json').exists() else []
    published=read(DOC/'CPU/ATTEMPTS.json') if (DOC/'CPU/ATTEMPTS.json').exists() else []
    if len(attempts)<len(published) or attempts[:len(published)]!=published:raise RuntimeError('cannot reset published CPU history')
    if len(attempts)>=3 or any(a['status']=='STARTED' for a in attempts):raise RuntimeError('CPU attempt cap or interrupted attempt')
    counter=Counter(root/'PHYSICAL.jsonl',96);begin=counter.count
    if begin!=sum(a['calls'] for a in attempts) or begin+sum(CASES.values())>96:raise RuntimeError('CPU ledger mismatch/cap')
    assert sum(CASES.values())==29<=32
    current=dict(invocation=len(attempts)+1,status='STARTED',cases=CASES,planned_calls=29,started=time.time())
    attempts.append(current);write(root/'ATTEMPTS.json',attempts);write(DOC/'CPU/ATTEMPTS.json',attempts)
    torch.set_num_threads(2);device=torch.device('cpu');permit=synthetic_capability();evidence={};results=[];counts=Counts();meter_cost=Counts()
    def make(a,w):return generated_trainer(reference,a,device,permit,w,32)

    def canonical_binding():
        p=canonical_plan();assert p==read(DOC/'PLAN.json') and len(p['nodes'])==16 and len(p['imports'])==12 and sum(n['updates'] for n in p['nodes'])==42400
        mutations={'dose':lambda x:x['nodes'][0].update(lambda_align=.05),'C0':lambda x:x['nodes'][0].update(arm='C0'),
            'C1':lambda x:x['nodes'][0].update(arm='C1'),'seed':lambda x:x['nodes'][0].update(seed=162),
            'stage1':lambda x:x['nodes'][0].update(stage=1),'updates':lambda x:x['nodes'][0].update(updates=2099),
            'prefix':lambda x:x['nodes'][0].update(prefix_binding_sha256='wrong'),
            'options':lambda x:x['options']['Drishti_GS'].update(lambda_U=.25),
            'import':lambda x:x['imports'][0].update(origin='new_execution'),
            'environment':lambda x:x['environment']['fingerprint'].update(python='other'),
            'analysis':lambda x:x['analysis'].update(performance_final_min=0),
            'domain':lambda x:x['nodes'][0].update(domain='RIM_ONE_r3')}
        for change in mutations.values():
            bad=copy.deepcopy(p);change(bad);bad['plan_sha256']=digest({k:v for k,v in bad.items() if k!='plan_sha256'});refuse(lambda:validate_plan(bad))
        original=Path.read_bytes
        for name in ANCHORS:
            with patch.object(Path,'read_bytes',lambda path,name=name:original(path)+(b' ' if path==DOC/'inputs'/name else b'')):refuse(canonical_plan)
        refuse(lambda:baseline_environment({}));refuse(lambda:authorize({}, {}, {}))
        from ..native_key_alignment_v0_1.authority import synthetic_capability as old_capability
        t=make('C3',.5)
        refuse(lambda:validate_runtime(t.model,t.provider,t.options,'C3',old_capability(),None,.5))
        refuse(lambda:validate_runtime(t.model,t.provider,t.options,'C3',permit,None,1.))
        from .execution import construct,accept_prefix
        for fn in (construct,):refuse(lambda:fn(p['nodes'][0],{},p,permit,Path('/unopened'),{},device))
        refuse(lambda:accept_prefix(p['nodes'][0],{},p,permit,root,device))
        t.align_basis[0,0]+=.1;refuse(t.semantic_record)
        from .qualification import common,exact
        left=make('C2',.5);left_state=common(left);del left
        right=make('C2',2.);exact(left_state,common(right));del right
        evidence['canonical']=dict(dose_independent_initial_state_RNG_basis=True,rehashed_negative_cases=list(mutations),changed_input_anchors=3,old_capability_rejected=True,real_prefix_refused_before_IO=True)

    def low_dose_equivalence():
        evidence['low_dose']=[low_equivalence(reference,a,device,permit,counter,32) for a in ('C2','C3')]

    def continuation_restore():
        from .state import validate_payload,metadata_path
        evidence['continuation']=[]
        for arm,w in [('C2',.5),('C3',2.)]:
            rows,path,t=continuation(reference,arm,w,device,permit,counter,root,32,failure=False)
            value=torch.load(path,map_location='cpu',weights_only=False);meta=read(metadata_path(path));ident=generated_identity(t)
            for field in ('dose','prefix','basis','lambda','diagnostics','meter'):
                bad=copy.deepcopy(value)
                if field=='dose':bad['nka']['dose_id']='WRONG'
                elif field=='prefix':bad['nka']['prefix']={}
                elif field=='basis':bad['basis'][0,0]+=.1
                elif field=='lambda':bad['nka']['lambda_align']=.05
                elif field=='meter':bad['nka']['meter_version']='other'
                else:bad['diagnostics']['999']={}
                refuse(lambda:validate_payload(bad,meta,ident,arm,t.options,t.provider,w))
            evidence['continuation'].append(dict(rows=rows,negative_restores=6,diagnostic_steps=list(t.diagnostics)))
            del t

    def other_cells_gradients():
        evidence['active_cells']=[]
        for a,w in [('C2',2.),('C3',.5)]:
            t=make(a,w);t.step=t.cursor=1;counter.wrap(t.optimizer);t.update()
            d=t.diagnostics['2'];assert d['gradients']['upstream_AB']['norms']['alignment']>0 and d['prediction']['passed']
            evidence['active_cells'].append(dict(arm=a,lambda_align=w,alignment_gradient=d['gradients']['upstream_AB']['norms']['alignment'],prediction=d['prediction']))
            del t

    def failure_and_tail():
        from .state import save
        from .execution import stage_state,identity
        t=make('C2',.5);t.step=t.cursor=1;counter.wrap(t.optimizer);before=counter.count
        try:t.update(fault='after_optimizer')
        except RuntimeError as e:assert 'injected after_optimizer' in str(e)
        else:raise AssertionError('expected injected failure')
        assert counter.count==before+1 and t.step==1 and t.requires_restore and not t.diagnostics
        refuse(t.update);refuse(lambda:save(t,root/'failed.pt',generated_identity(t)))
        n=canonical_plan()['nodes'][0];case=root/'generated_tail';nr=case/n['id'];nr.mkdir(parents=True,exist_ok=True)
        # Generated ledger metadata, not an optimizer call or production node.
        (nr/'physical.jsonl').write_text('{"invocation":1}\n');refuse(lambda:stage_state(n,case,{'execution_commit':'generated'}))
        write(nr/'receipt.json',dict(identity=identity(n,{'execution_commit':'generated'}),status='SEALED',step=n['updates'],physical_optimizer_calls=n['updates']))
        refuse(lambda:stage_state(n,case,{'execution_commit':'generated'}))
        write(nr/'failure.json',dict(status='ENGINEERING_STOP'));refuse(lambda:stage_state(n,case,{}))
        evidence['failure']=dict(physical=1,committed=0,second_call_refused=True,tail_and_false_seal_refused=True)

    def adam_oracle():evidence['adam_oracle']=oracle(device,counter)

    def effective_weights():
        from .meter import effective,ratio,validate_adam
        a=torch.arange(12,dtype=torch.float32).reshape(3,4)/10;b=torch.arange(6,dtype=torch.float32).reshape(2,3)/10
        v=torch.eye(4)[:,:1];base=torch.ones(2,4);da=torch.ones_like(a)*.2;db=torch.ones_like(b)*.3
        projection=lambda z:z-(z@v)@v.T
        delta=effective(base,a+da,b+db,v)-effective(base,a,b,v)
        linear=db@projection(a)+b@projection(da);cross=db@projection(da)
        assert cross.norm()>0 and torch.allclose(delta,linear+cross,atol=1e-6) and not torch.allclose(delta,linear)
        assert ratio([base],[base],[base+1])['rho'] is None and ratio([base],[base],[base+1])['reason']
        opt=torch.optim.Adam([torch.nn.Parameter(base)],weight_decay=4e-5,amsgrad=True);refuse(lambda:validate_adam(opt))
        evidence['effective_W']=dict(quadratic_cross_term=True,zero_denominator_null=True,unsupported_flags_rejected=True)

    def integrity_and_costs():
        from ..f5_confirmation_v1.review_tests import integrity_faults,cumulative_costs_rng
        integrity_faults();cumulative_costs_rng()
        evidence['integrity']='generated corruption/schema/identity/F/hash/stale proof/unclosed session checks PASS; zero optimizer calls'

    def report_schema():
        from .reporting import historical,summarize,export
        from .execution import identity
        plan=canonical_plan();rows=historical(plan)
        for n in plan['nodes']:
            src=next(r for r in rows if r['identity']['arm']==n['arm'] and r['identity']['seed']==n['seed'] and r['identity']['order']==n['order'] and r['origin']=='historical_import')
            row=copy.deepcopy(src);row.update(node_id=n['id'],identity=identity(n,{'execution_commit':'GENERATED'}),origin='new_execution',lambda_align=n['lambda_align'])
            row.pop('adam_diagnostics')
            row['diagnostics']={str(x):dict(extra_vjps=4,read_only_candidates=2,prediction={'passed':True},gradients={},alignment={},losses={},rho_theta={},rho_W={}) for x in execution_plan()['diagnostics'][n['domain']]}
            rows.append(row)
        summary=summarize(rows);out=root/'generated_report';export(out,dict(status='GENERATED_SCHEMA_FIXTURE_NOT_SCIENCE',rows=rows,costs={},**summary))
        for name,count in [('FINAL_METRICS.csv',28),('DOMAIN_METRICS.csv',84),('PAIRED_COMPARISONS.csv',63)]:
            with (out/name).open() as f:assert len(list(csv.DictReader(f)))==count
        assert len(read(out/'ADAM_COUNTERFACTUAL.json'))==64
        bad=copy.deepcopy(rows);bad[-1]['diagnostics'].pop(next(iter(bad[-1]['diagnostics'])));refuse(lambda:summarize(bad))
        bad=copy.deepcopy(rows);bad[0]['origin']='new_execution';refuse(lambda:summarize(bad))
        refuse(lambda:summarize(rows[:-1]));evidence['report_schema']=dict(final=28,domain=84,paired=63,adam=64,generated_only=True)

    from . import meter
    original_measure=meter.measure
    def counted_measure(t,*a,**kw):
        before=copy.deepcopy(t.alignment_cost)
        try:return original_measure(t,*a,**kw)
        finally:
            for k in ('diagnostic_vjps','adam_candidates','adam_arithmetic_calls','meter_seconds','effective_W_evaluations','effective_W_seconds'):
                meter_cost[k]+=t.alignment_cost[k]-before[k]
    checks=locals()
    with patch.object(meter,'measure',counted_measure):
        for name,expected in CASES.items():
            start=counter.count;counter.cap=min(96,start+expected);operations=None
            try:
                if name=='integrity_and_costs':checks[name]();counts.update(backward=3,backward_attempts=3)
                else:
                    with NativeOperations(root/'operations'/str(len(attempts))/name) as operations:checks[name]()
                assert counter.count-start==expected
                results.append(dict(test=name,status='PASS',optimizer_calls=counter.count-start))
            except Exception:results.append(dict(test=name,status='FAIL',optimizer_calls=counter.count-start,traceback=traceback.format_exc().replace(str(ROOT),'<repo>').replace(str(root),'<generated-evidence>')))
            finally:
                if operations is not None:counts.update(operations.counts)
    current.update(status='PASS' if all(x['status']=='PASS' for x in results) else 'FAIL',results=results,calls=counter.count-begin,seconds=time.time()-current['started'])
    write(root/'ATTEMPTS.json',attempts);write(DOC/'CPU/ATTEMPTS.json',attempts)
    result=dict(status=current['status'],study_id=STUDY,tests=results,evidence=evidence,code_tree_sha256=code_manifest()['code_tree_sha256'],science_sha256=canonical_plan()['plan_sha256'],execution_sha256=digest(execution_plan()),
        cost=dict(original_other_study_CPU_calls=64,new_cumulative_calls=counter.count,new_cap=96,new_attempts=len(attempts),attempt_cap=3,this_attempt_calls=current['calls'],planned_calls=29,operation_counts=counts,meter=meter_cost,seconds=current['seconds']),
        environment=dict(python=sys.version.split()[0],torch=torch.__version__,device='cpu',optimize=sys.flags.optimize,CUDA_initialized=torch.cuda.is_initialized()),real_prefix_tensor_reads=0,patient_payload_reads=0,production_approvals_created=0,CUDA_calls=0,
        pending=['source/prefix real acceptance','CUDA','smoke','formal'])
    for directory in (root,DOC/'CPU'):
        write(directory/('ATTEMPT_'+str(len(attempts))+'_REPORT.json'),result);write(directory/'TEST_REPORT.json',result)
    if result['status']!='PASS':raise RuntimeError('CPU generated failure; preserve costs before any bounded repair')
    return result

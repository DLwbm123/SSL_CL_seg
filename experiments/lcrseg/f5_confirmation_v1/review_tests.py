"""R1 regression tests: generated tensors and aggregates only; zero optimizer calls."""
import copy
import tempfile
from pathlib import Path
from . import protocol as p


def integrity_faults():
    import torch
    from .assurance import schema,verify_file,integrity_current
    from ..five_frameworks_v1.semantics import tensor_fingerprint
    from .tests import rejects
    with tempfile.TemporaryDirectory(prefix='integrity-') as tmp:
        path=Path(tmp)/'student.pt';ident=dict(node_id='SYNTHETIC_ONLY',family=p.F5,seed=163,order=1,stage=2)
        payload=dict(identity=ident,step=2,student={'weight':torch.arange(6,dtype=torch.float32).reshape(2,3)},transform=torch.eye(16))
        receipt=dict(node_id=ident['node_id'],identity=ident,step=2,
                     student_hash=tensor_fingerprint(payload['student']),transform_hash=tensor_fingerprint({'F':payload['transform']}))
        expected=schema(payload['student']);binding=dict(node_id=ident['node_id'],execution_commit='synthetic_fixture',plan_sha256='fixture')
        torch.save(payload,path);rng=torch.get_rng_state().clone()
        proof=verify_file(path,receipt,expected,binding)
        assert torch.equal(rng,torch.get_rng_state()) and integrity_current(path,receipt,proof,binding)
        assert verify_file(path,receipt,expected,binding)['student_hash']==proof['student_hash']
        for change in ('weights','F','identity','schema','nan','F_shape','F_dtype'):
            bad=copy.deepcopy(payload)
            if change=='weights':bad['student']['weight'].add_(1)
            if change=='F':bad['transform'][0,0]=2
            if change=='identity':bad['identity']['node_id']='OTHER'
            if change=='schema':bad['student']['weight']=torch.zeros(3,2)
            if change=='nan':bad['student']['weight'][0,0]=float('nan')
            if change=='F_shape':bad['transform']=torch.eye(4)
            if change=='F_dtype':bad['transform']=bad['transform'].double()
            torch.save(bad,path)
            assert not integrity_current(path,receipt,proof,binding)
            rejects(lambda:verify_file(path,receipt,expected,binding))
        torch.save(payload,path);missing={k:v for k,v in receipt.items() if k!='student_hash'}
        rejects(lambda:verify_file(path,missing,expected,binding))
        path.write_bytes(b'NOT A MODEL CHECKPOINT')
        rejects(lambda:verify_file(path,receipt,expected,binding))
        assert not integrity_current(path,receipt,proof,binding)


def qualification_coverage():
    from .execution import require_qualification,ledger_count
    from .native_qualification import qualification_plan
    from .tests import rejects
    plan=p.read(p.DOC/'PLAN.json');q=qualification_plan()
    # Regression for the native bias-free head; no optimizer or extra RNG stream.
    import torch
    from types import SimpleNamespace
    from .native_qualification import foreground_fixture
    head=torch.nn.Conv2d(16,3,3,bias=False)
    native=SimpleNamespace(decoder=SimpleNamespace(conv_logit=SimpleNamespace(mu=head)))
    keys=list(head.state_dict());rng=torch.get_rng_state().clone()
    foreground_fixture(native)
    assert head.bias is None and list(head.state_dict())==keys and torch.equal(rng,torch.get_rng_state())
    assert (head(torch.ones(2,16,5,5)).argmax(1)==1).all()
    assert q['planned_calls']==sum(c['physical_calls'] for c in q['cases'])==21<=60
    assert len(q['cases'])==12 and {c['kind'] for c in q['cases']}=={'warmup','resume','transition','failure'}
    with tempfile.TemporaryDirectory(prefix='metadata-fixture-') as tmp:
        root=Path(tmp);cfg={'execution_commit':'SYNTHETIC_ONLY'};env={'fixture':'not a runtime qualification'}
        p.write(root/'RUNTIME_ENVIRONMENT.json',dict(fingerprint=env,sha256=p.digest(env)))
        rows=[];cursor=0
        for spec in q['cases']:
            count=spec['physical_calls'];rows.append(dict(id=spec['id'],status='PASS',physical_calls=count,
               ledger_start=cursor+1,ledger_end=cursor+count,checks={k:True for k in spec['checks']}));cursor+=count
        receipt=dict(status='PASS',execution_commit=cfg['execution_commit'],plan_sha256=plan['plan_sha256'],
                     physical_calls=21,environment_sha256=p.digest(env),rows=rows)
        (root/'cuda_physical.jsonl').write_text(''.join(__import__('json').dumps({'invocation':i})+'\n' for i in range(1,22)))
        cost=root/'costs'/'CUDA_QUALIFICATION'/'0001';cost.mkdir(parents=True)
        p.write(cost/'session.json',dict(status='PASS',seconds=0,operation_counts={},peak_cuda_allocated=0,peak_cuda_reserved=0))
        p.write(root/'CUDA_QUALIFICATION.json',receipt)
        require_qualification(root,cfg,plan,p.digest(env),names=('CUDA_QUALIFICATION',))
        for change in ('empty','missing','check','count','environment'):
            bad=copy.deepcopy(receipt)
            if change=='empty':bad['rows']=[]
            if change=='missing':bad['rows'].pop()
            if change=='check':bad['rows'][0]['checks']['zero_U']=False
            if change=='count':bad['rows'][0]['physical_calls']=2
            if change=='environment':bad['environment_sha256']='wrong'
            p.write(root/'CUDA_QUALIFICATION.json',bad)
            rejects(lambda:require_qualification(root,cfg,plan,p.digest(env),names=('CUDA_QUALIFICATION',)))
        (root/'cuda_physical.jsonl').write_text('{"invocation":1}\n{"invocation":3}\n')
        rejects(lambda:ledger_count(root/'cuda_physical.jsonl'),RuntimeError)


def cumulative_costs_rng():
    import torch
    from .assurance import cost_session,session_totals
    from .tests import rejects
    torch.manual_seed(213);x=torch.randn(5,requires_grad=True);before=torch.get_rng_state().clone()
    plain=(x*x).sum();plain.backward();expected=x.grad.clone();x.grad=None
    with tempfile.TemporaryDirectory(prefix='cost-') as tmp:
        root=Path(tmp)
        with cost_session(root,'synthetic_fixture',cuda=False) as ops:
            measured=(x*x).sum();measured.backward()
        assert torch.equal(x.grad,expected) and torch.equal(before,torch.get_rng_state())
        assert ops.counts['backward']==1
        try:
            with cost_session(root,'synthetic_fixture',cuda=False):
                (x*x).sum().backward();raise RuntimeError('synthetic interruption')
        except RuntimeError:pass
        total=session_totals(root)
        assert total['sessions']==2 and total['failed_sessions']==1 and total['operation_counts']['backward']==2
        assert total['worker_seconds']>=0 and total['peak_cuda_allocated']==0
        p.write(root/'0003'/'session.json',dict(status='STARTED'))
        rejects(lambda:session_totals(root),ValueError)
        def enter():
            with cost_session(root,'synthetic_fixture',cuda=False):pass
        rejects(enter,RuntimeError)


def report_artifacts():
    from .reporting import export_reports,validate_exports,complete_report
    from .execution import stage_state
    from .tests import rejects
    rows=[]
    def scores(ds,values):return {d:dict(rim=v-.01,cup=v+.01,disc_union=v+.1,macro_Dice=v) for d,v in zip(ds,values)}
    for f in p.CANDIDATES:
        for seed in (162,163,164):
            for order,domains in enumerate(p.ORDERS,1):
                bonus=.01 if f==p.F5 else 0.
                for stage in (1,2):
                    rows.append(dict(node_id=f'SYNTH_{f}_{seed}_{order}_{stage}',identity=dict(family=f,seed=seed,order=order,stage=stage),
                        scores=scores(domains[:stage+1],[.6+bonus,.7+bonus,.8+bonus]),
                        timeline={'0':scores(domains[:1],[.9]),'1':scores(domains[:2],[.8,.85])},
                        origin='historical_import' if seed==162 and f in (p.B0,p.F5) else 'new_execution'))
    primary=p.paired(rows,[163,164]);supp=p.paired(rows,[162,163,164])
    result=dict(status='SYNTHETIC_RENDERER_FIXTURE_NOT_PRODUCTION',execution_commit='fixture',plan_sha256='fixture',
                rows=rows,primary=primary,supplementary_descriptive=supp,gates=p.decisions(primary),
                costs={'fixture_optimizer_calls':0},integrity={'fixture':'no production acceptance'},qualification={},environment={})
    with tempfile.TemporaryDirectory(prefix='reports-') as tmp:
        root=Path(tmp);export_reports(root,result);validate_exports(root,result)
        assert all(result['gates'][k] for k in ('G1','G2','G3'))
        (root/'FINAL_METRICS.csv').write_text('invalid schema\n')
        rejects(lambda:validate_exports(root,result),ValueError)
    with tempfile.TemporaryDirectory(prefix='pending-') as tmp:
        root=Path(tmp);assert complete_report(root,'fixture')['status']=='PENDING_FULL_MATRIX'
        # A metadata-sealed fake file must NOT satisfy the completion path.
        n=p.matrix()[0];nr=root/n['id'];nr.mkdir()
        (nr/'physical.jsonl').write_text(''.join(__import__('json').dumps({'invocation':i})+'\n' for i in range(1,n['updates']+1)))
        ident={k:n[k] for k in ('family','candidate_id','seed','order','stage','sequence_id','domain')}
        ident.update(node_id=n['id'],execution_commit='fixture')
        p.write(nr/'receipt.json',dict(identity=ident,node_id=n['id'],status='SEALED',step=n['updates'],physical_optimizer_calls=n['updates'],
                 resolved_options=p.read(p.DOC/'PLAN.json')['options'][n['family']][n['domain']]))
        (nr/'student.pt').write_bytes(b'NOT A MODEL')
        assert stage_state(n,root,'fixture')=='METADATA_SEALED'
        assert complete_report(root,'fixture')['status']=='PENDING_INTEGRITY'
        assert not (root/'STATUS.json').exists()

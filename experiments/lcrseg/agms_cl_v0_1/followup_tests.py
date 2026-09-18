"""One independent 21-call synthetic CPU attempt; never runs the old suite."""
import copy
import csv
import os
import sys
import time
from pathlib import Path
from .protocol import DOC, STUDY, read, write, digest, canonical_plan, execution_plan, code_manifest


def validate_cpu(plan, tree):
    r=read(DOC/'CPU/TEST_REPORT.json')
    attempts=read(DOC/'CPU/ATTEMPTS.json')
    lines=(DOC/'CPU/PHYSICAL.jsonl').read_text().splitlines()
    calls=[__import__('json').loads(x)['invocation'] for x in lines]
    if (r['status']!='PASS' or r['study_id']!=STUDY or r['code_tree_sha256']!=tree
        or r['plan_sha256']!=plan['plan_sha256'] or r['execution_sha256']!=digest(execution_plan())
        or len(attempts)!=1 or attempts[0]['status']!='PASS' or attempts[0]['calls']!=21
        or calls!=list(range(1,22)) or r['cost']['new_CPU']!=21
        or set(r['checks'])!={'scope','native','continuation','baseline','failure','dose','overfit','reports','identity'}
        or not all(r['checks'].values()) or r['cost']['real_optimizer']!=0):
        raise PermissionError('fresh independent CPU evidence required')


def run_tests(reference,evidence_root):
    import torch
    from .authority import synthetic_capability, authorize
    from .qualification import generated_trainer, continuation, baseline_equivalence, failures, generated_identity
    from .tests import native_arm_checks, geometry_checks, refuse, public_record
    from .state import validate_payload, metadata_path
    from .execution import accept_prefix, construct, identity, stage_state
    from .reporting import historical, summarize, export
    from ..five_frameworks_v1.native_runner import Counter
    from ..f5_confirmation_v1.assurance import cost_session
    if sys.flags.optimize or 'PYTHONOPTIMIZE' in os.environ or torch.cuda.is_initialized():raise RuntimeError('unoptimized CPU only')
    root=Path(evidence_root).resolve()
    from .protocol import ROOT
    if root==ROOT or ROOT in root.parents:raise RuntimeError('payload evidence outside code')
    root.mkdir(parents=True,exist_ok=False)  # create-only: one attempt, never overwrite/retry
    if (DOC/'CPU/ATTEMPTS.json').exists():raise RuntimeError('independent study attempt already consumed')
    spec=execution_plan()['CPU'];assert spec['planned_calls']==21 and spec['cap']==24
    counter=Counter(root/'PHYSICAL.jsonl',spec['cap']);attempt=dict(status='STARTED',planned_calls=21,calls=0,started=time.time())
    write(root/'ATTEMPTS.json',[attempt]);write(DOC/'CPU/ATTEMPTS.json',[attempt])
    torch.set_num_threads(2);device=torch.device('cpu');permit=synthetic_capability();checks={}
    try:
        with cost_session(root/'costs','DS_HALF_CPU',cuda=False):
            plan=canonical_plan()
            assert len(plan['nodes'])==2 and sum(n['updates'] for n in plan['nodes'])==5300
            assert {n['arm'] for n in plan['nodes']}=={'A5'} and plan['method']['lambda_DS']==.125
            from .protocol import validate_plan, controls
            assert controls()['gates']=={'P_perf':False,'P_joint':False}
            for key,val in [('lambda_DS',.25),('lambda_H',.6)]:
                wrong=copy.deepcopy(plan);wrong['method'][key]=val;wrong['plan_sha256']=digest({k:v for k,v in wrong.items() if k!='plan_sha256'})
                refuse(lambda:validate_plan(wrong))
            refuse(lambda:authorize({}, {}, {}))
            refuse(lambda:accept_prefix(plan['nodes'][0],{},plan,permit,root,device))
            refuse(lambda:construct(plan['nodes'][0],{},plan,permit,Path('/never-opened'),{},device))
            geometry_checks();checks['scope']=True
            t=generated_trainer(reference,'A5',device,permit)
            native_arm_checks(t,counter);checks['native']=True
            d=t.diagnostics['2']['gradients']['DS']['groups']
            assert all(abs(v['weighted_norm']-.125*v['raw_norm'])<1e-6 for v in d.values())
            assert abs(t.last['losses']['weighted_DS']-.125*t.last['losses']['DS'])<1e-6
            assert abs(t.last['labeled_loss']-(t.last['losses']['supervised']+t.last['losses']['constraint']+.125*t.last['losses']['DS']))<2e-6
            checks['dose']=True;del t
            t,path=continuation(reference,'A5',device,permit,counter,root,32);checks['continuation']=True
            v=torch.load(path,map_location=device,weights_only=False);meta=read(metadata_path(path))
            for field in ['lambda_DS','prefix','identity']:
                bad=copy.deepcopy(v)
                if field=='lambda_DS':bad['agms']['method']['lambda_DS']=.25
                elif field=='prefix':bad['agms']['prefix']={}
                else:bad['identity']['study_id']='AGMS_CL_V0_1'
                refuse(lambda:validate_payload(bad,meta,generated_identity(t),'A5',t.options,t.provider))
            checks['identity']=True
            rows=historical(plan);assert len(rows)==4 and summarize(rows)['gates'] is None
            for n in plan['nodes']:
                base=copy.deepcopy(next(x for x in rows if x['arm']=='A5_CONTROL' and x['identity']['order']==n['order']))
                base.update(arm='A5',node_id=n['id'],identity=identity(n,{'execution_commit':'generated'}),origin='new_execution',support={},extra_cost={'diagnostic_vjps':16})
                base['diagnostics']={str(k):copy.deepcopy(t.diagnostics['2']) for k in execution_plan()['diagnostics'][n['domain']]}
                rows.append(base)
            analysis=summarize(rows);assert analysis['gates']=={'DS_PRESSURE_SUPPORTED':False}
            export(root/'reports',{**analysis,'rows':rows,'status':'GENERATED_TEST_ONLY','costs':{}})
            for name,count in [('FINAL_METRICS.csv',6),('DOMAIN_METRICS.csv',18),('PAIRED_COMPARISONS.csv',9)]:
                assert len(list(csv.DictReader((root/'reports'/name).open())))==count
            assert len(read(root/'reports/GRADIENT_DIAGNOSTICS.json'))==8
            tail=root/'tail'/plan['nodes'][0]['id'];tail.mkdir(parents=True)
            (tail/'physical.jsonl').write_text('{"invocation":1,"scientific_step_attempt":1}\n')
            refuse(lambda:stage_state(plan['nodes'][0],root/'tail',{'execution_commit':'generated'}))
            from ..five_frameworks_v1.checkpoint import atomic_save
            from ..five_frameworks_v1.semantics import tensor_fingerprint
            from ..f5_confirmation_v1.assurance import verify_file,schema,integrity_current
            deployed=t.model.deploy();weights=deployed.parent.native.state_dict();transform=deployed.transform.detach()
            ident={**generated_identity(t),'node_id':'SYNTHETIC_INTEGRITY'}
            payload=root/'synthetic_integrity.pt'
            atomic_save(dict(identity=ident,student=weights,transform=transform,step=t.step),payload)
            receipt=dict(node_id=ident['node_id'],identity=ident,step=t.step,
                student_hash=tensor_fingerprint(weights),transform_hash=tensor_fingerprint({'F':transform}))
            proof=verify_file(payload,receipt,schema(weights),{'node_id':ident['node_id']})
            assert integrity_current(payload,receipt,proof,{'node_id':ident['node_id']})
            bad=copy.deepcopy(receipt);bad['student_hash']='0'*64
            refuse(lambda:verify_file(payload,bad,schema(weights),{'node_id':ident['node_id']}))
            checks['reports']=True;del t
            baseline_equivalence(reference,device,permit,counter,32);checks['baseline']=True
            failures(reference,device,permit,counter,32);checks['failure']=True
            # Two bounded fixed-batch L-only fitting cases, synthetic fixtures only.
            for mode in ['stripes','background']:
                t=generated_trainer(reference,'A5',device,permit);t.options['warmup_fraction']=1.
                x,y,ids=t.provider.labeled(0)
                if mode=='background':y=y*0
                t.provider.labeled=lambda *a, x=x,y=y,ids=ids,**kw:(x,y,ids)
                counter.wrap(t.optimizer);loss=[]
                for _ in range(4):t.update();loss.append(t.last['labeled_loss'])
                if not loss[-1]<loss[0]:raise AssertionError(('synthetic fitting did not decrease',mode,loss))
            checks['overfit']=True
        assert counter.count==21
        attempt['status']='PASS'
    except BaseException as e:
        attempt.update(status='FAIL',error=repr(e));raise
    finally:
        attempt.update(calls=counter.count,seconds=time.time()-attempt['started'])
        report=dict(status=attempt['status'],study_id=STUDY,checks=checks,
          plan_sha256=canonical_plan()['plan_sha256'],execution_sha256=digest(execution_plan()),
          code_tree_sha256=code_manifest()['code_tree_sha256'],
          cost=dict(new_CPU=counter.count,attempts=1,prior_AGMS_CPU=80,earlier_CPU=134,real_optimizer=0),
          environment=dict(python=sys.version.split()[0],torch=torch.__version__,device='cpu',optimize=sys.flags.optimize),
          future=dict(CUDA='PENDING_FRESH_REVIEW',smoke='PENDING',formal='PENDING'))
        for name,value in [('ATTEMPTS',[attempt]),('TEST_REPORT',report)]:
            write(root/(name+'.json'),value);write(DOC/'CPU'/(name+'.json'),public_record(value,root))
        if (root/'PHYSICAL.jsonl').exists():(DOC/'CPU/PHYSICAL.jsonl').write_bytes((root/'PHYSICAL.jsonl').read_bytes())
        for p in (root/'costs').glob('*/session.json'):write(DOC/'CPU/costs'/p.parent.name/'session.json',read(p))
    return report

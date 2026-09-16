"""Finite controller around the unchanged native target_task, with fresh authority.

No call to native_runner.admit/run_finite, no old DAG, no source training,
no new optimizer interception. The existing Counter supplies durable calls.
"""
import contextlib
import fcntl
import json
import os
import subprocess
from pathlib import Path
from .protocol import (ROOT,DOC,OLD,PARENT,WORKER,B0,B2,F5,CAPS,MANIFEST,SPLIT,
                       read,write,digest,manifest,validate_plan,metrics,paired,decisions)


def authorization(review, launch, actual):
    """Validate metadata, never generate approval or infer it from passing tests."""
    repair=launch.get('engineering_repair')
    if repair is not None:
        base=repair.get('reviewed_base_bindings',{})
        if (repair.get('user_instruction')!='你解决一下问题，并继续训练'
                or repair.get('authority')!='explicit_user_engineering_repair_and_continue'
                or base.get('reviewed_code_commit')!='2070845b679da42ea7a4df4268f492f2943deb6e'):
            raise PermissionError('explicit bounded engineering repair authority required')
        # The external approval stays byte-for-byte original and covers only base.
        original={**launch,**base};original.pop('engineering_repair')
        authorization(review,original,base)
        if (set(base)!=set(actual) or any(actual[k]!=base[k] for k in
                ('plan_sha256','source_reuse_sha256','parent_binding_sha256'))
                or any(launch.get(k)!=v for k,v in actual.items())):
            raise PermissionError('repair must preserve scientific/source/parent binding')
        return
    if (review.get('study_id') != 'F5_CONFIRMATION_V1' or review.get('is_template',True)
            or review.get('decision') != 'APPROVED_FOR_EXPERIMENTS'
            or not review.get('reviewer') or not review.get('review_evidence')
            or review.get('reviewer_role') != 'external' or review.get('approved_phases') != ['P1']):
        raise PermissionError('STOP_AWAITING_EXTERNAL_CODE_REVIEW')
    for key,value in actual.items():
        if review.get(key) != value:
            raise PermissionError('review binding mismatch: '+key)
    if review.get('caps') != CAPS:
        raise PermissionError('review must bind exact separate caps')
    if (launch.get('study_id') != 'F5_CONFIRMATION_V1' or launch.get('user_confirmed') is not True
            or launch.get('review_sha256') != digest(review)
            or any(launch.get(k) != v for k,v in actual.items())):
        raise PermissionError('fresh user launch confirmation bound to this review required')


def preflight(config):
    # All approval checks precede imports that can reach tensor/patient IO.
    plan=validate_plan(read(DOC/'PLAN.json'))
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    subprocess.run(['git','diff','--quiet','HEAD'],cwd=ROOT,check=True)
    actual=dict(reviewed_code_commit=head,code_tree_sha256=manifest()['code_tree_sha256'],
                plan_sha256=plan['plan_sha256'],source_reuse_sha256=digest(read(DOC/'SOURCE_REUSE.json')),
                parent_binding_sha256=digest(read(OLD/'PARENT_BINDING.json')))
    if actual['code_tree_sha256'] != read(DOC/'CODE_MANIFEST.json')['code_tree_sha256']:
        raise PermissionError('code differs from manifest')
    authorization(read(config['review']),read(config['launch_confirmation']),actual)
    launch=read(config['launch_confirmation'])
    if launch.get('engineering_repair') is not None:
        base=launch['engineering_repair']['reviewed_base_bindings']['reviewed_code_commit']
        subprocess.run(['git','merge-base','--is-ancestor',base,head],cwd=ROOT,check=True)
        changed=set(subprocess.check_output(['git','diff','--name-only',base,head],cwd=ROOT,text=True).splitlines())
        allowed={'experiments/lcrseg/f5_confirmation_v1/'+n for n in
                 ('native_qualification.py','execution.py','review_tests.py')}
        allowed.update('experiments/lcrseg/docs/f5_confirmation_v1/'+n for n in
                       ('CODE_MANIFEST.json','TEST_REPORT.json','CPU_ATTEMPTS.json','CPU_PHYSICAL.jsonl','NATIVE_FIX_REPORT.md'))
        if not changed or not changed<=allowed:
            raise PermissionError('engineering repair exceeds qualification/admission-only file scope')
    cpu=read(DOC/'TEST_REPORT.json')
    if cpu['status']!='PASS' or cpu['code_tree_sha256']!=actual['code_tree_sha256'] or cpu['plan_sha256']!=plan['plan_sha256']:
        raise PermissionError('current code-bound CPU qualification required')
    if Path(config['code']).resolve()!=ROOT or config['execution_commit']!=head:
        raise PermissionError('runtime checkout differs from reviewed commit')
    root=Path(config['run_root']).resolve();old=Path(config['reuse_root']).resolve()
    if root==old or root in old.parents or old in root.parents:
        raise PermissionError('new run must be isolated from historical results')
    from ..five_frameworks_v1.integration import ExecutionPermit,_PERMIT_SEAL
    from ..five_frameworks_v1.native_runner import options_for
    from ..five_frameworks_v1.native_data import inspect
    from ..five_frameworks_v1.native_parent import IDENTITY
    if IDENTITY!=PARENT:
        raise PermissionError('parent changed')
    study=read(OLD/'RESOLVED_PROTOCOL.json')['study']
    selections={'SELECT_PARENT':{'candidate_id':'P1'}}
    for n in plan['nodes']:
        if options_for(n,study,selections)!=plan['options'][n['family']][n['domain']]:
            raise ValueError('CONFIG_BINDING_MISMATCH: actual resolver '+n['id'])
    digests=[digest(dict(domain=n['domain'],seed=n['seed'],order=n['order'],stage=n['stage'],
                         manifest=MANIFEST,split=SPLIT)) for n in plan['nodes']]
    permit=ExecutionPermit({**actual,'execution_scope':'formal','authorized_manifest_digests':digests},('P1',),CAPS,_PERMIT_SEAL)
    inspect(config['data'])  # Metadata hashes/counts only; no patient payload here.
    return plan,permit,study


@contextlib.contextmanager
def owned_root(config,plan):
    root=Path(config['run_root'])
    # Protocol roots contain large checkpoints: require the canonical NAS wrapper.
    canonical=Path(os.environ['SSLCL_STORAGE_ROOT']).resolve()
    if canonical not in root.resolve().parents:
        raise PermissionError('run root must be under canonical NAS')
    root.mkdir(parents=True,exist_ok=True)
    fs=subprocess.check_output(['findmnt','-rn','-T',str(root),'-o','FSTYPE'],text=True).strip()
    if fs not in ('nfs','nfs4'):raise PermissionError('NAS wrapper/mount required')
    import shutil
    if shutil.disk_usage(root).free < 4*1024**3:
        raise RuntimeError('RESOURCE_WAIT: less than 4 GiB NAS output margin')
    with __import__('tempfile').NamedTemporaryFile(dir=root) as probe:
        probe.write(b'NAS_WRITE_OK');probe.flush();os.fsync(probe.fileno());probe.seek(0)
        if probe.read()!=b'NAS_WRITE_OK':raise OSError('NAS probe failed')
    if canonical not in root.resolve().parents:
        raise PermissionError('run root must be under canonical NAS')
    fd=(root/'.controller.lock').open('a')
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        marker=root/'RUN_IDENTITY.json'
        wanted={'study_id':plan['study_id'],'plan_sha256':plan['plan_sha256'],'execution_commit':config['execution_commit']}
        if marker.exists():
            if read(marker)!=wanted:raise PermissionError('existing run identity differs')
        elif any(p.name!='.controller.lock' for p in root.iterdir()):
            raise PermissionError('nonempty unbound run root')
        else:write(marker,wanted)
        yield root
    finally:
        fd.close()


def ledger_count(path):
    path=Path(path)
    rows=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    if any(r.get('invocation')!=i for i,r in enumerate(rows,1)):
        raise RuntimeError('ENGINEERING_STOP: corrupt physical ledger')
    return len(rows)


def stage_state(node, root, execution_commit):
    """Metadata only. This state never proves model tensor integrity."""
    nr=root/node['id'];count=ledger_count(nr/'physical.jsonl');cap=node['updates']
    if count>cap:raise RuntimeError('ENGINEERING_STOP: physical stage cap')
    receipt=nr/'receipt.json'
    if receipt.exists():
        r=read(receipt);i=r['identity']
        wanted={k:node[k] for k in ('family','candidate_id','seed','order','stage','sequence_id','domain')}
        wanted.update(node_id=node['id'],execution_commit=execution_commit)
        if (i!=wanted or r.get('node_id')!=node['id'] or r['status']!='SEALED' or r['step']!=cap
                or r['physical_optimizer_calls']!=count or count!=cap
                or not (nr/'student.pt').is_file() or (nr/'student.pt').stat().st_size<=0):
            raise RuntimeError('ENGINEERING_STOP: invalid sealed receipt')
        return 'METADATA_SEALED'
    if (nr/'failure.json').exists():
        raise RuntimeError('ENGINEERING_STOP: prior failure; no automatic retry')
    checkpoint=nr/'latest.pt'
    if checkpoint.exists():
        # Metadata permits rejecting a lost tail before opening any tensor file.
        committed=read(nr/'latest.pt.receipt.json')
        if not committed['committed'] or committed['step']!=count:
            raise RuntimeError('ENGINEERING_STOP: physical calls exceed saved step; no replay budget')
    elif count:
        raise RuntimeError('ENGINEERING_STOP: physical calls without recoverable checkpoint')
    return 'RESUME' if checkpoint.exists() else 'NEW'


def check_costs(plan,root):
    ids={n['id'] for n in plan['nodes']}
    for p in root.iterdir():
        if p.is_dir() and (p/'physical.jsonl').exists() and p.name not in ids and p.name not in plan['reused_sources']:
            raise RuntimeError('ENGINEERING_STOP: unknown formal node ledger')
    formal=sum(ledger_count(root/n['id']/'physical.jsonl') for n in plan['nodes'])
    smoke=ledger_count(root/'smoke_physical.jsonl');cuda=ledger_count(root/'cuda_physical.jsonl')
    if formal>CAPS['formal_physical'] or smoke>24 or cuda>60 or formal+smoke>74224:
        raise RuntimeError('ENGINEERING_STOP: cumulative budget exceeded')
    return dict(formal=formal,real_L_smoke=smoke,synthetic_cuda=cuda,real_total=formal+smoke)


def verify_sources(config,root,device,audit=None):
    """Future production only: actual tensor hash, schema and synthetic forward."""
    import torch
    from ..five_frameworks_v1.native_parent import build
    from ..five_frameworks_v1.semantics import tensor_fingerprint
    rows={};cost=audit if audit is not None else {}
    cost.update(tensor_loads=0,tensor_hashes=0,bytes_in_tensor_files=0,synthetic_forwards=0,synthetic_images=0,patient_forwards=0,optimizer_calls=0)
    for expected in read(DOC/'SOURCE_REUSE.json')['sources']:
        ident=expected['node_id'];old=Path(config['reuse_root'])/ident;r=read(old/'receipt.json')
        if r['identity']!=expected['identity'] or r['student_hash']!=expected['student_hash_declared'] or r['status']!='SEALED':
            raise RuntimeError('SOURCE_REUSE_BLOCKED: receipt identity/hash mismatch '+ident)
        cost['tensor_loads']+=1;cost['bytes_in_tensor_files']+=(old/'student.pt').stat().st_size
        value=torch.load(old/'student.pt',map_location='cpu',weights_only=False)
        cost['tensor_hashes']+=1;actual_hash=tensor_fingerprint(value['student'])
        if (value['identity']!=r['identity'] or value['step']!=8000 or value['transform'] is not None
                or actual_hash!=r['student_hash']):
            raise RuntimeError('SOURCE_REUSE_BLOCKED: actual tensors '+ident)
        m=build(config['reference'],device,r['identity']['seed']);m.load_state_dict(value['student'],strict=True)
        m.eval().requires_grad_(False)
        with torch.no_grad():
            cost['synthetic_forwards']+=1;cost['synthetic_images']+=2
            out=m(torch.zeros(2,3,384,384,device=device),stochastic_classifier=False)[0]
            if out.shape!=(2,3,384,384) or not torch.isfinite(out).all():
                raise RuntimeError('SOURCE_REUSE_BLOCKED: synthetic forward '+ident)
        del m,value,out
        link=root/ident
        if link.is_symlink():
            if link.resolve()!=old.resolve():raise PermissionError('source link changed')
        elif link.exists():raise PermissionError('source destination exists and is not approved read-only reuse link')
        else:link.symlink_to(old.resolve(),target_is_directory=True)
        rows[ident]=r
    write(root/'SOURCE_PREFLIGHT.json',dict(status='PASS',execution_commit=config['execution_commit'],sources=list(rows),cost=cost))
    return rows


def choose_gpu():
    free={int(a):int(b) for a,b in (line.split(',') for line in subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines())}
    gpu=next((g for g in (5,6,7) if free.get(g,0)>=12000),None)
    if gpu is None:raise RuntimeError('RESOURCE_WAIT: no authorized GPU has 12 GB free; no process changed')
    # Single finite controller owns one GPU; no duplicated launcher or parallel state races.
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    return gpu


def require_qualification(root,config,plan,environment_sha256=None,names=('CUDA_QUALIFICATION','SMOKE')):
    """Require exact cases, individual checks, contiguous ranges and live ledger."""
    from .native_qualification import qualification_plan
    from .assurance import session_totals
    recorded=read(root/'RUNTIME_ENVIRONMENT.json')
    env_hash=digest(recorded['fingerprint'])
    if recorded['sha256']!=env_hash or (environment_sha256 is not None and environment_sha256!=env_hash):
        raise PermissionError('runtime environment mismatch')
    for name in names:
        r=read(root/(name+'.json'))
        specs=plan['native_qualification']['cases'] if name=='CUDA_QUALIFICATION' else [
            dict(id=f+'/smoke',physical_calls=8,checks=['L_only','discarded']) for f in (B0,B2,F5)]
        expected=sum(c['physical_calls'] for c in specs)
        ledger='cuda_physical.jsonl' if name=='CUDA_QUALIFICATION' else 'smoke_physical.jsonl'
        if (r['status']!='PASS' or r['execution_commit']!=config['execution_commit']
                or r['plan_sha256']!=plan['plan_sha256'] or r['environment_sha256']!=env_hash
                or r['physical_calls']!=expected or ledger_count(root/ledger)!=expected):
            raise PermissionError('missing/mismatched '+name)
        rows=r.get('rows',[])
        if len(rows)!=len(specs) or [c['id'] for c in rows]!=[c['id'] for c in specs]:
            raise PermissionError('qualification case coverage')
        cursor=0
        for row,spec in zip(rows,specs):
            calls=spec['physical_calls']
            if (row['status']!='PASS' or row['physical_calls']!=calls or row['ledger_start']!=cursor+1
                    or row['ledger_end']!=cursor+calls or row['checks']!={k:True for k in spec['checks']}):
                raise PermissionError('qualification case evidence '+spec['id'])
            cursor+=calls
        session_totals(root/'costs'/name)


def integrity_bindings(node,config,plan):
    return dict(node_id=node['id'],execution_commit=config['execution_commit'],plan_sha256=plan['plan_sha256'],
                code_tree_sha256=read(DOC/'CODE_MANIFEST.json')['code_tree_sha256'])


def accept_target(node,config,plan,permit,root,device):
    """Only production caller owns permit. Tests use generated verify_file fixtures."""
    import random,torch
    from .assurance import verify_file,schema,cost_session
    from ..five_frameworks_v1.native_parent import build
    permit.validate();nr=root/node['id'];receipt=read(nr/'receipt.json')
    python_state=random.getstate()
    try:
        with torch.random.fork_rng(devices=[device.index or 0]), cost_session(nr/'integrity_costs','model_integrity') as observations:
            model=build(config['reference'],device,node['seed'])
            expected=schema(model.state_dict());del model
            observations.extra_cost={}
            record=verify_file(nr/'student.pt',receipt,expected,integrity_bindings(node,config,plan),observations.extra_cost)
            observations.extra_cost=record['cost']
    finally:random.setstate(python_state)
    write(nr/'INTEGRITY.json',record)
    return record


def qualify(config,mode):
    plan,permit,study=preflight(config);choose_gpu()
    import torch
    from dataclasses import replace
    from .assurance import environment,bind_environment,cost_session,session_totals
    from ..single_teacher_scd_v0_1.engine import precision
    from ..five_frameworks_v1.native_runner import Counter
    from ..five_frameworks_v1.native_parent import build,NativeLRParent
    from ..five_frameworks_v1.native_data import NativeCurrentDomain
    from ..five_frameworks_v1.model import Model
    from ..five_frameworks_v1.train_stage import StageTrainer
    precision();device=torch.device('cuda:0');torch.cuda.set_device(device)
    name='CUDA_QUALIFICATION' if mode=='cuda' else 'SMOKE'
    with owned_root(config,plan) as root:
        env_hash=bind_environment(root,environment());path=root/(name+'.json')
        if path.exists():
            require_qualification(root,config,plan,env_hash,names=(name,))
            return {'status':'VERIFIED_QUALIFICATION_SKIP','qualification':name}
        ledger=root/('cuda_physical.jsonl' if mode=='cuda' else 'smoke_physical.jsonl')
        if ledger_count(ledger):raise RuntimeError('ENGINEERING_STOP: partial qualification; no automatic retry')
        cap=60 if mode=='cuda' else 24;counter=Counter(ledger,cap)
        permit=replace(permit,bindings={**permit.bindings,'execution_scope':'synthetic' if mode=='cuda' else 'smoke'})
        if mode=='smoke':
            require_qualification(root,config,plan,env_hash,names=('CUDA_QUALIFICATION',))
            with cost_session(root/'costs'/'SOURCE_PREFLIGHT','source_integrity') as observations:
                observations.extra_cost={}
                sources=verify_sources(config,root,device,observations.extra_cost)
                observations.extra_cost=read(root/'SOURCE_PREFLIGHT.json')['cost']
        try:
            with cost_session(root/'costs'/name,name):
                if mode=='cuda':
                    from .native_qualification import run_cases
                    rows=run_cases(config,plan,permit,root,device,counter)
                else:
                    rows=[]
                    for family in (B0,B2,F5):
                        source=sources['SOURCE_S163'];sid=dict(node_id=source['node_id'],seed=163,domain='REFUGE',student_hash=source['student_hash'],transform_hash=None)
                        native=build(config['reference'],device,163)
                        value=torch.load(root/'SOURCE_S163'/'student.pt',map_location=device,weights_only=False)
                        native.load_state_dict(value['student']);del value
                        provider=NativeCurrentDomain(config['data'],163,1,1,sid,device,permit,allow_u=False)
                        options=plan['options'][family]['RIM_ONE_r3'].copy()
                        t=StageTrainer(Model(NativeLRParent(native,163,sid),family,ratio=options.get('rank_ratio',.5)).to(device),provider,options,execution=permit)
                        counter.wrap(t.optimizer);start=counter.count
                        for _ in range(8):t.update()
                        if provider.u_reads:raise RuntimeError('forbidden smoke U read')
                        rows.append(dict(id=family+'/smoke',status='PASS',physical_calls=counter.count-start,
                                         ledger_start=start+1,ledger_end=counter.count,checks=dict(L_only=True,discarded=True),
                                         telemetry=t.telemetry,U_batches=0))
                        del t,native,provider;torch.cuda.empty_cache()
            result=dict(status='PASS',execution_commit=config['execution_commit'],plan_sha256=plan['plan_sha256'],
                        environment_sha256=env_hash,physical_calls=counter.count,cap=cap,rows=rows,inherited_by_formal=False,
                        measured_cost=session_totals(root/'costs'/name))
            write(path,result);require_qualification(root,config,plan,env_hash,names=(name,));return result
        except BaseException as e:
            write(path,dict(status='ENGINEERING_STOP',execution_commit=config['execution_commit'],error=str(e),physical_calls=counter.count));raise


def run(config):
    plan,permit,study=preflight(config)
    choose_gpu()
    import torch
    from ..five_frameworks_v1.native_runner import target_task
    from ..single_teacher_scd_v0_1.engine import precision
    from .assurance import cost_session,session_totals,environment,bind_environment
    precision();device=torch.device('cuda:0');torch.cuda.set_device(device)
    with owned_root(config,plan) as root:
        env_hash=bind_environment(root,environment());require_qualification(root,config,plan,env_hash)
        with cost_session(root/'costs'/'SOURCE_PREFLIGHT','source_integrity') as observations:
            observations.extra_cost={}
            receipts=verify_sources(config,root,device,observations.extra_cost)
            observations.extra_cost=read(root/'SOURCE_PREFLIGHT.json')['cost']
        receipts['SELECT_PARENT']={'candidate_id':'P1'}
        for node in plan['nodes']:
            check_costs(plan,root)
            state=stage_state(node,root,config['execution_commit']);nr=root/node['id']
            if state=='METADATA_SEALED':
                accept_target(node,config,plan,permit,root,device)
                receipts[node['id']]=read(nr/'receipt.json');continue
            if node['parent_checkpoint'] not in receipts:raise RuntimeError('unsealed own predecessor')
            nr.mkdir(parents=True,exist_ok=True)
            try:
                if state=='RESUME':
                    saved=torch.load(nr/'latest.pt',map_location='cpu',weights_only=False)
                    if saved['step']!=ledger_count(nr/'physical.jsonl') or saved['cursor']!=saved['step']:
                        raise RuntimeError('ENGINEERING_STOP: actual saved state differs from ledger')
                    del saved
                # The original task constructs/reset/restores, trains, merges and evaluates unchanged.
                with cost_session(nr/'costs','formal_target'):
                    result=target_task(config,node,permit,study,receipts,nr,device)
                measured=session_totals(nr/'costs')
                result.update(operation_counts=measured['operation_counts'],peak_cuda_allocated=measured['peak_cuda_allocated'],
                              peak_cuda_reserved=measured['peak_cuda_reserved'],session_cost=measured)
                if result['physical_optimizer_calls']!=node['updates'] or result['step']!=node['updates']:
                    raise RuntimeError('ENGINEERING_STOP: sealed budget mismatch')
                write(nr/'receipt.json',result)
                accept_target(node,config,plan,permit,root,device);receipts[node['id']]=result
                write(root/'STATUS.json',dict(status='RUNNING',sealed=sum(n['id'] in receipts for n in plan['nodes']),total=28))
            except BaseException as e:
                write(nr/'failure.json',dict(status='ENGINEERING_STOP',error=repr(e)));raise
        result=report(root,config['execution_commit'])
        if result['status']!='COMPLETE_P1_AWAITING_SCIENTIFIC_REVIEW':
            write(root/'STATUS.json',dict(status=result['status'],sealed=28,total=28,P2_started=False,publication_status='NOT_READY'))
        return result


def report(root,execution_commit):
    from .reporting import complete_report
    return complete_report(root,execution_commit)

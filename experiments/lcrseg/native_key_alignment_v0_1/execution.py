"""Future authorized finite D1; no production authority is shipped or inferred."""
import copy
import math
from pathlib import Path
from .protocol import STUDY,DOC,read,write,digest,canonical_plan,validate_prefix
from .authority import preflight,CAPS,Capability,_SEAL,execution_plan
from ..f5_confirmation_v1.execution import owned_root,choose_gpu,ledger_count
from ..f5_confirmation_v1.assurance import cost_session,session_totals,environment,bind_environment,verify_file,schema,integrity_current


def identity(node,config):
    return dict(study_id=STUDY,arm=node['arm'],family='B2_PARENT_PAS_KL',candidate_id='B2_PARENT_PAS_KL_C06',
        seed=node['seed'],order=node['order'],stage=2,sequence_id=node['id'].removesuffix('__STAGE2'),
        domain=node['domain'],node_id=node['id'],execution_commit=config['execution_commit'],
        prefix_binding_sha256=node['prefix_binding_sha256'])


def proof_binding(node,config,plan):
    return dict(node_id=node['id'],execution_commit=config['execution_commit'],plan_sha256=plan['plan_sha256'],
        code_tree_sha256=read(DOC/'CODE_MANIFEST.json')['code_tree_sha256'])


def source_identity(node,receipt):
    return dict(node_id=receipt['node_id'],seed=node['seed'],domain=receipt['identity']['domain'],
        student_hash=receipt['student_hash'],transform_hash=receipt['transform_hash'],
        prefix_binding_sha256=node['prefix_binding_sha256'])


def check_costs(root,plan):
    counts={n['id']:ledger_count(root/n['id']/'physical.jsonl') for n in plan['nodes']}
    for n in plan['nodes']:
        if counts[n['id']]>n['updates']:raise RuntimeError('formal stage cap exceeded')
    if any(p.parent.name not in counts for p in root.glob('*/physical.jsonl')):raise RuntimeError('unknown node ledger')
    result=dict(formal=sum(counts.values()),smoke=ledger_count(root/'smoke_physical.jsonl'),
                synthetic_cuda=ledger_count(root/'cuda_physical.jsonl'))
    if result['formal']>42400 or result['smoke']>32 or result['synthetic_cuda']>24 or result['formal']+result['smoke']>42432:
        raise RuntimeError('cumulative protocol budget exceeded')
    return result


def stage_state(node,root,config):
    nr=root/node['id'];calls=ledger_count(nr/'physical.jsonl')
    if calls>node['updates'] or (nr/'failure.json').exists():raise RuntimeError('failed/over-budget node; no automatic retry')
    if (nr/'receipt.json').exists():
        r=read(nr/'receipt.json')
        if (r['identity']!=identity(node,config) or r['status']!='SEALED' or r['step']!=node['updates']
                or r['step']!=calls or r['physical_optimizer_calls']!=calls):raise RuntimeError('invalid sealed receipt')
        return 'METADATA_SEALED'
    if (nr/'latest.pt').exists():
        from .state import metadata_path
        m=read(metadata_path(nr/'latest.pt'))
        if m['identity']!=identity(node,config) or m['step']!=calls:
            raise RuntimeError('lost optimizer tail; cannot replay')
        return 'RESUME'
    if calls:raise RuntimeError('physical calls without committed checkpoint')
    return 'NEW'


def accept_prefix(node,config,plan,permit,root,device):
    if not isinstance(permit,Capability):raise PermissionError('new study capability required')
    permit.validate()
    if node not in canonical_plan()['nodes'] or plan!=canonical_plan():raise PermissionError('canonical node required')
    if permit.bindings.get('execution_scope') not in ('formal','smoke'):raise PermissionError('real prefix capability required')
    import torch
    from ..five_frameworks_v1.native_parent import build
    record=next(p for p in plan['prefixes'] if p['node_id']==node['prefix_node'])
    old=Path(config['prefix_root'])/record['node_id'];receipt=read(old/'receipt.json')
    validate_prefix(node,record,receipt)
    with cost_session(root/'costs'/('PREFIX_'+record['node_id']),'prefix_integrity') as obs:
        obs.extra_cost={}
        with torch.random.fork_rng(devices=[device] if device.type=='cuda' else []):native=build(config['reference'],device,node['seed'])
        actual=verify_file(old/'student.pt',receipt,schema(native.state_dict()),
            dict(node_id=record['node_id'],execution_commit=record['origin_execution_commit']),obs.extra_cost)
        if (actual['file_sha256']!=record['file_sha256'] or actual['student_hash']!=record['student_sha256']
                or actual['transform_hash']!=record['transform_sha256']):raise RuntimeError('PREFIX_REUSE_BLOCKED')
        del native
    write(root/'prefix_proofs'/(record['node_id']+'.json'),actual)
    return old,receipt


def accept_target(node,config,plan,permit,root,device):
    if not isinstance(permit,Capability):raise PermissionError('new study capability required')
    permit.validate()
    if permit.bindings.get('execution_scope')!='formal' or node not in canonical_plan()['nodes'] or plan!=canonical_plan():
        raise PermissionError('canonical formal integrity request required')
    import torch
    from ..five_frameworks_v1.native_parent import build
    nr=root/node['id'];receipt=read(nr/'receipt.json')
    with cost_session(nr/'integrity_costs','target_integrity') as obs:
        obs.extra_cost={}
        with torch.random.fork_rng(devices=[device] if device.type=='cuda' else []):model=build(config['reference'],device,node['seed'])
        result=verify_file(nr/'student.pt',receipt,schema(model.state_dict()),proof_binding(node,config,plan),obs.extra_cost)
    write(nr/'INTEGRITY.json',result)


def construct(node,config,plan,permit,prefix_path,receipt,device,smoke=False):
    if not isinstance(permit,Capability):raise PermissionError('new study capability required')
    permit.validate()
    if permit.bindings.get('execution_scope') not in ('formal','smoke') or node not in canonical_plan()['nodes'] or plan!=canonical_plan():
        raise PermissionError('canonical production request required before prefix payload')
    import torch
    from ..five_frameworks_v1.native_parent import build,NativeLRParent
    from ..five_frameworks_v1.native_data import NativeCurrentDomain
    from ..five_frameworks_v1.model import Model
    from .trainer import KeyAlignmentTrainer
    from ..five_frameworks_v1.semantics import tensor_fingerprint
    record=next(p for p in plan['prefixes'] if p['node_id']==node['prefix_node'])
    validate_prefix(node,record,receipt)
    if prefix_path.resolve()!=(Path(config['prefix_root'])/record['node_id']).resolve():
        raise PermissionError('prefix payload path outside bound node')
    if smoke!=(permit.bindings['execution_scope']=='smoke'):raise PermissionError('smoke/provider scope mismatch')
    opts=plan['options'][node['domain']];sid=source_identity(node,receipt)
    value=torch.load(prefix_path/'student.pt',map_location=device,weights_only=False)
    if value['identity']!=receipt['identity'] or tensor_fingerprint(value['student'])!=receipt['student_hash']:
        raise RuntimeError('prefix changed after acceptance')
    if not torch.equal(value['transform'],torch.eye(16,device=device)):raise RuntimeError('non-identity B2 prefix transform')
    native=build(config['reference'],device,node['seed']);native.load_state_dict(value['student']);del value
    model=Model(NativeLRParent(native,node['seed'],sid),'B2_PARENT_PAS_KL').to(device)
    provider=NativeCurrentDomain(config['data'],node['seed'],node['order'],2,sid,device,permit,allow_u=not smoke)
    return KeyAlignmentTrainer(model,provider,opts,node['arm'],permit,node=node)


def target(node,config,plan,permit,root,device,prefix_path,predecessor):
    import torch
    from ..five_frameworks_v1.native_runner import Counter,evaluate
    from ..five_frameworks_v1.native_data import NativeCurrentDomain,ORDERS
    from ..five_frameworks_v1.semantics import tensor_fingerprint
    from ..five_frameworks_v1.checkpoint import atomic_save
    from . import state
    nr=root/node['id'];nr.mkdir(exist_ok=True);ident=identity(node,config)
    counter=Counter(nr/'physical.jsonl',node['updates'])
    if (nr/'latest.pt').exists():
        provider=NativeCurrentDomain(config['data'],node['seed'],node['order'],2,source_identity(node,predecessor),device,permit)
        trainer=state.resume(nr/'latest.pt',config['reference'],device,provider,plan['options'][node['domain']],ident,permit,node['arm'],node)
    else:trainer=construct(node,config,plan,permit,prefix_path,predecessor,device)
    if trainer.step!=counter.count:raise RuntimeError('physical/committed cursor mismatch')
    counter.wrap(trainer.optimizer)
    for _ in range(trainer.step,node['updates']):
        trainer.update()
        if trainer.step==1 or trainer.step%trainer.provider.steps_per_epoch==0 or trainer.step==node['updates'] or str(trainer.step) in trainer.diagnostics:
            state.save(trainer,nr/'latest.pt',ident)
            write(nr/'DIAGNOSTICS.json',trainer.diagnostics)
            write(nr/'progress.json',dict(step=trainer.step,physical_calls=counter.count,loss=trainer.last['labeled_loss']))
    expected={str(x) for x in execution_plan()['diagnostics'][node['domain']]}
    if set(trainer.diagnostics)!=expected:raise RuntimeError('incomplete committed diagnostics')
    diagnostics=copy.deepcopy(trainer.diagnostics);support=copy.deepcopy(trainer.support_summary)
    cost=copy.deepcopy(trainer.alignment_cost);telemetry=copy.deepcopy(trainer.telemetry)
    model=trainer.model;del trainer
    deployed=model.deploy();del model
    seen=list(ORDERS[node['order']-1]);scores,private=evaluate(deployed,config['data'],seen,device)
    write(nr/'private_val.json',private)
    atomic_save(dict(identity=ident,student=deployed.parent.native.state_dict(),transform=deployed.transform.detach(),step=node['updates']),nr/'student.pt')
    receipt=dict(node_id=node['id'],identity=ident,status='SEALED',step=node['updates'],physical_optimizer_calls=counter.count,
        student_hash=tensor_fingerprint(deployed.parent.native.state_dict()),transform_hash=tensor_fingerprint({'F':deployed.transform}),
        scores=scores,prefix_scores=predecessor['scores'],prefix_timeline=predecessor['timeline'],
        resolved_options=plan['options'][node['domain']],diagnostics=diagnostics,support=support,alignment_cost=cost,telemetry=telemetry)
    write(nr/'receipt.json',receipt)


def require_qualification(root,config,plan):
    from .qualification import validate_receipt
    for name,mode in [('CUDA_QUALIFICATION','CUDA'),('SMOKE','smoke')]:
        r=read(root/(name+'.json'));validate_receipt(r,mode,config,plan,root)
        session_totals(root/'costs'/name)


def run(config):
    plan,permit=preflight(config);choose_gpu()
    import torch
    from ..single_teacher_scd_v0_1.engine import precision
    precision();device=torch.device('cuda:0');torch.cuda.set_device(device)
    if Path(config['run_root']).resolve()==Path(config['prefix_root']).resolve():raise PermissionError('historical root protected')
    with owned_root(config,plan) as root:
        bind_environment(root,environment());require_qualification(root,config,plan)
        for node in plan['nodes']:
            check_costs(root,plan);status=stage_state(node,root,config);nr=root/node['id']
            if status=='METADATA_SEALED':accept_target(node,config,plan,permit,root,device);continue
            try:
                # Closed sessions are retained; STARTED sessions block new work.
                if list((nr/'costs').glob('*/session.json')):session_totals(nr/'costs')
                path,receipt=accept_prefix(node,config,plan,permit,root,device)
                with cost_session(nr/'costs','formal_target'):
                    target(node,config,plan,permit,root,device,path,receipt)
                accept_target(node,config,plan,permit,root,device)
            except BaseException as e:
                write(nr/'failure.json',dict(status='ENGINEERING_STOP',error=repr(e)));raise
        from .reporting import report
        return report(root,config)

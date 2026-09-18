"""Predeclared 28-call CPU generated suite; all failures remain in the ledger."""
import copy
import csv
import os
import platform
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
import torch
from .protocol import ROOT,DOC,STUDY,ARMS,read,write,digest,canonical_plan,validate_plan,execution_plan,code_manifest
from .authority import synthetic_capability,authorize,baseline_environment
from .qualification import generated_trainer,continuation,baseline_equivalence,failures,snapshot,exact,generated_identity
from ..five_frameworks_v1.native_runner import Counter
from ..f5_confirmation_v1.assurance import cost_session,session_totals,schema,verify_file,integrity_current


def refuse(fn):
    try:fn()
    except (ValueError,PermissionError,RuntimeError,KeyError,FileNotFoundError):return
    raise AssertionError('invalid operation was accepted')


def math_metadata():
    from .core import brier,weights,selector,parent_loss
    from .p0 import opportunities,count
    plan=canonical_plan();assert len(plan['nodes'])==10 and sum(n['updates'] for n in plan['nodes'])==26500
    assert len(plan['imports'])==2 and all(n['seed']==163 and n['stage']==2 for n in plan['nodes'])
    for key,value in [('arm','A0'),('seed',164),('domain','REFUGE'),('updates',2099),('prefix_node','NKA')]:
        bad=copy.deepcopy(plan);bad['nodes'][0][key]=value;bad['plan_sha256']=digest({k:v for k,v in bad.items() if k!='plan_sha256'})
        refuse(lambda:validate_plan(bad))
    for key,value in [('lambda_H',.6),('lambda_DS',.3)]:
        bad=copy.deepcopy(plan);bad['method'][key]=value;bad['plan_sha256']=digest({k:v for k,v in bad.items() if k!='plan_sha256'});refuse(lambda:validate_plan(bad))
    refuse(lambda:authorize({}, {}, {}));refuse(lambda:baseline_environment({}))
    q=torch.tensor([.04,.48,.48]).view(1,3,1,1).expand(2,3,4,6).clone().requires_grad_()
    risk=torch.tensor([.1,.25,.9],requires_grad=True);g=torch.ones(2,4,6,dtype=torch.bool);fine=torch.zeros_like(g)
    m,s=selector([q],g,fine,'A1',risk);assert m.all() and not m.requires_grad
    assert opportunities(q,g).all()
    fine[:,0,0]=True;g[:,1,1]=False;m,_=selector([q],g,fine,'A1',risk);assert not m[fine].any() and not m[~g].any()
    logp=torch.randn(2,3,4,6,requires_grad=True).log_softmax(1);loss=parent_loss(logp,m,g)
    expected=(-torch.logsumexp(logp[:,1:3],dim=1)*m).sum((1,2))/g.sum((1,2))
    assert torch.allclose(loss,expected.mean())
    grads=torch.autograd.grad(loss,(logp,q,risk),allow_unused=True,retain_graph=True);assert grads[1] is None and grads[2] is None
    z=parent_loss(logp,m&False,g&False);gz,=torch.autograd.grad(z,logp);assert z==0 and not gz.any()
    extreme=torch.tensor([0.,-1000.,-1001.]).view(1,3,1,1).log_softmax(1).requires_grad_()
    assert torch.isfinite(parent_loss(extreme,torch.ones(1,1,1,dtype=torch.bool),torch.ones(1,1,1,dtype=torch.bool)))
    a=weights(risk,True);assert not a.requires_grad and abs(float(a.sum())-1)<1e-6 and a.min()>=.1
    labels=torch.zeros(2,4,6,dtype=torch.long);labels[0,0,0]=2;labels[1]=255
    r=brier([q,q,q],labels);assert r is not None and not r.requires_grad
    assert torch.allclose(r,torch.full((3,),(.96**2+.04**2)/2),atol=1e-6)
    assert brier([q],labels*0+255) is None
    assert torch.equal(weights(risk,True),a)  # pending observation did not mutate committed risk
    for arm in ('A4','A5'):
        m,_=selector([q,q,q],g,fine,arm,risk);assert m.sum()==g.sum()-2
        low=q.detach().clone();low[:,0]=.8;low[:,1:]=.1
        m,_=selector([q,low,q],g,fine,arm,torch.full((3,),.25));assert not m.any()
    return dict(parent_example=True,geometry_ignore_empty=True,detached_risk_and_targets=True,alpha_floor=True,
                balanced_missing_class=True,canonical_rehash_rejections=7,old_authority_rejected=True)


def geometry_checks():
    from .model import AGMSModel
    from ..five_frameworks_v1.kernels import collect_sources
    from ..five_frameworks_v1.recipes import complementary,generator
    # Auxiliary grid interpolation contract against explicit bilinear reference,
    # including impulses/checkerboard/borders and an ignore hole in supervision.
    from torch.nn import functional as F
    fields=[torch.zeros(1,3,4,4),torch.arange(16).reshape(1,1,4,4).remainder(2).float().expand(1,3,4,4).clone()]
    fields[0][:,:,0,0]=1;fields[0][:,:,-1,-1]=2
    mask=torch.zeros(1,8,8,dtype=torch.bool);mask[:,2:6,2:6]=True
    for field in fields:
        a=F.interpolate(field,size=(8,8),mode='bilinear',align_corners=True).log_softmax(1)
        b=F.interpolate(field.flip(-1),size=(8,8),mode='bilinear',align_corners=True).log_softmax(1)
        gathered=collect_sources(a,b,mask)[0]
        assert torch.equal(gathered,torch.where(mask[:,None],a,b))
    x=torch.arange(64).reshape(1,1,8,8).float();y=x+100
    u,v,m=complementary(x,y,generator(163,1,2,1,'LCTX'),noise=0)
    assert torch.equal(collect_sources(u,v,m)[0],x)
    from types import SimpleNamespace
    with torch.random.fork_rng(devices=[]):
        heads=torch.nn.ModuleList([torch.nn.Conv2d(c,3,1) for c in (64,32)])
    for head in heads:
        with torch.no_grad():
            head.weight.zero_();head.bias.zero_()
            for c in range(3):head.weight[c,c,0,0]=1
    fixture=SimpleNamespace(aux=heads)
    for field in fields:
        features={name:torch.cat([field,torch.zeros(1,c-3,4,4)],1) for name,c in [('dec3',64),('dec2',32)]}
        actual=AGMSModel.read_aux(fixture,features,(8,8))
        expected=F.interpolate(field,size=(8,8),mode='bilinear',align_corners=True).log_softmax(1)
        for output in actual:assert torch.equal(output,expected)
    return dict(same_complementary_mask=True,aux_no_valid_crop=True,impulse_checkerboard_border=True,actual_aux_readout=True)


def native_arm_checks(t,counter):
    from .protocol import ARMS
    from .core import weights
    heads=list(t.model.aux.parameters());ema=list(t.ema.aux.parameters())
    before=[x.clone() for x in heads];eb=[x.clone() for x in ema]
    assert sum(p.numel() for p in heads)==(294 if ARMS[t.arm]['M'] else 0)
    counter.wrap(t.optimizer);t.update();assert t.provider.u_reads==0
    if heads:
        assert any(not torch.equal(x,y) for x,y in zip(before,heads))
        for a,b,c in zip(ema,eb,heads):assert torch.equal(a,b.mul(.99).add(c,alpha=.01))
    previous=t.risk.clone();grads=t.update();assert t.last['active_U']
    if t.arm!='A0':
        assert t.last['risk']==previous.tolist()
        assert all(not (p.requires_grad) for p in t.ema.parameters())
        assert t.extra_cost['extra_backbone_forwards']==0
        if heads:
            assert t.risk_updates==1
            assert torch.allclose(torch.tensor(t.last['alpha']),weights(previous,t.arm=='A5').cpu())
            assert all(grads[id(p)]['U'] is None for p in heads)
            diag=t.diagnostics['2']['gradients']['DS']['groups']
            assert diag['aux_heads']['raw_norm']>0 and diag['aux_upstream_AB']['raw_norm']>0
    assert not any(p.requires_grad for p in t.model.parent.native.decoder.conv_logit.parameters())
    for adapter in t.model.parent.adapters:
        v=getattr(t.model.parent,'V64_'+str(adapter.index))
        delta=adapter.layer.delta().double()
        assert float((delta@v).norm())<=1e-5*(float(delta.norm())+1e-12)
    x,_,_=t.provider.labeled(0)
    with torch.no_grad():before_deploy=t.model(x)
    deployed=t.model.deploy()
    with torch.no_grad():after_deploy=deployed(x)
    assert torch.allclose(before_deploy,after_deploy,atol=2e-6,rtol=1e-5)
    assert not any('aux' in name for name in deployed.state_dict())
    return dict(arm=t.arm,heads=len(heads),updates=2,EMA=True,U_whitelist=True,A_only=True,deployment=True,
                teacher_backbone_forwards=t.telemetry['teacher_full_forwards'],student_backbone_forwards=t.telemetry['student_full_forwards'])


def integrity_reports(root,t):
    from ..five_frameworks_v1.checkpoint import atomic_save
    from ..five_frameworks_v1.semantics import tensor_fingerprint
    from .reporting import historical,summarize,export
    from .execution import stage_state,identity
    deployed=t.model.deploy();state=deployed.parent.native.state_dict();ident={**generated_identity(t),'node_id':'GENERATED_INTEGRITY'}
    path=root/'generated_student.pt';transform=deployed.transform.detach()
    atomic_save(dict(identity=ident,student=state,transform=transform,step=2),path)
    receipt=dict(node_id=ident['node_id'],identity=ident,step=2,student_hash=tensor_fingerprint(state),transform_hash=tensor_fingerprint({'F':transform}))
    proof=verify_file(path,receipt,schema(state),{'node_id':ident['node_id']})
    assert proof['status']=='VERIFIED' and integrity_current(path,receipt,proof,{'node_id':ident['node_id']})
    bad=copy.deepcopy(receipt);bad['student_hash']='0'*64;refuse(lambda:verify_file(path,bad,schema(state),{'node_id':ident['node_id']}))
    plan=canonical_plan();node=plan['nodes'][0];config={'execution_commit':'generated'}
    nr=root/'tail'/node['id'];nr.mkdir(parents=True);(nr/'physical.jsonl').write_text('{"invocation":1,"scientific_step_attempt":1}\n')
    refuse(lambda:stage_state(node,root/'tail',config))
    rows=historical(plan)
    assert abs(sum(__import__('experiments.lcrseg.agms_cl_v0_1.reporting',fromlist=['metrics']).metrics(r)['Final'] for r in rows)/2-.657465896603577)<1e-12
    assert summarize(rows)['status']=='NOT_ASSESSED_REDUCED_SCOPE'
    for n in plan['nodes']:
        base=copy.deepcopy(next(r for r in rows[:2] if r['identity']['order']==n['order']))
        base.update(arm=n['arm'],node_id=n['id'],identity=identity(n,config),origin='new_execution',support={},extra_cost={'diagnostic_vjps':16})
        sample=copy.deepcopy(t.diagnostics['2'])
        base['diagnostics']={str(step):sample for step in execution_plan()['diagnostics'][n['domain']]}
        rows.append(base)
    out=summarize(rows);assert len(out['pairs'])==27 and out['gates']=={'P_perf':False,'P_joint':False}
    exported=root/'generated_reports';export(exported,{**out,'rows':rows,'status':'GENERATED_TEST_ONLY','costs':{}})
    counts={name:len(list(csv.DictReader((exported/name).open()))) for name in ('FINAL_METRICS.csv','DOMAIN_METRICS.csv','PAIRED_COMPARISONS.csv')}
    assert list(counts.values())==[12,36,27] and len(read(exported/'GRADIENT_DIAGNOSTICS.json'))==40
    return dict(actual_generated_file_verified=True,wrong_hash_refused=True,tail_replay_refused=True,report_counts=counts,diagnostics=40,reduced_gate_not_assessed=True)


def public_record(value, root):
    """Normalize only private path spellings; preserve every historical result/call."""
    if isinstance(value, dict):return {k:public_record(v,root) for k,v in value.items()}
    if isinstance(value, list):return [public_record(v,root) for v in value]
    if isinstance(value, str):
        for path,label in ((str(ROOT),'<CHECKOUT>'),(str(root),'<CPU_EVIDENCE>'),
                           (str(Path('/tmp')/root.name),'<CPU_EVIDENCE>'),(sys.prefix,'<CPU_RUNTIME>')):
            value=value.replace(path,label)
    return value


def run_tests(reference,evidence_root):
    if sys.flags.optimize or 'PYTHONOPTIMIZE' in os.environ or torch.cuda.is_initialized():raise RuntimeError('unoptimized CPU required')
    root=Path(evidence_root).resolve();root.mkdir(parents=True,exist_ok=True)
    if root==ROOT or ROOT in root.parents:raise RuntimeError('generated payloads must remain outside checkout')
    attempts=read(root/'ATTEMPTS.json') if (root/'ATTEMPTS.json').exists() else []
    published=read(DOC/'CPU/ATTEMPTS.json') if (DOC/'CPU/ATTEMPTS.json').exists() else []
    if len(attempts)<len(published) or public_record(attempts[:len(published)],root)!=published:raise RuntimeError('CPU history reset refused')
    if len(attempts)>=execution_plan()['CPU']['attempt_cap'] or any(a['status']=='STARTED' for a in attempts):raise RuntimeError('CPU attempt cap/interrupted attempt')
    class AttemptCounter(Counter):
        def call(self,step):
            if self.count-self.begin>=28:raise RuntimeError('declared attempt optimizer cap')
            return super().call(step)
    counter=AttemptCounter(root/'PHYSICAL.jsonl',96);begin=counter.count;counter.begin=begin;spec=execution_plan()['CPU']
    if begin!=sum(a['calls'] for a in attempts) or begin+28>96:raise RuntimeError('CPU ledger/cap mismatch')
    current=dict(invocation=len(attempts)+1,status='STARTED',cases=spec['cases'],planned_calls=28,calls=0,started=time.time())
    attempts.append(current);write(root/'ATTEMPTS.json',attempts);write(DOC/'CPU/ATTEMPTS.json',public_record(attempts,root))
    payload_root=root/'generated'/('attempt_'+str(len(attempts)))
    torch.set_num_threads(2);permit=synthetic_capability();device=torch.device('cpu');evidence={};baseline=False
    try:
        payload_root.mkdir(parents=True,exist_ok=False)
        with cost_session(root/'costs','CPU_GENERATED',cuda=False):
            evidence['math_metadata']=math_metadata();evidence['geometry']=geometry_checks()
            from .execution import accept_prefix,construct
            plan=canonical_plan()
            refuse(lambda:accept_prefix(plan['nodes'][0],{},plan,permit,root,device))
            refuse(lambda:construct(plan['nodes'][0],{},plan,permit,Path('/unopened'),{},device))
            evidence['arms']=[];initial=None;base_rng=None
            for arm in ARMS:
                torch.manual_seed(381);random.seed(381)
                before=torch.get_rng_state()
                t=generated_trainer(reference,arm,device,permit)
                if arm=='A0':base_rng=torch.get_rng_state().clone()
                else:assert torch.equal(base_rng,torch.get_rng_state())
                if arm in ('A2','A3','A4','A5'):
                    init=dict(main=copy.deepcopy(t.model.parent.state_dict()),aux=copy.deepcopy(t.model.aux.state_dict()),rng=torch.get_rng_state())
                    if initial is None:initial=init
                    else:exact(initial,init)
                evidence['arms'].append(native_arm_checks(t,counter));del t
            evidence['continuation']=[]
            from .state import validate_payload,metadata_path
            for arm in ('A1','A5'):
                t,path=continuation(reference,arm,device,permit,counter,payload_root,32)
                value=torch.load(path,map_location='cpu',weights_only=False);meta=read(metadata_path(path))
                for field in ('arm','prefix','temperature','geometry','aux_schema','risk','identity'):
                    bad=copy.deepcopy(value)
                    if field=='arm':bad['agms']['arm']='A0'
                    elif field=='prefix':bad['agms']['prefix']={}
                    elif field=='temperature':bad['agms']['method']['risk']['temperature']=.2
                    elif field=='geometry':bad['agms']['geometry']={}
                    elif field=='aux_schema':bad['agms']['aux_schema']={"wrong":[]}
                    elif field=='identity':bad['identity']['order']=2
                    else:bad['risk'][0]+=.1
                    refuse(lambda:validate_payload(bad,meta,generated_identity(t),arm,t.options,t.provider))
                evidence['continuation'].append(dict(arm=arm,exact=True,negative_restore_cases=7))
                if arm=='A5':evidence['integrity_reports']=integrity_reports(payload_root,t)
                del t
            evidence['baseline']=baseline_equivalence(reference,device,permit,counter,32);baseline=True
            failures(reference,device,permit,counter,32);evidence['failures']=dict(after_optimizer=2,risk_diagnostics_not_committed=True)
        assert counter.count-begin==28
        current['status']='PASS'
    except BaseException as e:
        current.update(status='FAIL',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc().replace(str(ROOT),'<CHECKOUT>'))
        raise
    finally:
        current.update(calls=counter.count-begin,seconds=time.time()-current['started'])
        write(root/'ATTEMPTS.json',attempts)
        report=dict(status=current['status'],study_id=STUDY,baseline_equivalence=baseline,evidence=evidence,
                    code_tree_sha256=code_manifest()['code_tree_sha256'],plan_sha256=canonical_plan()['plan_sha256'],
                    execution_sha256=digest(execution_plan()),cost={'old_CPU':134,'new_CPU':counter.count,'attempts':len(attempts),
                        'real_optimizer':0,'CUDA':0,'P0':0,'smoke':0,'formal':0},
                    environment={'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','optimize':sys.flags.optimize},
                    future={k:'PENDING' for k in ('prefix_tensors','P0','CUDA','smoke','P1')})
        write(root/f'ATTEMPT_{len(attempts)}_REPORT.json',report);write(root/'TEST_REPORT.json',report)
        out=DOC/'CPU';out.mkdir(exist_ok=True)
        for p in root.glob('*.json'):
            if not (out/p.name).exists() or p.name in ('ATTEMPTS.json','TEST_REPORT.json'):
                write(out/p.name,public_record(read(p),root))
        if (root/'PHYSICAL.jsonl').exists():shutil.copyfile(root/'PHYSICAL.jsonl',out/'PHYSICAL.jsonl')
        # Publish aggregate cost records only, never generated model payloads.
        for p in (root/'costs').glob('*/session.json'):
            write(out/'costs'/p.parent.name/'session.json',read(p))
        write(DOC/'COST_SUMMARY.json',report['cost'])
    return report

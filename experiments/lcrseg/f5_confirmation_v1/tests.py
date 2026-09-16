"""Finite CPU-only synthetic checks. No production permit is ever constructed."""
import copy
import json
import math
import os
import tempfile
import time
import traceback
from pathlib import Path
from . import protocol as p

# Fixed collection, bounded repeated invocations including failed attempts.
TEST_NAMES=('matrix_and_imports','reject_authority','receipt_binding','options_and_gradients',
            'loss_nesting','merge_current_ema_resume','random_stream_logging',
            'failure_budget_sealed','metric_pairs_gates','integrity_faults','qualification_coverage','cumulative_costs_rng','report_artifacts')
CPU_PHYSICAL_CAP=40
MAX_INVOCATIONS=8


def rejects(fn,kind=Exception):
    try:fn()
    except kind:return
    raise AssertionError('expected rejection')


def run_tests():
    import torch
    from ..five_frameworks_v1.native_runner import Counter,options_for
    from ..five_frameworks_v1.parent_bridge import SyntheticParentBridge
    from ..five_frameworks_v1.recipes import SyntheticCurrentDomain,generator
    from ..five_frameworks_v1.model import Model
    from ..five_frameworks_v1.train_stage import StageTrainer,split_gradients
    from ..five_frameworks_v1 import checkpoint
    from ..five_frameworks_v1.semantics import tensor_fingerprint
    from .execution import authorization,stage_state,ledger_count
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():raise RuntimeError('CPU preparation cannot initialize CUDA')
    plan=p.validate_plan(p.read(p.DOC/'PLAN.json'))
    history=p.DOC/'CPU_ATTEMPTS.json'
    attempts=p.read(history) if history.exists() else []
    if len(attempts)>=MAX_INVOCATIONS:raise RuntimeError('bounded CPU invocation cap reached')
    attempt={'invocation':len(attempts)+1,'status':'STARTED','tests':list(TEST_NAMES),'started_unix':time.time()}
    attempts.append(attempt);p.write(history,attempts)
    counter=Counter(p.DOC/'CPU_PHYSICAL.jsonl',CPU_PHYSICAL_CAP)
    before=counter.count

    def trainer(family,options=None,stage=1,parent=None,previous=None):
        torch.manual_seed(7)
        parent=parent or SyntheticParentBridge(d=16,rank=2)
        opts={**plan['options'][family]['RIM_ONE_r3'], 'total_steps':4, 'warmup_fraction':.25,'U_ramp_fraction':.25,**(options or {})}
        t=StageTrainer(Model(parent,family,ratio=opts.get('rank_ratio',.5),previous=previous),
                       SyntheticCurrentDomain(seed=163,stage=stage),opts)
        counter.wrap(t.optimizer)
        return t

    def matrix_and_imports():
        assert len(plan['nodes'])==28 and sum(n['updates'] for n in plan['nodes'])==74200
        assert len({n['sequence_id'] for n in plan['nodes']})==14
        assert len(plan['historical_target_imports'])==8 and len(plan['reused_sources'])==3
        assert all(n['kind']=='target' and n['phase']=='P1' and n['seed']!=161 for n in plan['nodes'])
        for bad in ('source','old_DAG','F1','seed161','extra'):
            q=copy.deepcopy(plan)
            if bad=='extra':q['nodes'].append(copy.deepcopy(q['nodes'][0]))
            elif bad=='seed161':q['nodes'][0]['seed']=161
            elif bad=='F1':q['nodes'][0]['family']='F1'
            else:q['nodes'][0]['kind']=bad
            rejects(lambda:p.validate_plan(q),ValueError)
        for n in plan['nodes']:
            if n['stage']==2:
                prev=next(x for x in plan['nodes'] if x['id']==n['parent_checkpoint'])
                checkpoint.require_predecessor(n,prev)
                bad={**prev,'family':'F1'};rejects(lambda:checkpoint.require_predecessor(n,bad),ValueError)

    def reject_authority():
        rejects(lambda:authorization({}, {}, {}),PermissionError)
        # Reject the actual previous round's authority, without generating a fake approval.
        rejects(lambda:authorization(p.read(p.OLD/'USER_AUTHORIZATION.json'),{},{}),PermissionError)
        rejects(lambda:authorization(p.read(p.DOC/'external_review_R1/REVIEW_DECISION.json'),{},{}),PermissionError)
        rejects(lambda:authorization({'study_id':'F5_CONFIRMATION_V1','is_template':True},{},{}),PermissionError)

    def receipt_binding():
        study=p.read(p.OLD/'RESOLVED_PROTOCOL.json')['study']
        for n in plan['nodes']:
            assert options_for(n,study,{'SELECT_PARENT':{'candidate_id':'P1'}})==plan['options'][n['family']][n['domain']]
        e=p.read(p.OLD/'FINAL_RESULTS_REDUCED.json')
        r=next(r for r in e['index'] if r.get('identity',{}).get('candidate_id')=='F5_C02')
        r['resolved_options']['feature_lr_multiplier']=.5
        rejects(lambda:p.bound_options(e),ValueError)
        meta=p.read(p.DOC/'REUSE_METADATA.json');meta['sources'][0]['identity']['seed']=161
        rejects(lambda:p.source_reuse(meta),ValueError)

    def options_and_gradients():
        for f in (p.B0,p.B2,p.F5):
            t=trainer(f);groups={g['name']:g for g in t.optimizer.param_groups}
            assert groups['A']['lr']==groups['B']['lr']==.0005
            assert t.optimizer.defaults['betas']==(.9,.999) and t.optimizer.defaults['eps']==1e-8
            if f==p.F5:
                assert groups['R']['lr']==.001 and t.model.sidecar.r.shape==(4,4)
            else:assert t.model.sidecar is None
            t.step=t.cursor=1  # synthetic active-U branch; not a scientific training result
            l,u,_=t.losses();g=split_gradients(t.model,l,u)
            if f==p.B0:assert u is None and t.provider.u_reads==0
            elif f==p.F5:
                assert all(v['U'] is None for ident,v in g.items() if ident!=id(t.model.sidecar.r))
                assert g[id(t.model.sidecar.r)]['U'] is not None
                assert t.telemetry['extra_clean_U_forwards']==1
            else:
                assert {id(x) for x in t.model.u_parameters()}=={id(x) for x in t.model.parameters() if x.requires_grad}
                assert any(v['U'] is not None for v in g.values())
            assert all(v['L'] is not None for v in g.values())
            assert t.provider.u_reads==(0 if f==p.B0 else 1)

    def loss_nesting():
        # Same initialized model/data/stream. Three existing losses identify the actual
        # SWD contribution without replacing/mocking math, adding a teacher, or using real data.
        def u(weight,swd):
            t=trainer(p.F5,dict(lambda_U=weight,lambda_SWD=swd,PAS_confidence=0.,PAS_cosine=-1.))
            t.step=t.cursor=1
            return t.losses()[1].detach()
        a,b,c=u(.25,.2),u(1.,.2),u(.25,0.)
        assert torch.allclose(a,.25*b,atol=1e-8)
        d=u(1.,1.);e=u(1.,0.)
        assert float((d-e).abs())>1e-9  # ensure SWD support is nonempty
        assert torch.allclose(a-c,.05*(d-e),atol=1e-7)
        assert torch.isfinite(a)

    def merge_current_ema_resume():
        t=trainer(p.F5);t.update();t.update()
        assert t.ema is not t.model and not any(x.requires_grad for x in t.ema.parameters())
        with tempfile.TemporaryDirectory(prefix='synthetic-',dir=os.environ.get('TMPDIR')) as tmp:
            file=Path(tmp)/'state.pt';identity=dict(family=p.F5,seed=163,order=1,stage=1)
            checkpoint.save(t,file,identity)
            saved=torch.load(file,weights_only=False)
            for key in ('student','ema','optimizer','scheduler','prototypes','support','cursor','torch_rng','python_rng','provider_reads'):
                assert key in saved
            t.update();expected=tensor_fingerprint(t.model.state_dict());expected_ema=tensor_fingerprint(t.ema.state_dict())
            t2=trainer(p.F5);checkpoint.restore(t2,file,identity);t2.update()
            assert tensor_fingerprint(t2.model.state_dict())==expected
            assert tensor_fingerprint(t2.ema.state_dict())==expected_ema
            assert t2.cursor==t.cursor and torch.equal(t2.prototypes.values,t.prototypes.values)
            t2.model.eval().requires_grad_(False);t2.model.role='eval'
            x=t2.provider.labeled(4)[0]
            with torch.no_grad():expected_output=t2.model(x);deploy=t2.model.deploy();actual=deploy(x)
            assert torch.allclose(expected_output,actual,atol=2e-6)
            stage2=trainer(p.F5,stage=2,parent=copy.deepcopy(deploy.parent),previous=deploy.transform)
            assert stage2.step==0 and not stage2.optimizer.state
            assert stage2.ema is not t2.ema
            assert torch.equal(stage2.model.sidecar.previous,deploy.transform)
            assert torch.equal(stage2.model.sidecar.r,torch.zeros(4,4))
            rejects(lambda:checkpoint.restore(stage2,file,{**identity,'stage':2}),ValueError)

    def random_stream_logging():
        t=trainer(p.F5)
        first=torch.rand(12,generator=t.rng('UL'))
        state=torch.get_rng_state().clone()
        # Production run_id/output never enter the shared stateless key; detached receipt logging only.
        for run_id in ('old','new'):
            json.dumps({'run_id':run_id,'last':copy.deepcopy(t.last)})
        assert torch.equal(state,torch.get_rng_state())
        assert torch.equal(first,torch.rand(12,generator=generator(163,1,1,0,'UL')))
        t2=trainer(p.F5,{'output_path':'different','log_every':999})
        assert torch.equal(t.rng('UL').get_state(),t2.rng('UL').get_state())
        x=t.provider.labeled(0)[0];y=t2.provider.labeled(0)[0];assert torch.equal(x,y)

    def failure_budget_sealed():
        t=trainer(p.B0);start=counter.count
        rejects(lambda:t.update(fault='after_optimizer'),RuntimeError)
        assert counter.count==start+1 and t.step==0
        rejects(t.update,RuntimeError);assert counter.count==start+1
        with tempfile.TemporaryDirectory(prefix='ledger-',dir=os.environ.get('TMPDIR')) as tmp:
            root=Path(tmp);n=copy.deepcopy(plan['nodes'][0]);n['updates']=1;nr=root/n['id'];nr.mkdir()
            c=Counter(nr/'physical.jsonl',1);c.call(1)
            rejects(lambda:c.call(2),RuntimeError);assert c.count==1
            rejects(lambda:stage_state(n,root,'fixture'),RuntimeError)
            p.write(nr/'latest.pt.receipt.json',{'committed':True,'step':0});(nr/'latest.pt').write_text('synthetic metadata only')
            rejects(lambda:stage_state(n,root,'fixture'),RuntimeError)
            p.write(nr/'latest.pt.receipt.json',{'committed':True,'step':1})
            assert stage_state(n,root,'fixture')=='RESUME'
            (nr/'student.pt').write_text('synthetic only')
            identity={k:n[k] for k in ('family','candidate_id','seed','order','stage','sequence_id','domain')}
            identity.update(node_id=n['id'],execution_commit='fixture')
            p.write(nr/'receipt.json',dict(identity=identity,node_id=n['id'],status='SEALED',step=1,physical_optimizer_calls=1))
            assert stage_state(n,root,'fixture')=='METADATA_SEALED'
            assert stage_state(n,root,'fixture')=='METADATA_SEALED' and ledger_count(nr/'physical.jsonl')==1

    def metric_pairs_gates():
        rows=[]
        def scores(domains,values):return {d:dict(rim=v-.01,cup=v+.01,disc_union=v+.1,macro_Dice=v) for d,v in zip(domains,values)}
        for seed in (162,163,164):
            for order,domains in enumerate(p.ORDERS,1):
                for f in p.CANDIDATES:
                    bonus=.01 if f==p.F5 else 0.
                    r=dict(identity=dict(family=f,seed=seed,order=order,stage=2),scores=scores(domains,[.6+bonus,.7+bonus,.8+bonus]),
                           timeline={'0':scores(domains[:1],[.9]),'1':scores(domains[:2],[.8,.85])})
                    m=p.metrics(r)
                    assert math.isclose(m['Final'],.7+bonus) and math.isclose(m['Old'],.65+bonus)
                    assert math.isclose(m['Incoming'],.8+bonus) and math.isclose(m['Forget'],.225-bonus)
                    rows.append(r)
        a=p.paired(rows,[163,164]);g=p.decisions(a)
        assert all(g[k] for k in ('G1','G2','G3')) and g['recommendation']=='P2_REVIEW_ONLY'
        rejects(lambda:p.paired(rows[:-1],[163,164]),ValueError)
        bad=copy.deepcopy(a);bad[p.B0]['per_seed']['163']['Final']=-.01
        assert not p.decisions(bad)['G1']
        bad=copy.deepcopy(a);bad[p.B0]['per_order']['163/O2']['Old']=-.1
        assert not p.decisions(bad)['G2']
        assert len(p.paired(rows,[162,163,164])[p.B0]['per_seed'])==3

    from .review_tests import integrity_faults,qualification_coverage,cumulative_costs_rng,report_artifacts
    checks=locals();results=[]
    for name in TEST_NAMES:
        try:checks[name]();results.append({'test':name,'status':'PASS'})
        except Exception:
            results.append({'test':name,'status':'FAIL','traceback':traceback.format_exc().replace(str(p.ROOT), '<repo>')})
    attempt.update(status='PASS' if all(r['status']=='PASS' for r in results) else 'FAIL',results=results,
                   physical_optimizer_calls=counter.count-before,cumulative_physical_calls=counter.count)
    p.write(history,attempts)
    report=dict(status=attempt['status'],tests=results,invocations=len(attempts),
                cpu_physical_calls_cumulative=counter.count,cpu_physical_cap=CPU_PHYSICAL_CAP,max_invocations=MAX_INVOCATIONS,
                python=__import__('platform').python_version(),torch=torch.__version__,device='cpu',
                plan_sha256=plan['plan_sha256'],code_tree_sha256=p.manifest()['code_tree_sha256'],
                patient_payload_reads=0,real_checkpoint_tensor_reads=0,real_optimizer_updates=0,cuda_optimizer_calls=0,
                limitations='CPU synthetic bridges/tensor files/metadata only; native CUDA/source tensor/current-L smoke PENDING; not external approval')
    p.write(p.DOC/'TEST_REPORT.json',report)
    if attempt['status']!='PASS':raise RuntimeError('CPU synthetic checks failed; see TEST_REPORT.json')
    return report

"""Two create-only attempts maximum; 12 predeclared calls each, cumulative <=32.

Zero-update math/authority/report tests precede native generated updates. No
historical suite, real weights, patients or CUDA is read by this entry point.
"""
import copy
import csv
import json
import os
import sys
import time
from pathlib import Path
from .protocol import ROOT,DOC,STUDY,CAPS,read,write,digest,canonical_plan,execution_plan,code_manifest
CHECKS={'math','authority','reports','native','continuation','identity','shadow','failure','integrity'}


def refuse(fn):
    try:fn()
    except (ValueError,PermissionError,RuntimeError):return
    raise AssertionError('invalid operation accepted')


def validate_cpu(plan,tree):
    r=read(DOC/'CPU/TEST_REPORT.json');attempts=read(DOC/'CPU/ATTEMPTS.json')
    if not 1<=len(attempts)<=2 or sum(a['calls'] for a in attempts)>32:raise PermissionError('CPU budget')
    total=0
    for i,a in enumerate(attempts,1):
        lines=(DOC/'CPU'/f'attempt_{i}'/'PHYSICAL.jsonl').read_text().splitlines()
        if [json.loads(x)['invocation'] for x in lines]!=list(range(1,a['calls']+1)) or not 0<=a['calls']<=a['planned_calls']<=16:
            raise PermissionError('CPU ledger/range')
        if a['status'] not in ('PASS','FAIL'):raise PermissionError('unclosed CPU attempt')
        total+=a['calls']
    if (r['status']!='PASS' or attempts[-1]['status']!='PASS' or attempts[-1]['calls']!=12
        or r['study_id']!=STUDY or r['code_tree_sha256']!=tree or attempts[-1]['code_tree_sha256']!=tree
        or r['plan_sha256']!=plan['plan_sha256'] or r['execution_sha256']!=digest(execution_plan())
        or set(r['checks'])!=CHECKS or not all(r['checks'].values()) or r['cost']['new_CPU']!=total
        or r['cost']['real_optimizer']!=0 or r['cost']['prior_CPU']!=235):raise PermissionError('fresh observer CPU evidence required')


def math_checks():
    import torch
    from .diagnostics import kl_parts,selection_counts,selection_ratios,labeled_diagnostics
    from ..five_frameworks_v1.kernels import masked_kl
    # Independent local generator: diagnostic math must never consume training RNG.
    gen=torch.Generator().manual_seed(777)
    for scale in (1.,1000.):
        logits=torch.randn(2,3,7,7,generator=gen,dtype=torch.float64)*scale
        q=(torch.randn(2,3,7,7,generator=gen,dtype=torch.float64)*scale).softmax(1)
        for mask in (torch.ones(2,7,7,dtype=torch.bool),torch.zeros(2,7,7,dtype=torch.bool),torch.arange(98).reshape(2,7,7)%3==0):
            a,b=kl_parts(logits,q,mask)
            assert torch.isfinite(a+b) and torch.allclose(a+b,masked_kl(logits,q,mask),atol=1e-10,rtol=1e-10)
    q=torch.zeros(1,3,5,5);q[:,0]=.05;q[:,1]=.75;q[:,2]=.2
    other=q.clone();other[:,0]=.5;other[:,1]=.3
    qs=[q,other,q];g=torch.ones(1,5,5,dtype=torch.bool);fine=g&False;risk=torch.tensor([.1,.8,.1])
    labels=torch.arange(25).reshape(1,5,5)%3;labels[:,0,0]=255
    rng=torch.get_rng_state().clone();before=[p.clone() for p in qs]
    a=selection_counts(qs,g,fine,risk);b=labeled_diagnostics(qs,labels)
    assert a['candidate']==25 and a['main_selected']==25 and a['equal_selected']==0
    assert selection_ratios(a)['main__equal_jaccard']==0
    assert labeled_diagnostics(qs,labels*0+255)['scales'][0]['parent_brier'] is None
    assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(x,y) for x,y in zip(qs,before))
    assert all(x['selected']==__import__('math').ceil(x['eligible']*.25) for x in b['ranking'])
    empty=selection_counts(qs,g&False,fine,risk);assert selection_ratios(empty)['risk_selected_per_geometry'] is None


def report_checks(root,sample=None,support=None):
    from .reporting import historical,summarize,export
    from .execution import identity
    from .protocol import half_results
    p=canonical_plan();rows=historical(p);assert len(rows)==8 and summarize(rows)['gates'] is None
    # Algebra-only fixture explicitly marked generated, never a measured endpoint.
    if sample is None:
        sample=dict(extra_vjps=4,coverage=[],risk=[.25]*3,alpha=[1/3]*3,shadow_L=[],L_errors=[],uniform_risk_disagreement=0,
                    gradients={},L_parent_conditional={},U_same_state={},KL_decomposition={})
    for n in p['nodes']:
        base=copy.deepcopy(next(r for r in rows if r['arm']=='A0' and r['identity']['order']==n['order']))
        base.update(arm='OBS0',node_id=n['id'],identity=identity(n,{'execution_commit':'GENERATED_ONLY'}),origin='new_execution',
            support=support or {'U_same_state':{}},extra_cost={'diagnostic_vjps':16},
            diagnostics={str(k):copy.deepcopy(sample) for k in execution_plan()['diagnostics'][n['domain']]})
        rows.append(base)
    out=summarize(rows);assert out['gates']=={'DESCRIPTIVE_RECOVERY':True}
    export(root,{**out,'rows':rows,'status':'GENERATED_TEST_ONLY','costs':{}})
    for file,count in [('FINAL_METRICS.csv',10),('DOMAIN_METRICS.csv',30),('PAIRED_COMPARISONS.csv',12)]:
        assert len(list(csv.DictReader((root/file).open())))==count
    assert len(read(root/'GRADIENT_DIAGNOSTICS.json'))==8
    # Source gain must never hide a first-target loss, even at unchanged Final/Old.
    bad=copy.deepcopy(rows);obs=next(r for r in bad if r['arm']=='OBS0' and r['identity']['order']==1)
    for key in ('rim','cup','macro_Dice'):
        obs['scores']['REFUGE'][key]+=.003
        obs['scores']['RIM_ONE_r3'][key]-=.003
    assert summarize(bad)['gates']=={'DESCRIPTIVE_RECOVERY':False}
    # Read-only re-evaluation of published aggregates: original negative gate fixed.
    h=half_results();assert h['gates']=={'DS_PRESSURE_SUPPORTED':False}
    ds=[]
    for o in (1,2):
        a=next(r for r in rows if r['arm']=='HALF_CONTROL' and r['identity']['order']==o)
        b=next(r for r in rows if r['arm']=='A5_CONTROL' and r['identity']['order']==o)
        first='RIM_ONE_r3' if o==1 else 'Drishti_GS'
        ds.append(dict(source=a['scores']['REFUGE']['macro_Dice']-b['scores']['REFUGE']['macro_Dice'],
                       first_target=a['scores'][first]['macro_Dice']-b['scores'][first]['macro_Dice']))
    mean={k:sum(r[k] for r in ds)/2 for k in ds[0]}
    assert mean['source']>0 and mean['first_target']<0
    write(root/'HISTORICAL_RECOMPUTATION.json',dict(per_order=ds,mean=mean,HALF_gate_unchanged=h['gates'],new_evaluations=0))


def native_checks(t,counter):
    import torch
    from .diagnostics import labeled_diagnostics,selection_counts
    from ..agms_cl_v0_1.core import weights,parent_loss
    from ..five_frameworks_v1.kernels import masked_kl
    heads=list(t.model.aux.parameters());ema=list(t.ema.aux.parameters())
    before=[p.detach().clone() for p in heads];eb=[p.detach().clone() for p in ema]
    assert sum(p.numel() for p in heads)==294
    counter.wrap(t.optimizer);t.update();assert t.provider.u_reads==0
    assert any(not torch.equal(a,b) for a,b in zip(before,heads))
    for a,b,c in zip(ema,eb,heads):assert torch.equal(a,b.mul(.99).add(c,alpha=.01))
    previous=t.risk.clone();grads=t.update()
    assert t.last['active_U'] and t.risk_updates==1 and t.last['risk']==previous.tolist()
    assert all(not p.requires_grad for p in t.ema.parameters()) and not t.risk.requires_grad
    assert torch.allclose(torch.tensor(t.last['alpha']),weights(previous,True).cpu())
    assert all(grads[id(p)]['U'] is None for p in heads)
    d=t.diagnostics['2']['gradients']['DS']['groups']
    assert d['all_AB']['raw_norm']==0 and d['aux_upstream_AB']['raw_norm']==0 and d['aux_heads']['raw_norm']>0
    assert abs(t.last['losses']['weighted_DS']-.25*t.last['losses']['DS'])<1e-6
    assert t.support_summary['U_same_state']['steps']==1 and t.extra_cost['extra_backbone_forwards']==0
    assert not any(p.requires_grad for p in t.model.parent.native.decoder.conv_logit.parameters())
    # Generated route fixture checks all three original upstream paths, even if a
    # generated teacher rejects every actual coarse pixel in the update above.
    x,y,_=t.provider.labeled(0)
    with t.model.capture() as features:logp=t.model(x).log_softmax(1)
    aux=t.model.read_aux(features,x.shape[-2:])
    for (name,head),a in zip(zip(('dec3','dec2'),t.model.aux),aux):
        expected=torch.nn.functional.interpolate(head(features[name]),size=x.shape[-2:],mode='bilinear',align_corners=True).log_softmax(1)
        assert torch.equal(a,expected)
    q=logp.detach().roll(1,1).exp();g=y!=255
    ab=list(t.model.u_parameters())
    for name,loss in [('L',t.model.parent.supervised(logp,y)),('KL',masked_kl(logp,q,g)),('H',parent_loss(logp,g,g))]:
        gs=torch.autograd.grad(loss,ab,allow_unused=True,retain_graph=True)
        assert any(v is not None and v.norm()>0 for v in gs),name
    ds=torch.stack([t.model.parent.supervised(a,y) for a in aux]).mean()
    gs=torch.autograd.grad(ds,ab+heads,allow_unused=True)
    assert all(v is None or not v.count_nonzero() for v in gs[:len(ab)]) and any(v is not None and v.norm()>0 for v in gs[len(ab):])
    rng=torch.get_rng_state().clone();stored=[None if p.grad is None else p.grad.clone() for p in ab+heads];risk=t.risk.clone()
    labeled_diagnostics([q]*3,y);selection_counts([q]*3,g,g&False,t.risk)
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(risk,t.risk)
    for p,v in zip(ab+heads,stored):assert (p.grad is None and v is None) or (v is not None and torch.equal(p.grad,v))
    for adapter in t.model.parent.adapters:
        v=getattr(t.model.parent,'V64_'+str(adapter.index));delta=adapter.layer.delta().double()
        assert float((delta@v).norm())<=1e-5*(float(delta.norm())+1e-12)
    with torch.no_grad():before=t.model(x)
    deployed=t.model.deploy()
    with torch.no_grad():after=deployed(x)
    assert torch.allclose(before,after,atol=2e-6,rtol=1e-5) and not any('aux' in n for n in deployed.state_dict())


def run_tests(reference,evidence_root):
    import torch
    from .authority import synthetic_capability,authorize
    from .qualification import generated_trainer,continuation,baseline_equivalence,failures,generated_identity
    from .state import validate_payload,metadata_path
    from .execution import accept_prefix,construct,stage_state
    from .protocol import validate_plan
    from ..five_frameworks_v1.native_runner import Counter
    from ..f5_confirmation_v1.assurance import cost_session,session_totals,verify_file,schema,integrity_current
    if sys.flags.optimize or 'PYTHONOPTIMIZE' in os.environ or torch.cuda.is_initialized():raise RuntimeError('unoptimized CPU only')
    root=Path(evidence_root).resolve();nas=Path('/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg')
    if nas not in root.parents or os.environ.get('SSLCL_STORAGE_ROOT')!=str(nas):raise RuntimeError('NAS wrapper required')
    root.mkdir(parents=True,exist_ok=True)
    attempts=read(root/'ATTEMPTS.json') if (root/'ATTEMPTS.json').exists() else []
    public=read(DOC/'CPU/ATTEMPTS.json') if (DOC/'CPU/ATTEMPTS.json').exists() else []
    if attempts!=public or len(attempts)>=2 or any(a['status']=='STARTED' for a in attempts) or (attempts and attempts[-1]['status']=='PASS'):
        raise RuntimeError('attempt history mismatch, consumed or unclosed; no automatic retry')
    spec=execution_plan()['CPU'];used=sum(a['calls'] for a in attempts)
    if used+spec['planned_calls']>32:raise RuntimeError('cumulative CPU cap')
    num=len(attempts)+1;ar=root/f'attempt_{num}';ar.mkdir(exist_ok=False)
    counter=Counter(ar/'PHYSICAL.jsonl',spec['planned_calls'])
    (ar/'PHYSICAL.jsonl').touch(exist_ok=False)
    attempt=dict(status='STARTED',planned_calls=12,calls=0,started=time.time(),code_tree_sha256=code_manifest()['code_tree_sha256'])
    attempts.append(attempt)
    for base in (root,DOC/'CPU'):write(base/'ATTEMPTS.json',attempts)
    torch.set_num_threads(2);device=torch.device('cpu');permit=synthetic_capability();checks={};plan=canonical_plan()
    try:
        with cost_session(ar/'costs','OBSERVER_CPU',cuda=False):
            math_checks();checks['math']=True
            bad=copy.deepcopy(plan);bad['method']['aux_features_detached']=False;bad['plan_sha256']=digest({k:v for k,v in bad.items() if k!='plan_sha256'})
            refuse(lambda:validate_plan(bad));refuse(lambda:authorize({}, {}, {}))
            refuse(lambda:accept_prefix(plan['nodes'][0],{},plan,permit,ar,device))
            refuse(lambda:construct(plan['nodes'][0],{},plan,permit,Path('/never-opened'),{},device))
            checks['authority']=True
            report_checks(ar/'zero_reports');checks['reports']=True
            t=generated_trainer(reference,'OBS0',device,permit);native_checks(t,counter)
            report_checks(ar/'native_reports',t.diagnostics['2'],t.support_summary);checks['native']=True;del t
            t,path=continuation(reference,'OBS0',device,permit,counter,ar,32);checks['continuation']=True
            v=torch.load(path,map_location=device,weights_only=False);meta=read(metadata_path(path))
            for field in ('detach','prefix','study'):
                bad=copy.deepcopy(v);bm=copy.deepcopy(meta)
                if field=='detach':bad['agms']['method']['aux_features_detached']=False
                elif field=='prefix':bad['agms']['prefix']={}
                else:bad['agms']['study_id']='AGMS_DS_HALF_V1'
                bm['agms_sha256']=digest(bad['agms'])
                refuse(lambda:validate_payload(bad,bm,generated_identity(t),'OBS0',t.options,t.provider))
            checks['identity']=True
            nr=ar/'tail'/plan['nodes'][0]['id'];nr.mkdir(parents=True)
            (nr/'physical.jsonl').write_text('{"invocation":1,"scientific_step_attempt":1}\n')
            refuse(lambda:stage_state(plan['nodes'][0],ar/'tail',{'execution_commit':'GENERATED_ONLY'}))
            from ..five_frameworks_v1.checkpoint import atomic_save
            from ..five_frameworks_v1.semantics import tensor_fingerprint
            deployed=t.model.deploy();weights=deployed.parent.native.state_dict();transform=deployed.transform.detach()
            ident=generated_identity(t);payload=ar/'integrity.pt'
            atomic_save(dict(identity=ident,student=weights,transform=transform,step=t.step),payload)
            receipt=dict(node_id=ident['node_id'],identity=ident,step=t.step,student_hash=tensor_fingerprint(weights),transform_hash=tensor_fingerprint({'F':transform}))
            proof=verify_file(payload,receipt,schema(weights),{'node_id':ident['node_id']})
            assert integrity_current(payload,receipt,proof,{'node_id':ident['node_id']})
            bad=copy.deepcopy(receipt);bad['student_hash']='0'*64
            refuse(lambda:verify_file(payload,bad,schema(weights),{'node_id':ident['node_id']}))
            checks['integrity']=True;del t,v,deployed,weights
            baseline_equivalence(reference,device,permit,counter,32);checks['shadow']=True
            failures(reference,device,permit,counter,32);checks['failure']=True
        assert counter.count==12 and set(checks)==CHECKS
        session_totals(ar/'costs');attempt['status']='PASS'
    except BaseException as e:
        attempt.update(status='FAIL',error=repr(e));raise
    finally:
        attempt.update(calls=counter.count,seconds=time.time()-attempt['started'])
        report=dict(status=attempt['status'],study_id=STUDY,checks=checks,plan_sha256=plan['plan_sha256'],
          code_tree_sha256=code_manifest()['code_tree_sha256'],execution_sha256=digest(execution_plan()),
          cost=dict(new_CPU=used+counter.count,attempts=num,prior_CPU=235,prior_breakdown={'earlier':134,'AGMS':80,'HALF':21},real_optimizer=0),
          environment=dict(python=sys.version.split()[0],torch=torch.__version__,device='cpu',optimize=sys.flags.optimize),
          future=dict(CUDA='PENDING_FRESH_REVIEW_AND_USER_LAUNCH',smoke='PENDING',formal='PENDING'))
        for base in (root,DOC/'CPU'):
            write(base/'ATTEMPTS.json',attempts);write(base/'TEST_REPORT.json',report)
        dest=DOC/'CPU'/f'attempt_{num}';dest.mkdir(parents=True,exist_ok=True)
        (dest/'PHYSICAL.jsonl').write_bytes((ar/'PHYSICAL.jsonl').read_bytes())
        for p in (ar/'costs').glob('*/session.json'):write(dest/'costs'/p.parent.name/'session.json',read(p))
    validate_cpu(plan,code_manifest()['code_tree_sha256'])
    return report

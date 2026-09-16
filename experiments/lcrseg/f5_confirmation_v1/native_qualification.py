"""Finite native CUDA definitions. Not executed during code-only repair."""
from .protocol import B0,B2,F5

CASE_CHECKS={
    'warmup':['native_parent','finite_loss','zero_U'],
    'resume':['native_resume','student','ema','optimizer','scheduler','prototypes_support','cursor_reads_rng','U_permissions','nondegenerate_support'],
    'transition':['full_model_merge','own_predecessor','weights_F','new_adapters_zero_R','reset_current_state','Q_origin'],
    'failure':['durable_failed_call','no_commit_after_failure']}
CALLS={'warmup':1,'resume':4,'transition':1,'failure':1}
QUAL_OPTIONS=dict(total_steps=4,warmup_fraction=.25,U_ramp_fraction=.25,PAS_confidence=0.,PAS_cosine=-1.)


def qualification_plan():
    return dict(cases=[dict(id=f'{f}/{name}',family=f,kind=name,physical_calls=CALLS[name],checks=keys)
                       for f in (B0,B2,F5) for name,keys in CASE_CHECKS.items()],planned_calls=21,cap=60,
                options=QUAL_OPTIONS,synthetic_shape=[2,3,384,384],
                fixture='random native network; fixed foreground-biased readout bias; generated rim/cup labels; no source file',
                extra_F5_loss_evaluations=2,comparison='same environment allclose rtol=1e-5 atol=2e-6; exact RNG/integer state',
                source_reads=0,source_training=0,state='PENDING_NATIVE_CUDA',
                historical_environment=dict(python='3.10.6',torch='2.2.1+cu121',GPU='UNVERIFIED',cudnn='UNVERIFIED'),
                environment_binding='actual fingerprint frozen at first authorized qualification; identical for smoke/run')


def run_cases(config,plan,permit,root,device,counter):
    import copy,random
    import torch
    from ..five_frameworks_v1.native_parent import build,NativeLRParent
    from ..five_frameworks_v1.recipes import SyntheticCurrentDomain
    from ..five_frameworks_v1.model import Model
    from ..five_frameworks_v1.train_stage import StageTrainer
    from ..five_frameworks_v1.native_state import resume
    from ..five_frameworks_v1 import checkpoint
    from ..five_frameworks_v1.kernels import channel_basis
    from ..five_frameworks_v1.semantics import tensor_fingerprint

    class GeneratedDomain(SyntheticCurrentDomain):
        def labeled(self,*a,**k):
            x,y,ids=super().labeled(*a,**k)
            y=(torch.arange(self.size)[None,None,:].expand(2,self.size,-1)%2+1).long()
            return x.to(device),y.to(device),ids
        def unlabeled(self,*a,**k):
            x,g,ids=super().unlabeled(*a,**k);return (x*1.7+.3).to(device),g.to(device),ids

    def equal(a,b):
        if isinstance(a,torch.Tensor):
            assert a.shape==b.shape and a.dtype==b.dtype
            assert torch.allclose(a.to('cpu'),b.to('cpu'),rtol=1e-5,atol=2e-6) if a.is_floating_point() else torch.equal(a.cpu(),b.cpu())
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:equal(a[k],b[k])
        elif isinstance(a,(tuple,list)):
            assert len(a)==len(b)
            for x,y in zip(a,b):equal(x,y)
        else:assert a==b

    def snapshot(t):
        return copy.deepcopy(dict(student=t.model.state_dict(),ema=t.ema.state_dict(),optimizer=t.optimizer.state_dict(),
                    scheduler=t.scheduler.state_dict(),prototypes=t.prototypes.values,support=t.prototypes.support,
                    cursor=t.cursor,step=t.step,reads=(t.provider.l_reads,t.provider.u_reads),torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate()))

    rows=[]
    def record(f,kind,start):
        calls=counter.count-start
        assert calls==CALLS[kind]
        row=dict(id=f'{f}/{kind}',status='PASS',physical_calls=calls,ledger_start=start+1,ledger_end=counter.count,
                 checks={k:True for k in CASE_CHECKS[kind]})
        rows.append(row)
        from .protocol import write
        write(root/'CUDA_CASE_PROGRESS.json',dict(rows=rows,physical_calls=counter.count))

    for f in (B0,B2,F5):
        options={**plan['options'][f]['RIM_ONE_r3'],**QUAL_OPTIONS}
        source=dict(kind='generated_native_fixture',node_id='QUAL_SOURCE',domain='REFUGE',seed=163)
        provider=GeneratedDomain(seed=163,size=384,stage_source=source)
        native=build(config['reference'],device,163)
        # Qualification-only synthetic readout fixture. Never applied to a real source/model.
        with torch.no_grad():native.decoder.conv_logit.mu.bias.copy_(torch.tensor([-4.,4.,-4.],device=device))
        model=Model(NativeLRParent(native,163,source),f,ratio=options.get('rank_ratio',.5)).to(device)
        t=StageTrainer(model,provider,options,execution=permit);counter.wrap(t.optimizer)
        ident=dict(family=f,candidate_id=plan['nodes'][0]['candidate_id'] if f==B0 else ('F5_C02' if f==F5 else 'B2_PARENT_PAS_KL_C06'),
                   seed=163,order=1,stage=1,sequence_id=f'QUAL_{f}',domain='RIM_ONE_r3',node_id=f'QUAL_{f}_1',execution_commit=config['execution_commit'])
        start=counter.count;t.update()
        assert t.native and isinstance(t.model.parent,NativeLRParent)
        assert t.provider.u_reads==0 and not t.last['active_U'] and __import__('math').isfinite(t.last['labeled_loss'])
        record(f,'warmup',start)
        if f==F5:
            # Two finite synthetic loss evaluations expose actual nonzero SWD contribution.
            # No optimizer call, no patient data. Counted by NativeOperations separately.
            _,with_swd,_=t.losses();assert max(t.last['swd_counts'].values())>=8
            weight=t.options['lambda_SWD'];t.options['lambda_SWD']=0.
            try:_,without_swd,_=t.losses()
            finally:t.options['lambda_SWD']=weight
            assert float((with_swd-without_swd).detach().abs())>1e-9
            del with_swd,without_swd
        path=root/f'{f}_resume.pt';checkpoint.save(t,path,ident)
        start=counter.count
        for _ in range(2):
            gradients=t.update()
            assert __import__('math').isfinite(t.last['labeled_loss'])
            if f==B0:assert t.provider.u_reads==0
            else:
                assert t.last['active_U'] and t.last['accepted']>0 and __import__('math').isfinite(t.last['unlabeled_loss'])
                allowed={id(p) for p in t.model.u_parameters()}
                assert all(v['U'] is None for k,v in gradients.items() if k not in allowed)
                assert any(v['U'] is not None and v['U'].abs().max()>0 for v in gradients.values())
                if f==F5:assert allowed=={id(t.model.sidecar.r)} and max(t.last['swd_counts'].values())>=8 and t.telemetry['extra_clean_U_forwards']>0
        expected=snapshot(t)
        resumed_provider=GeneratedDomain(seed=163,size=384,stage_source=source)
        r=resume(path,config['reference'],device,resumed_provider,options,None,ident,permit)
        counter.wrap(r.optimizer)
        for _ in range(2):r.update()
        actual=snapshot(r);equal(expected,actual)
        # RNG comparisons must be exact, independently of tensor tolerance.
        assert torch.equal(expected['torch_rng'],actual['torch_rng'])
        assert all(torch.equal(a,b) for a,b in zip(expected['cuda_rng'],actual['cuda_rng']))
        record(f,'resume',start)
        del expected,actual,t,model,native,gradients
        r.model.eval().requires_grad_(False);r.model.role='eval'
        x=r.provider.labeled(5)[0]
        with torch.no_grad():before=r.model(x);deployed=r.model.deploy();after=deployed(x)
        assert torch.allclose(before,after,rtol=1e-5,atol=2e-6)
        previous=deployed.transform.detach().clone();weights=copy.deepcopy(deployed.parent.native.state_dict())
        sid=dict(node_id=ident['node_id'],seed=163,domain='RIM_ONE_r3',student_hash=tensor_fingerprint(weights))
        native=build(config['reference'],device,163);native.load_state_dict(weights)
        p2=GeneratedDomain(seed=163,stage=2,size=384,stage_source=sid)
        stage2=StageTrainer(Model(NativeLRParent(native,163,sid),f,ratio=options.get('rank_ratio',.5),previous=previous).to(device),p2,options,execution=permit)
        ident2={**ident,'stage':2,'node_id':f'QUAL_{f}_2','domain':'Drishti_GS'}
        checkpoint.require_predecessor(ident2,ident)
        assert all(a.b.count_nonzero()==0 and a.new for a in stage2.model.parent.adapters)
        equal(stage2.ema.parent.native.state_dict(),weights)
        assert stage2.step==stage2.cursor==0 and not stage2.optimizer.state and stage2.ema is not r.ema
        if f!=B0:
            proto,support,*_=stage2.prototypes.estimate(stage2.ema,*p2.labeled(0,'prototype_initialization')[:2])
            equal(stage2.prototypes.values,proto);equal(stage2.prototypes.support,support)
        else:assert not stage2.prototypes.support.any()
        if f==F5:
            assert stage2.model.sidecar.r.count_nonzero()==0
            equal(stage2.model.sidecar.previous,previous)
            q,_=channel_basis(stage2.model.parent.effective_readout_kernel_at_entry(),16,previous)
            equal(stage2.model.sidecar.q,q[:,:4])
        else:assert stage2.model.sidecar is None and torch.equal(previous,torch.eye(16,device=device))
        start=counter.count;counter.wrap(stage2.optimizer);stage2.update();record(f,'transition',start)
        start=counter.count;committed=stage2.step
        try:stage2.update(fault='after_optimizer')
        except RuntimeError as e:assert 'injected after_optimizer' in str(e)
        else:raise AssertionError('failure injection did not fail')
        assert counter.count==start+1 and stage2.step==committed and stage2.requires_restore
        try:stage2.update()
        except RuntimeError as e:assert 'requires checkpoint restore' in str(e)
        else:raise AssertionError('failed trainer committed')
        assert counter.count==start+1;record(f,'failure',start)
        del stage2,r,native,deployed,weights,x,before,after
        torch.cuda.empty_cache()
    return rows

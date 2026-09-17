"""Bounded CPU-native synthetic qualification; no real data/checkpoint access."""
import copy
import os
import platform
import random
import tempfile
import time
import traceback
from pathlib import Path
import torch
from .protocol import DOC,STUDY,read,write,digest,validate_plan,validate_prefix,code_manifest
from .alignment import patches,interior,alignment_loss,binding,coordinate_basis,capture_input,LAYER
from .trainer import KeyAlignmentTrainer
from ..five_frameworks_v1.native_parent import build,NativeLRParent
from ..five_frameworks_v1.integration import ExecutionPermit,_PERMIT_SEAL
from ..five_frameworks_v1.model import Model
from ..five_frameworks_v1.recipes import SyntheticCurrentDomain
from ..five_frameworks_v1.train_stage import StageTrainer
from ..five_frameworks_v1.native_runner import Counter
from ..five_frameworks_v1.native_operations import NativeOperations

NAMES=('patch_geometry','loss_contract','matrix_prefix_scope','native_binding_gradients',
       'zero_equivalence','matched_arms','two_case_overfit')


def rejects(fn):
    try:fn()
    except (ValueError,PermissionError):return
    raise AssertionError('expected refusal')


class Generated(SyntheticCurrentDomain):
    def labeled(self,step,stream='labeled'):
        x,_,ids=super().labeled(step,stream)
        y=torch.arange(self.size)[None,None,:].expand(2,self.size,-1)%2+1
        return x,y.long(),ids


def run_tests(reference):
    import sys
    if sys.flags.optimize or os.environ.get('PYTHONOPTIMIZE') or torch.cuda.is_initialized():
        raise RuntimeError('CPU unoptimized interpreter required')
    torch.set_num_threads(2)
    plan=validate_plan(read(DOC/'D1_MATRIX.json'))
    attempts=read(DOC/'CPU_ATTEMPTS.json') if (DOC/'CPU_ATTEMPTS.json').exists() else []
    if len(attempts)>=2:raise RuntimeError('two-invocation synthetic cap')
    attempt=dict(invocation=len(attempts)+1,status='STARTED',started=time.time(),tests=list(NAMES))
    attempts.append(attempt);write(DOC/'CPU_ATTEMPTS.json',attempts)
    counter=Counter(DOC/'CPU_PHYSICAL.jsonl',32);before=counter.count
    permit=ExecutionPermit(dict(study_id=STUDY,execution_scope='cpu_synthetic'),('D0',),dict(optimizer_calls=32),_PERMIT_SEAL)
    observations={};trainer_costs=[]

    def trainer(arm=None,weight=None,provider=None,options=None):
        native=build(reference,torch.device('cpu'),163)
        # Generated-only foreground fixture. Native parameter schema unchanged.
        from ..f5_confirmation_v1.native_qualification import foreground_fixture
        foreground_fixture(native)
        p=provider or Generated(seed=163,order=1,stage=2,size=32,stage_source={'kind':'generated','domain':'RIM_ONE_r3'})
        m=Model(NativeLRParent(native,163,p.stage_source),'B2_PARENT_PAS_KL')
        opts={**plan['options']['Drishti_GS'],'total_steps':5,'warmup_fraction':.2,'U_ramp_fraction':.2,
              'PAS_confidence':0.,'PAS_cosine':-1.,**(options or {})}
        t=StageTrainer(m,p,opts,execution=permit) if arm is None else KeyAlignmentTrainer(m,p,opts,arm,permit,weight)
        counter.wrap(t.optimizer)
        return t

    def patch_geometry():
        x=torch.arange(2*16*9*11,dtype=torch.float32).reshape(2,16,9,11).requires_grad_()
        mask=interior(torch.ones(2,9,11,dtype=torch.bool));idx=torch.where(mask.flatten())[0][::7]
        actual=patches(x,idx)
        expected=torch.nn.functional.unfold(x,3,padding=1).transpose(1,2).reshape(-1,144)[idx]
        assert torch.equal(actual,expected)
        labels=torch.arange(9*11).reshape(1,9,11).expand(2,-1,-1)
        assert torch.equal(actual[:,4],x[:,0].flatten()[idx])  # center of channel 0
        assert torch.equal(labels.flatten()[idx],idx%(9*11))
        invalid=torch.ones(2,9,11,dtype=torch.bool);invalid[:,4,5]=False
        assert not interior(invalid)[:,3:6,4:7].any()
        rejects(lambda:patches(x,torch.tensor([0])))
        actual.sum().backward();assert x.grad.count_nonzero()>0
        observations['patch_geometry']=dict(input=[2,16,9,11],patch=[len(idx),144],unfold_exact=True,
            coordinate_mapping='identity grid, row-major center selection, channel/ky/kx flatten')

    def loss_contract():
        torch.manual_seed(90)
        a=torch.randn(2,16,12,12,requires_grad=True);b=torch.randn_like(a,requires_grad=True)
        labels=(torch.arange(12)[None,None,:].expand(2,12,-1)%2+1).long()
        mask=torch.ones_like(labels,dtype=torch.bool);p=Generated(size=12)
        v=torch.linalg.qr(torch.randn(144,8),mode='reduced')[0].requires_grad_()
        loss,stats=alignment_loss(a,b,labels,labels,mask,mask,v,p,4)
        ga,gb,gv=torch.autograd.grad(loss,(a,b,v),allow_unused=True)
        assert loss>0 and ga.norm()>0 and gb is None and gv is None and stats['valid_classes']==2
        zero,empty=alignment_loss(a,b,labels,labels,mask&False,mask,v,p,4)
        g,=torch.autograd.grad(zero,a);assert zero==0 and not g.any() and empty['invalid_step']
        positions=[]
        for key in (None,coordinate_basis('C2',v,p),v):
            value,s=alignment_loss(a,b,labels,labels,mask,mask,key,p,4);positions.append(s['positions'])
            assert abs(s['raw_scaled_loss']-s['raw_SWD']*s['dimension']/8)<1e-7
        assert positions[0]==positions[1]==positions[2]
        rejects(lambda:alignment_loss(a,b,labels[:,:,:-1],labels,mask,mask,v,p,4))
        observations['loss_contract']=dict(reference_and_key_detached=True,upstream_nonzero=True,
            zero_connected=True,positions_dimension_independent=True,equal_class_mean=True)

    def matrix_prefix_scope():
        assert len(plan['prefixes'])==4 and plan['counts']['formal_updates']==42400
        # Generated metadata exercises diagnostic permission; not a real receipt.
        node=copy.deepcopy(plan['nodes'][0]);record=copy.deepcopy(plan['prefixes'][0])
        receipt=dict(identity=record['identity'],status='SEALED',student_hash=record['student_sha256'])
        record['receipt_sha256']=digest(receipt);record['binding_sha256']=digest({k:v for k,v in record.items() if k!='binding_sha256'})
        node['prefix_binding_sha256']=record['binding_sha256'];original=copy.deepcopy(receipt)
        validate_prefix(node,record,receipt);assert receipt==original
        for key,value in [('seed',164),('order',2),('stage',1),('phase','D2'),('arm','F5')]:
            bad={**node,key:value};rejects(lambda:validate_prefix(bad,record,receipt))
        bad=copy.deepcopy(receipt);bad['identity']['family']='C3';rejects(lambda:validate_prefix(node,record,bad))
        observations['matrix']=dict(targets=16,formal_updates=42400,source=0,first_target=0,D2_D3_executable_nodes=0)

    def native_binding_gradients():
        t=trainer('C3');layer,v=binding(t.model.parent)
        x,y,_=t.provider.labeled(1);u,geometry,_=t.provider.unlabeled(1)
        with torch.no_grad(),capture_input(t.ema.parent.native.get_submodule(LAYER)) as ref:t.ema.parts(x,mode='teacher')
        with capture_input(layer) as cur:t.model.parts(u)
        loss,stats=alignment_loss(cur[0],ref[0],y,y,geometry,geometry,v,t.provider,1)
        params=[(n,p) for n,p in t.model.named_parameters() if p.requires_grad]
        grads=torch.autograd.grad(loss,[p for _,p in params],allow_unused=True)
        nonzero=[n for (n,p),g in zip(params,grads) if g is not None and g.norm()>0]
        assert loss>0 and any(n.endswith('.B') for n in nonzero)
        assert all(g is None for (n,p),g in zip(params,grads) if LAYER in n)
        assert not any(n.endswith('.A') for n in nonzero) # B=0 initialization, not a failure.
        t.model.parent.apply_constraints()
        assert not v.requires_grad and t.model.sidecar is None
        before=v.clone();c2=coordinate_basis('C2',v,t.provider)
        assert torch.equal(before,v) and torch.allclose(c2.T@c2,torch.eye(8,dtype=c2.dtype),atol=1e-12)
        observations['native_binding']=dict(layer=LAYER,input_shape=list(cur[0].shape),weight_shape=list(layer.weight.shape),
            V_shape=list(v.shape),V_dtype=str(v.dtype),V_source='parent.V64_13',right_dtype=str(layer.right.dtype),
            nonzero_auxiliary_parameters=nonzero,new_B_zero_A_gradient_zero=True,selected_layer_auxiliary_gradients_absent=True,
            no_added_model_parameters=True,parent_identity=t.model.parent.identity)
        old=ExecutionPermit(dict(study_id='F5_CONFIRMATION_V1',execution_scope='cpu_synthetic'),('P1',),{},_PERMIT_SEAL)
        rejects(lambda:KeyAlignmentTrainer(t.model,t.provider,t.options,'C3',old))
        rejects(lambda:KeyAlignmentTrainer(t.model,t.provider,t.options,'C3',permit,.1))
        rejects(lambda:KeyAlignmentTrainer(t.model,object(),t.options,'C3',permit))

    def snapshot(t):
        return copy.deepcopy(dict(student=t.model.state_dict(),ema=t.ema.state_dict(),optimizer=t.optimizer.state_dict(),
            scheduler=t.scheduler.state_dict(),prototypes=t.prototypes.values,support=t.prototypes.support,
            cursor=t.cursor,step=t.step,reads=(t.provider.l_reads,t.provider.u_reads)))

    def equal(a,b):
        if isinstance(a,torch.Tensor):assert torch.equal(a,b)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:equal(a[k],b[k])
        elif isinstance(a,(tuple,list)):
            assert len(a)==len(b)
            for x,y in zip(a,b):equal(x,y)
        else:assert a==b

    def zero_equivalence():
        states=[];random_states=[]
        for arm in (None,'C0','C3'):
            torch.manual_seed(114);random.seed(114)
            t=trainer(arm,0. if arm else None)
            for _ in range(2):t.update()
            states.append(snapshot(t));random_states.append((torch.get_rng_state().clone(),random.getstate()))
            if arm:trainer_costs.append(dict(arm=arm,**t.alignment_cost))
        equal(states[0],states[1]);equal(states[0],states[2]);equal(random_states[0],random_states[1]);equal(random_states[0],random_states[2])
        observations['lambda_zero']=dict(exact_state=True,exact_RNG=True,exact_buffers=True,exact_samples=True,
            compared=['native B2','C0 no-op','C3 lambda=0'])

    def matched_arms():
        stats=[];supervised=[];kl=[];exposure=[]
        noop=trainer('C0');noop.step=noop.cursor=1;noop.losses()
        supervised.append(noop.last['labeled_loss']);kl.append(noop.last['KL'])
        exposure.append((noop.provider.l_reads,noop.provider.u_reads));trainer_costs.append(dict(arm='C0 exposure-only',**noop.alignment_cost))
        for arm in ('C1','C2','C3'):
            t=trainer(arm);t.step=t.cursor=1  # generated fixture reaches active-U without extra updates.
            protected=[a.layer.right.clone() for a in t.model.parent.adapters]
            t.update();t.model.parent.apply_constraints()
            assert all(torch.equal(p,a.layer.right) for p,a in zip(protected,t.model.parent.adapters))
            stats.append(t.last['alignment']);supervised.append(t.last['labeled_loss']);kl.append(t.last['KL'])
            exposure.append((t.provider.l_reads,t.provider.u_reads));trainer_costs.append(dict(arm=arm,**t.alignment_cost))
            assert t.last['alignment']['valid_classes']>0
            assert t.last['gradient_diagnostics']['norms']['alignment']>0
        assert stats[0]['positions']==stats[1]['positions']==stats[2]['positions']
        assert len(set(supervised))==len(set(kl))==len(set(exposure))==1
        observations['matched_arms']=dict(same_positions=True,same_supervised=True,same_KL=True,same_exposure=True,
            dimensions=[s['dimension'] for s in stats],auxiliary_losses=[s['raw_scaled_loss'] for s in stats],
            native_protection_unchanged=True)

    def two_case_overfit():
        class Fixed(Generated):
            def labeled(self,step,stream='labeled'):return super().labeled(0,'fixed_case')
        rows=[]
        for seed in (1701,1702):
            p=Fixed(seed=seed,stage=2,size=32,stage_source={'kind':'generated','domain':'RIM_ONE_r3'})
            t=trainer('C3',provider=p,options={'total_steps':20,'warmup_fraction':1.})
            losses=[]
            for _ in range(3):t.update();losses.append(t.last['labeled_loss'])
            assert losses[-1]<losses[0]
            rows.append(dict(generated_case=seed,losses=losses,updates=3))
        observations['two_case_overfit']=dict(kind='finite loss-decrease sanity only, not convergence or scientific seeds',cases=rows)

    results=[]
    with tempfile.TemporaryDirectory(prefix='nka_cpu_') as tmp:
        operations=NativeOperations(Path(tmp)/'operations')
        with operations:
            checks=locals()
            for name in NAMES:
                try:checks[name]();results.append(dict(test=name,status='PASS'))
                except Exception:
                    results.append(dict(test=name,status='FAIL',traceback=traceback.format_exc().replace(str(DOC.parents[3]),'<repo>')))
        counts=dict(operations.counts)
    attempt.update(status='PASS' if all(r['status']=='PASS' for r in results) else 'FAIL',results=results,
        calls=counter.count-before,cumulative_calls=counter.count,seconds=time.time()-attempt['started'])
    write(DOC/'CPU_ATTEMPTS.json',attempts)
    report=dict(status=attempt['status'],study_id=STUDY,state='CODE_READY_FOR_REVIEW' if attempt['status']=='PASS' else 'TEST_FAILURE',
        tests=results,observations=observations,cost=dict(physical_calls_this_invocation=counter.count-before,
        cumulative_CPU_calls=counter.count,cap=32,invocations=len(attempts),max_invocations=2,operation_counts=counts,
        alignment=trainer_costs,test_only_gradient_VJPs=3,test_only_backward_calls=1,
        seconds=attempt['seconds']),plan_sha256=plan['plan_sha256'],code_tree_sha256=code_manifest()['code_tree_sha256'],
        environment=dict(python=platform.python_version(),torch=torch.__version__,device='cpu',optimize=sys.flags.optimize),
        real_data_reads=0,real_checkpoint_reads=0,CUDA_updates=0,real_optimizer_updates=0,
        limitations=['generated native CPU models, no real prefix tensors','native CUDA and real smoke not authorized',
                     'production execution/resume adapter deliberately disabled pending review'])
    write(DOC/'TEST_REPORT.json',report)
    if attempt['status']!='PASS':raise RuntimeError('CPU qualification failed; evidence retained; no automatic rerun')
    return report

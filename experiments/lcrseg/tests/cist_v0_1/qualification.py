"""26 synthetic updates per backend; bounded source/readout, isolation and recovery checks."""
import gc
import os
import platform
import tempfile
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import torch
from experiments.lcrseg.cist_v0_1 import core as c,engine as e,contract as ct
from experiments.lcrseg.ssl_anchored_mix_v0_1.telemetry import Operations

def rejected(fn):
    try:fn()
    except (RuntimeError,ValueError,PermissionError,FloatingPointError):return
    raise AssertionError('negative case accepted')

def run(output,reference,device):
    source=ct.verify();ct.neutral_subprocess_paths();dev=torch.device(device);b=Path(output);b.mkdir(parents=True);assert not (b/'qualification.json').exists()
    os.environ['SF_REFERENCE']=reference;os.environ['SF_DEVICE']=device
    from experiments.lcrseg.tests.ams_seq_transfer_v0_1.test_contract import fixture
    tests=[];started=time.time();cold=not torch.cuda.is_initialized() if dev.type=='cuda' else None
    try:
        with Operations(b/'operations',update_cap=26) as op,tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data';expected=fixture(data)
            def kw(tid,arm=c.MAIN,epochs=(20,21)):
                task=dict(task_id=tid,source_task_id='synthetic',seed=61,order='O1',source_domain=c.DOMAINS[0],domain=c.DOMAINS[1],arm=arm,updates=len(epochs),steps_per_epoch=1)
                return dict(base=root,task_id=tid,data=data,reference=reference,device=dev,fixture=dict(task=task,epochs=list(epochs),steps=1,expected=expected))
            if dev.type=='cuda':assert cold
            cold_result=e.train(**kw('cold',epochs=(20,)));gc.collect();assert cold_result['updates']==1
            tests.append('fresh_process_CUDA_init_before_peak_reset_and_pending_save')
            with c.rng(torch.device('cpu'),'CIST_qualification_math'):
                kernel=torch.randn(3,16,3,3);tensors,meta=c.readout_basis(kernel,61,c.DOMAINS[0]);v=torch.randn(16,dtype=torch.float64);pi=torch.eye(3,dtype=torch.float64)-torch.ones(3,3,dtype=torch.float64)/3
                direct=sum(float((pi@kernel[:,:,i,j].double()@v).square().sum()) for i in range(3) for j in range(3))
                assert abs(float(v@tensors['G_R']@v)-direct)<1e-10
                rejected(lambda:c.readout_basis(torch.zeros(3,16,3,3),61,c.DOMAINS[0]));rejected(lambda:c.readout_basis(torch.zeros(3,144),61,c.DOMAINS[0]))
                m=(torch.randn(1,8,8,dtype=torch.float64,device=dev)*.1).requires_grad_()
                assert torch.autograd.gradcheck(lambda x:c.cayley(x)[0],(m,),eps=1e-6,atol=1e-5,rtol=1e-3)
                for scale in (0.,.1,1.,10.):
                    q,a=c.cayley(torch.randn(2,8,8,device=dev)*scale)
                    assert float((q.transpose(-1,-2)@q-torch.eye(8,device=dev)).norm())<c.ISO64_LIMIT
                rejected(lambda:c.cayley(torch.full((1,8,8),float('nan'),device=dev)))
            tests+=['16_channel_readout_contrast_Gram_and_degenerate_inputs','FP64_Cayley_gradcheck_orthogonality_nonfinite']
            ds=c.DomainData(data,c.DOMAINS[1],'train_labeled',expected=expected,shape=(24,24));us=c.DomainData(data,c.DOMAINS[1],'train_unlabeled',expected=expected,shape=(24,24))
            lb=c.batch(ds,[0,1],dev,61,21,0,'labeled');ub=c.batch(us,[0,1],dev,61,21,0,'unlabeled',augment=False)
            assert set(ub)=={'image','geometry'} and all('label_h5_relpath' not in r for r in us._dataset.rows)
            image=ub['image'].clone();altered=c.appearance(image,(61,c.DOMAINS[1],21,0));assert torch.equal(image,ub['image']) and altered.shape==image.shape
            ramp=torch.linspace(0,1,24*24,device=dev).reshape(1,1,24,24).expand(2,3,-1,-1);view=c.appearance(ramp,('monotone_test',)).flatten(2)
            assert (view.diff(dim=-1)>=0).all()
            tests.append('unaltered_U_coordinates_and_no_U_GT_fields')
            for arm in c.ARMS:
                args=kw('manual',arm);model,opt,entry,bound=e.initial(root,args['fixture']['task'],reference,dev,args['fixture'])
                with torch.no_grad():
                    native,h=model.core(lb['image'],stochastic_classifier=False);z,(_,hp,detail)=model(lb['image'])
                    assert torch.equal(h,hp) and torch.equal(native,z)
                    module=model.transport
                    if arm!='SCALE_LU':
                        with c.rng(dev,'CIST_geometry_fixture',arm):module.mlp[-1].weight.normal_(std=.1);module.mlp[-1].bias.normal_(std=.1)
                        zz,dt=module(h);diag=c.mechanism(h,zz,dt)
                        if arm.startswith('ISO_'):assert diag['distance_relative_max']<=c.DISTANCE32_LIMIT
                        if arm=='ISO_GLOBAL_LU':assert torch.equal(dt['Q'][0],dt['Q'][1]) and torch.equal(dt['b'][0],dt['b'][1])
                        if arm==c.MAIN:
                            assert not torch.equal(dt['Q'][0],dt['Q'][1])
                            one,_=module(h[:1]);assert torch.allclose(one,zz[:1],atol=2e-6,rtol=2e-6)
                        torch.nn.init.zeros_(module.mlp[-1].weight);torch.nn.init.zeros_(module.mlp[-1].bias)
                for index,epoch in enumerate((1,21)):
                    u=ub if epoch>20 and arm!='ISO_COND_L' else None
                    row=c.step(model,opt,lb,u,args['fixture']['task'],epoch,index,Counter(),['p0','p1'])
                    assert row['labeled_source_scoring_multiplicity']==1 and row['donor_extra_image_reads']==row['pseudo_label_records']==row['EMA_updates']==0
                    if arm=='ISO_GLOBAL_LU':assert row['first_layer_weight_gradient_norm']==0
                assert c.state_hash(model.core)==bound['source_student_hash'] and e.diagnose(model,bound)['status']=='PASS'
                assert row['U_image_records']==(0 if arm=='ISO_COND_L' else 2)
                tests.append('identity_frozen_core_backward_and_geometry_'+arm);del model,opt,module;gc.collect()
            with c.training_access(c.DOMAINS[1],'ISO_COND_L'):
                rejected(lambda:c.DomainData(data,c.DOMAINS[1],'train_unlabeled',expected=expected))
            with c.training_access(c.DOMAINS[1],c.MAIN):
                rejected(lambda:c.DomainData(data,c.DOMAINS[0],'train_labeled',expected=expected))
                rejected(lambda:c.DomainData(data,c.DOMAINS[1],'val',evaluator=True,expected=expected))
            tests.append('current_domain_capabilities_and_L_only_U_denial')
            args=kw('overfit');model,opt,entry,bound=e.initial(root,args['fixture']['task'],reference,dev,args['fixture']);losses=[]
            for i in range(8):losses.append(c.step(model,opt,lb,None,args['fixture']['task'],1,i,Counter(),['p0','p1'])['loss'])
            assert losses[-1]<losses[0],losses
            del model,opt;gc.collect();tests.append('two_case_controller_overfit')
            full=e.train(**kw('full'));gc.collect()
            # An interruption after checkpoint save leaves exact POST_UPDATE_PENDING state.
            with patch.object(e,'pending',side_effect=RuntimeError('injected post-save interruption')):rejected(lambda:e.train(**kw('split')))
            gc.collect();part=root/'tasks/split';p=torch.load(part/'latest.pt',map_location='cpu',weights_only=False)
            assert p['position']==1 and p['state']=='POST_UPDATE_PENDING_DIAGNOSTICS'
            previous=op.counts['optimizer_steps'];oldhash=ct.sha256(part/'latest.pt')
            with patch.object(e,'diagnose',return_value=dict(status='FAIL',checks=dict(injected=False))):rejected(lambda:e.train(**kw('split'),resume=True))
            gc.collect();assert previous==op.counts['optimizer_steps'] and ct.read(part/'FAILURE_STATE_EVIDENCE.json')['position']==1
            def broken(payload,path):Path(path).write_bytes(b'partial');raise RuntimeError('interrupted atomic save')
            with patch.object(torch,'save',broken):rejected(lambda:e.save(part/'latest.pt',p))
            assert ct.sha256(part/'latest.pt')==oldhash
            resumed=e.train(**kw('split'),resume=True);gc.collect()
            for k in ('student_hash','controller_hash','optimizer_hash','label_order_hash','L_opens','U_opens'):assert full[k]==resumed[k],k
            assert (root/'tasks/full/steps.jsonl').read_text()==(part/'steps.jsonl').read_text()
            tests+=['pending_failure_does_not_repeat_optimizer','atomic_interruption_retains_exact_state','pending_pass_exact_resume']
            original=c.cayley
            def fail_in_training(x):
                if torch.is_grad_enabled():raise RuntimeError('injected solve failure')
                return original(x)
            with patch.object(c,'cayley',fail_in_training):rejected(lambda:e.train(**kw('solve_fail',epochs=(20,))))
            gc.collect();failed=root/'tasks/solve_fail';assert (failed/'pre_step.pt').exists() and (failed/'current_solve.pt').exists() and not (failed/'latest.pt').exists()
            e.train(**kw('solve_fail',epochs=(20,)),resume=True);gc.collect();tests.append('pre_solve_checkpoint_recovery_no_missing_optimizer')
            assert op.counts['optimizer_steps']==26 and op.counts['ema_updates']==0 and op.counts['sample_val']==0
        result=dict(status='PASS',source=source,tests=tests,counts=dict(op.counts),synthetic_updates=26,real_updates=0,cold_cuda=cold,
                    python=platform.python_version(),torch=torch.__version__,device=device,overfit_loss=losses,seconds=time.time()-started)
    except BaseException as exc:
        result=dict(status='FAIL',source=source,tests=tests,error=repr(exc),counts=dict(op.counts),real_updates=0,seconds=time.time()-started)
        c.write_json(b/'qualification.json',result);raise
    c.write_json(b/'qualification.json',result);print(dict(status=result['status'],updates=result['synthetic_updates'],tests=len(tests)),flush=True)

def smoke(output,base,data,reference,device):
    source=ct.verify();ct.neutral_subprocess_paths();b=Path(output);b.mkdir();dev=torch.device(device);rows=[];peak=reserved=0
    with Operations(b/'operations',update_cap=24) as op:
        for domain in c.DOMAINS:
            tasks=[t for t in ct.protocol()['tasks'] if t['domain']==domain and t['seed']==61]
            for task in tasks:
                with c.training_access(domain,task['arm']):
                    ds=c.DomainData(data,domain,'train_labeled');patients=c.patients_for(data,domain,(c.MANIFEST_SHA,c.SPLIT_SHA));l=c.batch(ds,[0,1],dev,61,21,0,'labeled')
                    u=None
                    if task['arm']!='ISO_COND_L':us=c.DomainData(data,domain,'train_unlabeled');u=c.batch(us,[0,1],dev,61,21,0,'unlabeled',augment=False)
                    torch.cuda.init();torch.cuda.reset_peak_memory_stats(dev);m,opt,entry,bound=e.initial(base,task,reference,dev)
                    for i in range(2):row=c.step(m,opt,l,u,task,21,i,Counter(),patients[:2])
                    assert e.diagnose(m,bound)['status']=='PASS'
                    peak=max(peak,torch.cuda.max_memory_allocated(dev));reserved=max(reserved,torch.cuda.max_memory_reserved(dev))
                    rows.append(dict(arm=task['arm'],domain=domain,loss=row['loss'],U_image_records=row['U_image_records'],controller_parameters=sum(p.numel() for p in m.transport.parameters())))
                    del m,opt,l,u;gc.collect();torch.cuda.empty_cache()
    import math
    result=dict(status='PASS',source=source,counts=dict(op.counts),discarded=True,real_updates=24,peak_allocated=peak,peak_reserved=reserved,
                admission_mib=max(1536,math.ceil(max(peak,reserved)/2**20*1.25+512)),rows=rows,source_retrained=False,val_GT_access=0)
    c.write_json(b/'receipt.json',result);print(dict(status='PASS',updates=24,admission_mib=result['admission_mib']),flush=True)

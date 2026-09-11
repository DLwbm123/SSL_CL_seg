"""29 synthetic updates per backend; no real image or evaluation access."""
import gc
import os
import platform
import tempfile
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from experiments.lcrseg.lctx_weight_memory_v0_1 import core as c,engine as e
from experiments.lcrseg.ssl_anchored_mix_v0_1.telemetry import Operations
from experiments.lcrseg.di_dmpa_gate1.binding import sha256
from . import diagnostics as d,state as s,contract as ct

def rejected(call):
    try:call()
    except (FloatingPointError,RuntimeError,PermissionError):return
    raise AssertionError('negative accepted')

def fingerprint(m,t,opt,device):
    return s.value_hash(dict(student=m.state_dict(),EMA=t.state_dict(),optimizer=opt.state_dict(),
                    grads={k:p.grad for k,p in m.named_parameters()},cpu=torch.get_rng_state(),
                    cuda=torch.cuda.get_rng_state(device) if device.type=='cuda' else None,
                    python=c.random.getstate(),numpy=c.np.random.get_state(),
                    modes=[x.training for x in m.modules()]+[x.training for x in t.modules()],backend=d.backend()))

def numerical(device):
    evidence=[]
    with c.rng(device,'numerical_recovery_synthetic'):
        conv=nn.Conv2d(128,128,3,bias=False).to(device)
        with torch.no_grad():conv.weight.normal_()
        tensors=c.basis(conv.weight,61,c.DOMAINS[0],'synthetic128x1152')[0]
        layer=c.LowRankConv(conv,tensors,'LR_SRC_AB')
        raw=layer.B.detach().clone().normal_()
        for scale in (0.,1e-10,1e-4,.1):
            with torch.no_grad():layer.B.copy_(raw*scale)
            r=d.layer_audit(layer,tensors,'LR_SRC_AB','synthetic128x1152',20)
            assert r['passed'],r
            if scale==0:assert r['status']=='ZERO_UPDATE'
            if scale==1e-10:assert r['nonzero_update'] and r['quantized_away_fraction']>0
            evidence.append(r)
        assert any(r['status']=='ROUNDOFF_LIMITED_REPRESENTATION' for r in evidence)
        # Independent exact FP32 cast and scalar math.fsum reference, no model arithmetic changed.
        with torch.no_grad():layer.B.copy_(raw*1e-4)
        dd=layer.delta().detach().cpu().double();w0=tensors['W0'].double().flatten(1)
        actual=layer.effective().detach().cpu().flatten(1)
        assert torch.equal((w0+dd).float(),actual)
        import math
        delta=actual.double()-w0;u=tensors['U'];prod=u.T@delta
        for i,j in [(0,0),(17,431),(63,1151)]:
            ref=math.fsum(float(u[k,i])*float(delta[k,j]) for k in range(128))
            bound=d.gamma(128)*float((u[:,i].abs()*delta[:,j].abs()).sum())
            assert abs(float(prod[i,j])-ref)<=bound+1e-30
        negatives=[]
        def bad(label):
            r=d.layer_audit(layer,tensors,'LR_SRC_AB','negative',20)
            assert not r['passed'],label
            negatives.append(dict(test=label,failures=r['failures']))
        original_delta=layer.delta
        with torch.no_grad():layer.A.add_(torch.randn_like(layer.A)*.01);layer.B.copy_(raw*.01)
        with patch.object(layer,'delta',lambda:layer.B@(layer.A-(layer.A@layer.right)@layer.right.T)):bad('missing_left_projection')
        with patch.object(layer,'delta',lambda:(layer.B-layer.left@(layer.left.T@layer.B))@layer.A):bad('missing_right_projection')
        with torch.no_grad():saved=layer.right.clone();layer.right[0,0]+=.001
        bad('runtime_basis_contamination')
        with torch.no_grad():layer.right.copy_(saved);saved_w=layer.weight.clone();layer.weight[0,0,0,0]+=.001
        bad('source_weight_identity')
        with torch.no_grad():layer.weight.copy_(saved_w)
        injection=layer.left[:,0,None]@torch.ones(1,1152,device=device)*.01
        with patch.object(layer,'delta',lambda:original_delta()+injection):bad('protected_direction_injection')
        effective=layer.effective
        with patch.object(layer,'effective',lambda:effective()+.01):bad('effective_outside_addition_budget')
        for value in (float('nan'),float('inf')):
            with patch.object(layer,'effective',lambda:effective()*value):bad('nonfinite_'+str(value))
        return dict(positive=evidence,negative=negatives,independent_reference='FP64 exact-sum-to-FP32 equality plus three scalar math.fsum dot products')

def run(output,reference,device,development=False):
    source='DEVELOPMENT' if development else ct.verify();device=torch.device(device)
    b=Path(output);b.mkdir(parents=True);assert not (b/'qualification.json').exists()
    os.environ['SF_REFERENCE']=reference;os.environ['SF_DEVICE']=str(device)
    from experiments.lcrseg.tests.ams_seq_transfer_v0_1.test_contract import fixture
    started=time.time();tests=[];numbers=None
    try:
        with Operations(b/'operations',update_cap=29) as op, tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data';expected=fixture(data)
            def kw(tid,epochs=1,steps=1):
                task=dict(task_id=tid,arm='LR_SRC_AB',seed=61,order='O1',source_domain=c.DOMAINS[0],domain=c.DOMAINS[1],source_task_id='synthetic',updates=epochs*steps)
                return dict(base=root,task_id=tid,data=data,reference=reference,device=device,recovery=True,
                            fixture=dict(task=task,epochs=epochs,steps=steps,expected=expected,diagnose_every_checkpoint=True))
            cold=not torch.cuda.is_initialized() if device.type=='cuda' else None
            if device.type=='cuda':assert cold,'qualification must start in fresh process'
            cold_result=e.train(**kw('cold'));gc.collect()
            assert cold_result['updates']==1 and (root/'tasks/cold/diagnostics_pass_1.json').exists()
            tests.append('fresh_process_cold_save_diagnostics_merge')
            numbers=numerical(device);tests.append('independent_roundoff_reference_and_negative_controls')
            ds=c.DomainData(data,c.DOMAINS[1],'train_labeled',expected=expected,shape=(24,24))
            lb=c.batch(ds,[0,1],device,61,21,0,'labeled')
            for arm in c.ARMS:
                comparisons=[]
                for audit in (False,True):
                    args=kw('trajectory')['fixture'];args['task']['arm']=arm
                    m,t,opt,load,entry,boundary=e.initial(root,args['task'],reference,device,args)
                    trace=[]
                    for index,epoch in enumerate((1,21)):
                        row=c.step(m,t,opt,lb,61,c.DOMAINS[1],epoch,index,Counter(),['p0','p1'],arm)
                        before=fingerprint(m,t,opt,device)
                        if audit:d.require(d.collect(m,load,epoch,arm))
                        assert before==fingerprint(m,t,opt,device)
                        trace.append(dict(row=row,effective=c.hash_state(dict(c.dense_items(m))),student=c.state_hash(m),
                                          gradients=s.value_hash({n:p.grad for n,p in m.named_parameters()}),
                                          EMA=c.state_hash(t),Adam=c.optimizer_hash(opt)))
                    comparisons.append(trace);del m,t,opt,load;gc.collect()
                assert comparisons[0]==comparisons[1],arm
                tests.append('exact_old_kernel_and_diagnostics_trajectory_'+arm)
            full=e.train(**kw('full',2));gc.collect()
            original=d.layer_audit
            def fault(layer,tensors,arm,name,epoch):
                if name==c.LAYERS[2]:raise FloatingPointError('synthetic injected diagnostic failure')
                return original(layer,tensors,arm,name,epoch)
            original_pending=s.check_pending
            def failed_pending(*a,**k):
                with patch.object(d,'layer_audit',fault):return original_pending(*a,**k)
            with patch.object(s,'check_pending',failed_pending):rejected(lambda:e.train(**kw('split',2)))
            gc.collect();part=root/'tasks/split';p=torch.load(part/'latest.pt',map_location='cpu',weights_only=False)
            assert p['position']==1 and p['checkpoint_state']=='POST_UPDATE_PENDING_DIAGNOSTICS'
            evidence=ct.read(part/'FAILURE_STATE_EVIDENCE.json');assert len(evidence['rows'])==14 and evidence['position']==1
            assert evidence['checkpoint_sha256']==sha256(part/'latest.pt')
            count=op.counts['optimizer_steps']
            with patch.object(s,'check_pending',failed_pending):rejected(lambda:e.train(**kw('split',2),resume=True))
            gc.collect();assert count==op.counts['optimizer_steps']
            old_hash=sha256(part/'latest.pt')
            def broken_save(payload,path):Path(path).write_bytes(b'partial');raise RuntimeError('synthetic interrupted write')
            with patch.object(torch,'save',broken_save):rejected(lambda:s.save(part/'latest.pt',p))
            assert sha256(part/'latest.pt')==old_hash
            split=e.train(**kw('split',2),resume=True);gc.collect()
            for key in ('student_hash','EMA_hash','optimizer_hash','label_order_hash','counts','L_opens','U_opens'):assert full[key]==split[key],key
            assert (root/'tasks/full/steps.jsonl').read_text()==(part/'steps.jsonl').read_text()
            tests+=['post_update_failure_all_14_layers_archived','pending_fail_no_repeated_update','atomic_interrupted_save_preserves_previous','pending_pass_exact_resume']
            assert op.counts['optimizer_steps']==29 and op.counts['sample_val']==0 and op.counts['sample_train_unlabeled']==0
        result=dict(status='PASS',source=source,tests=tests,numerical=numbers,counts=dict(op.counts),synthetic_updates=29,real_updates=0,
                    cold_cuda=cold,backend=d.backend(),python=platform.python_version(),seconds=time.time()-started)
    except BaseException as exc:
        result=dict(status='FAIL',source=source,tests=tests,error=repr(exc),counts=dict(op.counts),real_updates=0,seconds=time.time()-started)
        c.write_json(b/'qualification.json',result);raise
    c.write_json(b/'qualification.json',result);print({'status':result['status'],'updates':29,'tests':len(tests)},flush=True)

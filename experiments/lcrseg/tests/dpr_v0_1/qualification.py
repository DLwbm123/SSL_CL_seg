"""Synthetic algebra, actual PyTorch/CUDA Adam integration, and corrected-state recovery."""
import gc,os,time,tempfile,platform,copy
from pathlib import Path
from collections import Counter
from unittest.mock import patch
import torch
from experiments.lcrseg.dpr_v0_1 import core as c,engine as e,contract as ct
from experiments.lcrseg.ssl_anchored_mix_v0_1.telemetry import Operations
from experiments.lcrseg.tests.dpr_v0_1.formula_reference import run_checks,dense_kkt

def rejected(fn):
    try:fn()
    except (PermissionError,RuntimeError,ValueError):return
    raise AssertionError('forbidden operation accepted')

def algebra(dev):
    gen=torch.Generator(device=dev).manual_seed(2026091203);errors=[]
    for _ in range(60):
        g=torch.randn(37,device=dev,dtype=torch.float64,generator=gen);d0=-.01*g+.003*torch.randn(37,device=dev,dtype=torch.float64,generator=gen)
        J=torch.randn(2,37,device=dev,dtype=torch.float64,generator=gen);J/=J.norm();d,_=c.correct(d0,g,J,1)
        expected=torch.from_numpy(dense_kkt(d0.cpu().numpy(),g.cpu().numpy(),J.cpu().numpy(),1)).to(dev)
        error=float((d-expected).abs().max());errors.append(error)
        assert error<1e-12 and abs(g@(d-d0))<1e-12 and (J@d).norm()<=(J@d0).norm()+1e-12
        assert .5*(d-d0).square().sum()+.5*(J@d).square().sum()<=.5*(J@d0).square().sum()+1e-12
    for gg,jj,lam in [(g,J,0),(g,J*0,1),(g*0,J,1)]:assert torch.equal(c.correct(d0,gg,jj,lam)[0],d0)
    aligned=(g/g.norm())[None];assert torch.allclose(c.correct(d0,g,aligned,1)[0],d0,atol=1e-12,rtol=0)
    for preserve in (False,True):
        d,_=c.correct(d0,g,J,1,preserve);theta=torch.randn(37,device=dev,generator=gen).float();raw=(theta.double()+d0).float();d,_=c.correct(raw.double()-theta.double(),g,J,1,preserve)
        actual=(theta.double()+d).float();row=c.numerical_row(theta,raw,g,J,d,1,preserve,2,1,actual);assert all(row['checks'].values()),row
    return dict(max_dense_KKT_error=max(errors),cases=60)

def run(output,reference,device):
    source=ct.verify();ct.neutral_subprocess_paths();b=Path(output);b.mkdir();dev=torch.device(device)
    os.environ['SF_REFERENCE']=reference;os.environ['SF_DEVICE']=device
    from experiments.lcrseg.tests.ams_seq_transfer_v0_1.test_contract import fixture
    tests=[];started=time.time()
    try:
        formula=run_checks();alg=algebra(dev);tests.append('NumPy_and_device_FP64_KKT_degeneracies_rounding')
        with Operations(b/'operations',update_cap=48) as op,tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);data=base/'data';expected=fixture(data)
            def kw(tid,arm,epochs=(20,21,40)):
                task=dict(task_id=tid,source_task_id='synthetic',seed=61,order='O1',source_domain=c.DOMAINS[0],domain=c.DOMAINS[1],arm=arm,updates=len(epochs),steps_per_epoch=1,actual_diagnostic_positions=[2,3])
                return dict(base=base,task_id=tid,data=data,reference=reference,device=dev,fixture=dict(task=task,epochs=list(epochs),steps=1,expected=expected,U_count=3))
            ds=c.DomainData(data,c.DOMAINS[1],'train_labeled',expected=expected,shape=(24,24));lb=c.batch(ds,[0,1],dev,61,21,0,'labeled');probe={k:lb[k] for k in ('image','geometry')}
            rejected(lambda:c.response_input(lb));tests.append('response_label_capability_rejection')
            for arm in c.ARMS:
                args=kw('manual',arm);task=args['fixture']['task'];m,ema,opt,bound=e.initial(base,task,reference,dev,args['fixture'])
                row,_=c.step(m,ema,opt,lb,probe,task,40,0,Counter(),['p0','p1'])
                assert row['raw_response_norm']>0 and all(row['checks'].values()) and row['response_VJP']==2
                # Same keyed image-only batch produces the same response implementation regardless of L/U origin.
                J1,_=c.jacobian(m,dict(probe),(61,c.DOMAINS[1],40,1),Counter());J2,_=c.jacobian(m,dict(probe),(61,c.DOMAINS[1],40,1),Counter());assert torch.equal(J1,J2)
                assert e.diagnose(m,ema,bound,row)['status']=='PASS';del m,ema,opt,J1,J2;gc.collect();tests.append('actual_VJP_grad_isolation_'+arm)
                full=e.train(**kw('full_'+arm,arm));gc.collect()
                native_pending=e.pending
                def pause(root,m,ema,opt,p,counts,get_batches,**kwargs):
                    if p['position']==2:raise RuntimeError('injected pause after corrected pending save')
                    return native_pending(root,m,ema,opt,p,counts,get_batches,**kwargs)
                with patch.object(e,'pending',pause):rejected(lambda:e.train(**kw('split_'+arm,arm)))
                gc.collect();root=base/'tasks'/('split_'+arm);p=torch.load(root/'latest.pt',map_location='cpu',weights_only=False)
                assert p['position']==2 and p['state']=='POST_CORRECTION_PENDING_DIAGNOSTICS' and p['transient'] and 'J' not in p
                n=op.counts['optimizer_steps']
                with patch.object(e,'diagnose',return_value=dict(status='FAIL',checks={'injected':False})):rejected(lambda:e.train(**kw('split_'+arm,arm),resume=True))
                gc.collect();assert op.counts['optimizer_steps']==n
                resumed=e.train(**kw('split_'+arm,arm),resume=True);gc.collect()
                for key in ('student_hash','EMA_hash','optimizer_hash','label_order_hash','U_order_hash'):assert full[key]==resumed[key],(arm,key)
                assert (base/'tasks'/('full_'+arm)/'steps.jsonl').read_text()==(root/'steps.jsonl').read_text()
                payload=torch.load(root/'deploy_student.pt',map_location=dev,weights_only=False);m=c.from_state(reference,dev,payload.pop('student'),c.LINEAR)
                assert c.state_hash(m)==resumed['student_hash'];del m,p,payload;gc.collect();tests.append('pending_first_exact_resume_deploy_'+arm)
            # All warmup arms compare step-by-step with the genuine unchanged native F_CONV.
            from experiments.lcrseg.lctx_weight_memory_v0_1 import core as native
            records=[]
            for arm in ('native',*c.ARMS):
                task=kw('parity',c.MAIN)['fixture']['task'];m,ema,opt,bound=e.initial(base,task,reference,dev,True);record=[]
                for index in range(2):
                    before=e.rng_state(dev)
                    if arm=='native':row=native.step(m,ema,opt,lb,61,c.DOMAINS[1],20,index,Counter(),['p0','p1'],'F_CONV')
                    else:
                        task=dict(task,arm=arm);row,_=c.step(m,ema,opt,lb,None,task,20,index,Counter(),['p0','p1'])
                    assert torch.equal(before['cpu'],e.rng_state(dev)['cpu'])
                    record.append((c.state_hash(m),c.state_hash(ema),c.optimizer_hash(opt),row['loss']))
                records.append(record);del m,ema,opt;gc.collect()
            assert all(x==records[0] for x in records);tests.append('all_warmup_student_Adam_EMA_and_RNG_exact_native_parity')
            args=kw('overfit',c.MAIN);task=args['fixture']['task'];m,ema,opt,_=e.initial(base,task,reference,dev,True);losses=[]
            for i in range(4):losses.append(c.step(m,ema,opt,lb,None,task,1,i,Counter(),['p0','p1'])[0]['loss'])
            assert losses[-1]<losses[0];del m,ema,opt;gc.collect();tests.append('two_case_overfit')
            assert op.counts['optimizer_steps']==33 and op.counts['sample_val']==0
        result=dict(status='PASS',source=source,tests=tests,formula=formula,algebra=alg,counts=dict(op.counts),synthetic_updates=33,real_updates=0,
                    python=platform.python_version(),torch=torch.__version__,device=device,seconds=time.time()-started,overfit_loss=losses)
    except BaseException as exc:
        c.write_json(b/'qualification.json',dict(status='FAIL',source=source,tests=tests,error=repr(exc),counts=dict(op.counts) if 'op' in locals() else {},real_updates=0));raise
    c.write_json(b/'qualification.json',result);print(dict(status='PASS',updates=33,tests=len(tests)),flush=True)

def smoke(output,base,data,reference,device):
    source=ct.verify();ct.neutral_subprocess_paths();b=Path(output);b.mkdir();dev=torch.device(device);rows=[];peak=reserved=0
    with Operations(b/'operations',update_cap=12) as op:
        for task in [t for t in ct.protocol()['tasks'] if t['seed']==61]:
            with c.training_access(task['domain']):
                ds=c.DomainData(data,task['domain'],'train_labeled');patients=c.patients_for(data,task['domain'],(c.MANIFEST_SHA,c.SPLIT_SHA))
                torch.cuda.init();torch.cuda.reset_peak_memory_stats(dev);m,ema,opt,bound=e.initial(base,task,reference,dev)
                lb=c.batch(ds,[0,1],dev,61,40,0,'labeled');probe={k:lb[k] for k in ('image','geometry')}
                for i in range(2):
                    row,tmp=c.step(m,ema,opt,lb,probe,task,40,i,Counter(),patients[:2],diagnostic=i==1)
                    assert e.diagnose(m,ema,bound,row)['status']=='PASS'
                    if tmp:c.actual_diagnostic(m,lb,probe,task,40,i,Counter(),tmp,row)
                peak=max(peak,torch.cuda.max_memory_allocated(dev));reserved=max(reserved,torch.cuda.max_memory_reserved(dev))
                rows.append(dict(arm=task['arm'],domain=task['domain'],loss=row['loss'],response_norm=row['raw_response_norm'],numerics=row['checks'],trainable_parameters=bound['trainable_parameters']))
                del m,ema,opt,lb,probe,tmp;gc.collect();torch.cuda.empty_cache()
    import math
    result=dict(status='PASS',source=source,counts=dict(op.counts),real_updates=12,discarded=True,rows=rows,peak_allocated=peak,peak_reserved=reserved,admission_mib=math.ceil(max(peak,reserved)/2**20*1.25+768),U_images=0,val_GT=0,scope='L-only graph/memory qualification; U-arm query uses stripped L solely in discarded smoke')
    c.write_json(b/'receipt.json',result);print(dict(status='PASS',updates=12,admission_mib=result['admission_mib']),flush=True)

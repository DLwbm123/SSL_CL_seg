"""Bounded synthetic execution, finite counterexample, final-state recovery and L-only smoke."""
import gc,os,time,tempfile,platform
from pathlib import Path
from collections import Counter
from unittest.mock import patch
import torch
from experiments.lcrseg.dpr_finite_v0_2 import core as c,engine as e,contract as ct
from experiments.lcrseg.ssl_anchored_mix_v0_1.telemetry import Operations
from experiments.lcrseg.tests.dpr_finite_v0_2.formula_reference import main as formula,correction
from experiments.lcrseg.tests.dpr_v0_1.qualification import rejected

def algebra(dev):
    gen=torch.Generator(device=dev).manual_seed(2026091206);errors=[]
    for i in range(80):
        g=torch.randn(19,device=dev,dtype=torch.float64,generator=gen);R=torch.randn(2,19,device=dev,dtype=torch.float64,generator=gen);q=torch.randn(2,device=dev,dtype=torch.float64,generator=gen)
        if i==0:R[1]=0
        n=R.norm();eta,_=c.correct(g,R/n,q/n,1)
        ref=correction(g.cpu().numpy(),R.cpu().numpy(),q.cpu().numpy())[0]
        errors.append(float((eta-torch.from_numpy(ref).to(dev)).abs().max()));assert errors[-1]<1e-12
    for gg,jj,lam in [(g,R/n,0),(g*0,R/n,1),(g,R*0,1)]:assert torch.equal(c.correct(gg,jj,q,lam)[0],g*0)
    delta=torch.randn(2,2,32,32,device=dev,dtype=torch.float64,generator=gen,requires_grad=True);valid=torch.ones(2,1,32,32,device=dev,dtype=torch.bool);valid[:,:,:,0]=False
    key=(61,c.DOMAINS[1],40,0);v,_=c.probes(delta,valid,key,c.MAIN);s,_=c.probes(delta,valid,key,c.ARMS[1]);ss,_=c.probes(delta,valid,(*key[:-1],1),c.ARMS[1])
    assert not v.requires_grad and not s.requires_grad and torch.equal(v.abs(),s.abs()) and not torch.equal(s,ss) and torch.equal(s,c.probes(delta,valid,key,c.ARMS[1])[0])
    assert (v[:,:,:,0]==0).all()
    # Nonlinear VJP must be at raw, and detached probe differentiation excludes Hessian terms.
    x=torch.nn.Parameter(torch.tensor([1.,2.,3.],device=dev));x.grad=torch.tensor([0.,0.,1.],device=dev)
    chi=(x.square()[None,:,None,None]).expand(1,3,2,2)[:,:2];vv=torch.ones_like(chi).double().detach();rows=c.rows_at_raw(chi,vv,[x],Counter())
    assert torch.equal(rows,torch.tensor([[8.,0.,0.],[0.,16.,0.]],device=dev,dtype=torch.float64)) and torch.equal(x.grad,torch.tensor([0.,0.,1.],device=dev))
    return dict(cases=80,max_KKT_error=max(errors),stopgrad=True,raw_VJP=True,sign_absolute_match=True)

def guard_counterexample(dev):
    p=torch.nn.Parameter(torch.zeros(3,device=dev));g=torch.tensor([0.,0.,1.],device=dev);opt=torch.optim.Adam([p],lr=1.,eps=1e-8)
    d=torch.tensor([1.,.0001,-.1],device=dev);vnew=.999*torch.ones_like(p)+.001*g.square()
    opt.state[p].update(step=torch.tensor(0.),exp_avg=((-d*(vnew.div(.001).sqrt()+1e-8)*.1)-.1*g)/.9,exp_avg_sq=torch.ones_like(p))
    p.grad=g.clone();opt.step();raw=p.detach().clone();adam=c.optimizer_hash(opt)
    H=torch.diag(torch.tensor([1.,100.,0.],device=dev,dtype=torch.float64));delta=H@raw.double();v=delta/delta.norm();R=torch.stack([v@H,torch.zeros_like(v)]);q=torch.stack([v@delta,delta.new_tensor(0)])
    n=R.norm();eta,_=c.correct(g,R/n,q/n,1);candidate=(raw.double()+eta).float();c.assign([p],candidate)
    er=delta.square().sum();ec=(H@candidate.double()).square().sum();assert ec>er
    accepted=c.adopt([p],raw,candidate,er,ec)
    assert not accepted and torch.equal(p,raw) and c.optimizer_hash(opt)==adam and opt.state[p]['step']==1
    rejected(lambda:c.adopt([p],raw,candidate,er,ec*float('nan')))
    return dict(raw_energy=float(er),candidate_energy=float(ec),raw_restored_exactly=True,Adam_once=True)

def run(output,reference,device):
    source=ct.verify();ct.neutral_subprocess_paths();b=Path(output);b.mkdir();dev=torch.device(device)
    os.environ['SF_REFERENCE']=reference;os.environ['SF_DEVICE']=device
    from experiments.lcrseg.tests.ams_seq_transfer_v0_1.test_contract import fixture
    tests=[];started=time.time()
    try:
        with patch.object(Path,'write_text',return_value=None):fr=formula()
        alg=algebra(dev);tests+=['attached_80_case_algebra','device_KKT_degeneracy_detach_raw_VJP_sign_stream']
        with Operations(b/'operations',update_cap=36) as op,tempfile.TemporaryDirectory() as tmp:
            guard=guard_counterexample(dev);tests.append('linear_full_field_counterexample_actual_Adam_raw_restore')
            base=Path(tmp);data=base/'data';expected=fixture(data)
            def kw(tid,arm,epochs=(20,21,40)):
                task=dict(task_id=tid,source_task_id='synthetic',seed=61,order='O1',source_domain=c.DOMAINS[0],domain=c.DOMAINS[1],arm=arm,updates=len(epochs),steps_per_epoch=1,actual_diagnostic_positions=[2,3])
                return dict(base=base,task_id=tid,data=data,reference=reference,device=dev,fixture=dict(task=task,epochs=list(epochs),steps=1,expected=expected,U_count=3))
            ds=c.DomainData(data,c.DOMAINS[1],'train_labeled',expected=expected,shape=(24,24));lb=c.batch(ds,[0,1],dev,61,21,0,'labeled');probe={k:lb[k] for k in ('image','geometry')}
            rejected(lambda:c.response_input(lb));tests.append('response_GT_capability_rejection')
            for arm in c.ARMS:
                task=kw('manual',arm)['fixture']['task'];m,ema,opt,bound=e.initial(base,task,reference,dev,True)
                queries=[];native_query=c.query
                def traced(*a,**k):queries.append(c.vector(c.parameters(a[0])).clone());return native_query(*a,**k)
                with patch.object(c,'query',traced):row,_=c.step(m,ema,opt,lb,probe,task,40,0,Counter(),['p0','p1'])
                assert len(queries)==3 and not torch.equal(queries[0],queries[1]);assert row['response_VJP']==2 and all(row['checks'].values()),row
                assert e.diagnose(m,ema,bound,row)['status']=='PASS';assert all(s['step']==1 for s in opt.state.values())
                del m,ema,opt,queries;gc.collect();tests.append('one_Adam_three_queries_final_EMA_'+arm)
                full=e.train(**kw('full_'+arm,arm));gc.collect();native_pending=e.pending
                def pause(root,m,ema,opt,p,counts,get_batches,**kwargs):
                    if p['position']==2:raise RuntimeError('injected final pending pause')
                    return native_pending(root,m,ema,opt,p,counts,get_batches,**kwargs)
                with patch.object(e,'pending',pause):rejected(lambda:e.train(**kw('split_'+arm,arm)))
                gc.collect();root=base/'tasks'/('split_'+arm);p=torch.load(root/'latest.pt',map_location='cpu',weights_only=False)
                assert p['state']=='POST_CORRECTION_PENDING_DIAGNOSTICS' and p['position']==2 and p['transient'] and set(p['transient'])=={'theta','raw','candidate'}
                n=op.counts['optimizer_steps']
                with patch.object(e.common,'diagnose',return_value=dict(status='FAIL',checks={'injected':False})):rejected(lambda:e.train(**kw('split_'+arm,arm),resume=True))
                assert op.counts['optimizer_steps']==n;gc.collect()
                resumed=e.train(**kw('split_'+arm,arm),resume=True);gc.collect()
                for key in ('student_hash','EMA_hash','optimizer_hash','label_order_hash','U_order_hash'):assert full[key]==resumed[key],(arm,key)
                assert (base/'tasks'/('full_'+arm)/'steps.jsonl').read_text()==(root/'steps.jsonl').read_text()
                # Raw/candidate forensic states cannot be loaded as committed updates.
                for state in ('RAW_PROPOSAL','CANDIDATE','FORENSIC_ONLY_NOT_RESUMABLE'):
                    pp=dict(p,state=state);e.save(root/'latest.pt',pp);(root/'receipt.json').unlink(missing_ok=True)
                    rejected(lambda:e.train(**kw('split_'+arm,arm),resume=True));gc.collect()
                del p,pp;tests.append('pending_first_exact_resume_and_forbidden_phases_'+arm)
            # Genuine native synthetic parity, not historical baseline retraining.
            records=[]
            for arm in ('native',*c.ARMS):
                task=kw('parity',c.MAIN)['fixture']['task'];m,ema,opt,_=e.initial(base,task,reference,dev,True);record=[]
                for i in range(2):
                    if arm=='native':row=c.parent.step(m,ema,opt,lb,None,'T_LCTX',61,c.DOMAINS[1],20,i,Counter(),['p0','p1'])
                    else:row,_=c.step(m,ema,opt,lb,None,dict(task,arm=arm),20,i,Counter(),['p0','p1'])
                    record.append((c.state_hash(m),c.state_hash(ema),c.optimizer_hash(opt),row['loss']))
                records.append(record);del m,ema,opt;gc.collect()
            assert all(r==records[0] for r in records);tests.append('native_warmup_golden_hash_Adam_EMA_parity')
            task=kw('overfit',c.MAIN)['fixture']['task'];m,ema,opt,_=e.initial(base,task,reference,dev,True);losses=[]
            for i in range(4):losses.append(c.step(m,ema,opt,lb,None,task,1,i,Counter(),['p0','p1'])[0]['loss'])
            assert losses[-1]<losses[0];del m,ema,opt;gc.collect();tests.append('two_case_overfit')
            # Cold formal resolution and tail B1, entirely generated inputs.
            task=kw('cold',c.MAIN)['fixture']['task'];m,ema,opt,_=e.initial(base,task,reference,dev,True)
            big={k:torch.nn.functional.interpolate(v[:,None].float(),size=(384,384),mode='nearest')[:,0].to(v.dtype) if v.ndim==3 else torch.nn.functional.interpolate(v,size=(384,384),mode='nearest') for k,v in lb.items() if torch.is_tensor(v)}
            tail={k:v[:1] for k,v in big.items() if k in ('image','geometry')}
            row,_=c.step(m,ema,opt,big,tail,task,40,0,Counter(),['p0','p1']);assert all(row['checks'].values()) and row['response_images']==1
            del m,ema,opt;gc.collect();tests.append('cold384_tail_B1')
            assert op.counts['optimizer_steps']==26 and op.counts['sample_val']==0
        result=dict(status='PASS',source=source,tests=tests,formula=fr,algebra=alg,counterexample=guard,counts=dict(op.counts),synthetic_updates=op.counts['optimizer_steps'],real_updates=0,python=platform.python_version(),torch=torch.__version__,device=device,seconds=time.time()-started,overfit_loss=losses)
    except BaseException as exc:
        c.write_json(b/'qualification.json',dict(status='FAIL',source=source,tests=tests,error=repr(exc),counts=dict(op.counts) if 'op' in locals() else {},real_updates=0));raise
    c.write_json(b/'qualification.json',result);print(dict(status='PASS',updates=result['synthetic_updates'],tests=len(tests)),flush=True)

def smoke(output,base,data,reference,device):
    source=ct.verify();ct.neutral_subprocess_paths();b=Path(output);b.mkdir();dev=torch.device(device);rows=[];peak=reserved=0
    with Operations(b/'operations',update_cap=8) as op:
        for task in [t for t in ct.protocol()['tasks'] if t['seed']==61]:
            with c.training_access(task['domain']):
                ds=c.DomainData(data,task['domain'],'train_labeled');patients=c.patients_for(data,task['domain'],(c.MANIFEST_SHA,c.SPLIT_SHA))
                torch.cuda.init();torch.cuda.reset_peak_memory_stats(dev);m,ema,opt,bound=e.initial(base,task,reference,dev)
                lb=c.batch(ds,[0,1],dev,61,40,0,'labeled');probe={k:lb[k] for k in ('image','geometry')}
                for i in range(2):
                    row,tmp=c.step(m,ema,opt,lb,probe,task,40,i,Counter(),patients[:2],diagnostic=i==1)
                    assert e.diagnose(m,ema,bound,row)['status']=='PASS',row
                    if tmp:c.actual_diagnostic(m,lb,probe,task,40,i,Counter(),tmp,row)
                peak=max(peak,torch.cuda.max_memory_allocated(dev));reserved=max(reserved,torch.cuda.max_memory_reserved(dev));rows.append(dict(arm=task['arm'],domain=task['domain'],loss=row['loss'],response_norm=row['raw_response_norm'],numerics=row['checks']))
                del m,ema,opt,lb,probe,tmp;gc.collect();torch.cuda.empty_cache()
    import math
    result=dict(status='PASS',source=source,counts=dict(op.counts),real_updates=8,discarded=True,rows=rows,peak_allocated=peak,peak_reserved=reserved,admission_mib=math.ceil(max(peak,reserved)/2**20*1.25+768),U_images=0,val_GT=0,scope='L-only qualification; stripped current L only for discarded query smoke')
    c.write_json(b/'receipt.json',result);print(dict(status='PASS',updates=8,admission_mib=result['admission_mib']),flush=True)

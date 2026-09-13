"""Bounded qualification of labels, shared inputs, isolation, whitelist and recovery."""
import gc,math,os,tempfile,time
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import torch
from experiments.lcrseg.lctx_paper_closeout_v1 import core as c,contract as ct,engine as e
from experiments.lcrseg.ssl_anchored_mix_v0_1.telemetry import Operations

def reject(fn):
    try:fn()
    except (ValueError,PermissionError,RuntimeError,FloatingPointError):return
    raise AssertionError('expected rejection')

def labels(device):
    y=torch.tensor([[[0,1,255],[2,2,1]],[[2,0,1],[255,1,2]]],device=device)
    m=torch.tensor([[[1,0,1],[0,1,0]],[[0,1,0],[1,0,1]]],device=device,dtype=torch.bool)
    yy=c.mixed_labels(y,m)
    assert torch.equal(yy[:2],torch.where(m,y,y.flip(0))) and torch.equal(yy[2:],torch.where(m,y.flip(0),y))
    for cls in (0,1,2,255):assert int((yy==cls).sum())==2*int((y==cls).sum())
    z=torch.arange(72,device=device,dtype=torch.float32).reshape(4,3,2,3)/23;logp=z.log_softmax(1)
    ce,dice=c.parent.mathcore.supervised_parts(logp,yy);v=yy!=255
    manual_ce=-logp.permute(0,2,3,1)[v].gather(1,yy[v][:,None]).mean()
    parts=[]
    for i in range(4):
        for cls in (1,2):
            pp=logp[i,cls].exp().double()*v[i];tt=(yy[i]==cls).double()*v[i]
            parts.append(1-(2*(pp*tt).sum()+1e-5)/(pp.sum()+tt.sum()+1e-5))
    assert torch.allclose(ce,manual_ce) and torch.allclose(dice,torch.stack(parts).mean())
    zz=logp.clone();zz.permute(0,2,3,1)[~v]=torch.tensor([-.2,-.3,-.4],device=device)
    ce2,di2=c.parent.mathcore.supervised_parts(zz,yy);assert torch.equal(ce,ce2) and torch.equal(dice,di2)
    a,b=c.parent.mathcore.supervised_parts(logp,torch.full_like(yy,255));assert a==b==0
    reject(lambda:c.mixed_labels(y.float(),m))
    for n in (5,8):
        pairs=c.orders(n,71,c.DOMAINS[1],21,'labeled_order',21)
        assert all(len(p)==len(set(p))==2 for p in pairs) and set(sum(pairs,[]))==set(range(n))

def run(output,reference,device):
    source=ct.verify();ct.neutral_subprocess_paths();out=Path(output);out.mkdir();dev=torch.device(device)
    os.environ['SF_REFERENCE']=reference;os.environ['SF_DEVICE']=device
    from experiments.lcrseg.tests.ams_seq_transfer_v0_1.test_contract import fixture
    tests=[];started=time.time()
    try:
        with Operations(out/'operations',update_cap=128) as op,tempfile.TemporaryDirectory() as tmp:
            labels(dev);tests.append('complementary_integer_labels_ignore_CE_Dice_normalization_odd_pairing')
            base=Path(tmp);data=base/'data';expected=fixture(data);sd,td=c.DOMAINS
            def kw(tid,arm):
                t=dict(task_id=tid,source_task_id='source',seed=71,order='O1',source_domain=sd,domain=td,arm=arm,updates=2,steps_per_epoch=1,actual_diagnostic_positions=[])
                return dict(base=base,task_id=tid,data=data,reference=reference,device=dev,fixture=dict(task=t,epochs=[20,21],steps=1,expected=expected))
            records={}
            for arm in c.ARMS:
                r=e.train(**kw(arm,arm));gc.collect();records[arm]=r
                assert r['U_opens']==0 and r['counts'].get('response_VJP',0)==0 and r['boundary']['trainable_parameters']==438192
                assert r['boundary']['trainable_names']==list(c.LAYERS) and r['fixed_subset_unchanged']
            tests.append('five_recipes_whitelist_zero_U_finite_and_deploy')
            # Stop after the durable final state, before diagnostics/log adoption.
            native=e.common.pending
            def pause(*a,**kw):
                if a[4]['position']==2:raise RuntimeError('injected final pending interruption')
                return native(*a,**kw)
            with patch.object(e.common,'pending',pause):reject(lambda:e.train(**kw('resume','C_FULLMIX')))
            gc.collect();before=op.counts['optimizer_steps'];rr=e.train(**kw('resume','C_FULLMIX'),resume=True);gc.collect()
            assert op.counts['optimizer_steps']==before
            for key in ('student_hash','EMA_hash','optimizer_hash','label_order_hash','counts','L_opens'):assert rr[key]==records['C_FULLMIX'][key]
            e.train(**kw('mid','C_FULLMIX'),stop_after=1);gc.collect();rr=e.train(**kw('mid','C_FULLMIX'),resume=True);gc.collect()
            for key in ('student_hash','EMA_hash','optimizer_hash','label_order_hash','counts','L_opens'):assert rr[key]==records['C_FULLMIX'][key]
            tests.append('mid_update_and_final_pending_exact_recovery_no_repeated_final_update')
            ds=c.DomainData(data,td,'train_labeled',expected=expected,shape=(24,24));lb=c.batch(ds,[0,1],dev,71,21,0,'labeled')
            inputs=[];loss=[]
            for full in (False,True):
                t=kw('manual','C_FULLMIX' if full else 'C_LCTX_LOW')['fixture']['task'];m,ema,opt,_=e.initial(base,t,reference,dev,True)
                calls=[];fwd=c.parent.mathcore.fwd
                def capture(model,x,*a,**kw):
                    z=fwd(model,x,*a,**kw);calls.append((x.detach().clone(),z[0].detach().clone()));return z
                with patch.object(c.parent.mathcore,'fwd',capture):row,_=c.step(m,ema,opt,lb,None,t,21,0,Counter(),['a','b'])
                inputs.append([x for x,z in calls]);loss.append(row['loss'])
                if full:
                    mask,_=c.rectangles(2,24,24,(71,td,21,0),dev)
                    parts=c.parent.mathcore.supervised_parts(torch.cat([z.log_softmax(1) for x,z in calls]),c.mixed_labels(lb['label'],mask))
                    assert abs(row['loss']-float(sum(parts)))<1e-7
                del m,ema,opt,calls;gc.collect()
            assert len(inputs[0])==len(inputs[1])==2 and all(torch.equal(a,b) for a,b in zip(*inputs))
            tests.append('identical_native_LCTX_FULLMIX_inputs_single_full_supervised_parts')
            selected=[dict(case_id=f'{td}_train_labeled_{i}',patient_id=f'{td}_train_labeled_{i}') for i in (0,2)]
            q=kw('low_subset','C_FULLMIX_LOW');q['fixture']['subset']=selected
            r=e.train(**q);gc.collect();assert r['L_opens']==4 and r['L_patients']==2
            with c.selected_data([r['case_id'] for r in selected]),c.training_access(td):
                filtered=c.DomainData(data,td,'train_labeled',expected=expected,shape=(24,24))
                assert [r['case_id'] for r in filtered._dataset.rows]==[r['case_id'] for r in selected]
                reject(lambda:c.DomainData(data,td,'train_unlabeled',expected=expected))
                reject(lambda:c.DomainData(data,sd,'train_labeled',expected=expected))
                reject(lambda:c.DomainData(data,td,'val',evaluator=True,expected=expected))
            tests.append('nested_subset_before_open_source_U_evaluation_isolation')
            t=kw('overfit','C_CED')['fixture']['task'];m,ema,opt,_=e.initial(base,t,reference,dev,True)
            losses=[c.step(m,ema,opt,lb,None,t,1,i,Counter(),['a','b'])[0]['loss'] for i in range(4)]
            assert losses[-1]<losses[0];del m,ema,opt;gc.collect();tests.append('two_case_overfit')
            assert op.counts['sample_val']==op.counts['sample_train_unlabeled']==op.counts['autograd_grad']==0
        result=dict(status='PASS',source=source,tests=tests,synthetic_updates=op.counts['optimizer_steps'],counts=dict(op.counts),seconds=time.time()-started,device=device,overfit_losses=losses)
    except BaseException as exc:
        c.write_json(out/'qualification.json',dict(status='FAIL',source=source,tests=tests,error=repr(exc),synthetic_updates=op.counts['optimizer_steps'] if 'op' in locals() else 0));raise
    c.write_json(out/'qualification.json',result);print(result,flush=True)

def smoke(base,reference,device):
    source=ct.verify();ct.neutral_subprocess_paths();b=Path(base);out=b/'smoke';out.mkdir();inputs=ct.read(b/'private_inputs.json');subsets=ct.read(b/'private_subsets.json');dev=torch.device(device);peak=0;rows=[]
    torch.cuda.init()
    with Operations(out/'operations',update_cap=12) as op:
        for domain in c.DOMAINS:
            for arm in c.ARMS:
                task=next(t for t in ct.protocol()['tasks'] if t['domain']==domain and t['arm']==arm and t['seed']==71)
                selected=subsets[domain][task['budget']]
                with c.selected_data([r['case_id'] for r in selected]),c.training_access(domain):
                    ds=c.DomainData(inputs['data'],domain,'train_labeled');torch.cuda.reset_peak_memory_stats(dev)
                    m,ema,opt,_=e.initial(b,task,reference,dev);l=c.batch(ds,[0,1],dev,71,21,0,'labeled')
                    row,_=c.step(m,ema,opt,l,None,task,21,0,Counter(),[r['patient_id'] for r in selected[:2]])
                    assert all(row['checks'].values()) and e.diagnose(m,ema,{},row)['status']=='PASS'
                    rows.append(dict(domain=domain,arm=arm,loss=row['loss'],checks=row['checks']))
                    peak=max(peak,torch.cuda.max_memory_reserved(dev));del m,ema,opt,l,ds;gc.collect();torch.cuda.empty_cache()
        assert op.counts['optimizer_steps']==10 and op.counts['sample_train_unlabeled']==op.counts['sample_val']==op.counts['autograd_grad']==0
    r=dict(status='PASS',source=source,discarded=True,real_updates=10,counts=dict(op.counts),rows=rows,peak_reserved=peak,admission_mib=max(3072,math.ceil(peak/2**20*1.25+768)))
    c.write_json(out/'receipt.json',r);print(r,flush=True)

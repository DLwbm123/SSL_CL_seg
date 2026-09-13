"""Only new lineage, native-baseline and runner integration checks; no math redesign."""
import gc,os,tempfile,time,shutil
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import torch
from experiments.lcrseg.dpr_sign_replication_v0_1 import core as c,contract as ct,engine as e
from experiments.lcrseg.ssl_anchored_mix_v0_1.telemetry import Operations
from experiments.lcrseg.tests.dpr_v0_1.qualification import rejected

def run(output,reference,device):
    source=ct.verify();ct.neutral_subprocess_paths();out=Path(output);out.mkdir();dev=torch.device(device);os.environ['SF_REFERENCE']=reference;os.environ['SF_DEVICE']=device
    from experiments.lcrseg.tests.ams_seq_transfer_v0_1.test_contract import fixture
    tests=[];started=time.time()
    try:
        with Operations(out/'operations',update_cap=32) as op,tempfile.TemporaryDirectory() as temp:
            base=Path(temp);data=base/'data';ex=fixture(data);sd,td=c.DOMAINS
            st=dict(task_id='source',source_task_id=None,seed=71,order='O1',source_domain=sd,domain=sd,arm='SRC_CE',updates=2)
            def skw(task):return dict(base=base,task_id=task['task_id'],data=data,reference=reference,device=dev,fixture=dict(task=task,epochs=1,shape=(24,24),expected=ex))
            sr=e.train(**skw(st));gc.collect();part=dict(st,task_id='source_split')
            e.train(**skw(part),stop_after=1);gc.collect();rr=e.train(**skw(part),resume=True);gc.collect()
            for key in ('student_hash','EMA_hash','optimizer_hash','label_order_hash'):assert sr[key]==rr[key]
            tests.append('fresh_SRC_CE_exact_resume')
            def kw(tid,arm):
                task=dict(task_id=tid,source_task_id='source',seed=71,order='O1',source_domain=sd,domain=td,arm=arm,updates=2,steps_per_epoch=1,actual_diagnostic_positions=[2] if arm==c.MAIN else [])
                return dict(base=base,task_id=tid,data=data,reference=reference,device=dev,fixture=dict(task=task,epochs=[20,21],steps=1,expected=ex,U_count=3))
            fr=e.train(**kw('O1_s71_F_CONV','F_CONV'));gc.collect();sg=e.train(**kw('O1_s71_'+c.MAIN,c.MAIN));gc.collect()
            assert fr['U_opens']==0 and fr['counts'].get('response_VJP',0)==0 and sg['counts']['response_VJP']==2
            for r in (fr,sg):assert r['boundary']['source_student_hash']==sr['student_hash'] and r['boundary']['trainable_parameters']==438192
            assert ct.read(base/'tasks/O1_s71_F_CONV/warmup.json')==ct.read(base/('tasks/O1_s71_'+c.MAIN)/'warmup.json')
            tests.append('new_source_fork_native_F_CONV_zero_U_exact_warmup')
            (base/'tasks/split_F_CONV').mkdir();shutil.copy(base/'tasks/O1_s71_F_CONV/warmup.json',base/'tasks/split_F_CONV/warmup.json')
            native=e.pending
            def pause(*a,**k):
                if a[4]['position']==2:raise RuntimeError('injected final-pending interruption')
                return native(*a,**k)
            with patch.object(e,'pending',pause):rejected(lambda:e.train(**kw('split_'+c.MAIN,c.MAIN)))
            gc.collect();n=op.counts['optimizer_steps'];root=base/('tasks/split_'+c.MAIN)
            with patch.object(e.target,'diagnose',return_value=dict(status='FAIL',checks={'injected':False})):rejected(lambda:e.train(**kw('split_'+c.MAIN,c.MAIN),resume=True))
            gc.collect();assert op.counts['optimizer_steps']==n
            rr=e.train(**kw('split_'+c.MAIN,c.MAIN),resume=True);gc.collect()
            for key in ('student_hash','EMA_hash','optimizer_hash','label_order_hash','U_order_hash'):assert sg[key]==rr[key]
            tests.append('final_pending_first_exact_resume')
            task=kw('manual',c.MAIN)['fixture']['task'];m,ema,opt,bound=e.initial(base,task,reference,dev,True);assert c.state_hash(m)==sr['student_hash']
            ds=c.DomainData(data,td,'train_labeled',expected=ex,shape=(24,24));l=c.batch(ds,[0,1],dev,71,40,0,'labeled');u={k:l[k] for k in ('image','geometry')}
            records=[]
            for new in (False,True):
                if new:m,ema,opt,bound=e.initial(base,task,reference,dev,True)
                row,_=(c.step if new else c.sign.step)(m,ema,opt,l,u,task,40,0,Counter(),['p0','p1']);assert all(row['checks'].values())
                records.append((c.state_hash(m),c.state_hash(ema),c.optimizer_hash(opt),row));del m,ema,opt;gc.collect()
            assert records[0]==records[1];tests.append('unchanged_SIGN_kernel_exact_actual_update_parity')
            m,ema,opt,_=e.initial(base,dict(task,arm='F_CONV'),reference,dev,True);losses=[]
            for i in range(4):losses.append(c.step(m,ema,opt,l,None,dict(task,arm='F_CONV'),1,i,Counter(),['p0','p1'])[0]['loss'])
            assert losses[-1]<losses[0];del m,ema,opt;gc.collect();tests.append('two_case_overfit')
            payload=torch.load(root/'deploy_student.pt',map_location=dev,weights_only=False);m=c.from_state(reference,dev,payload['student'],c.LINEAR);assert c.state_hash(m)==rr['student_hash'];del m,payload;gc.collect()
            with c.training_access(td):
                rejected(lambda:c.DomainData(data,sd,'train_labeled',expected=ex));rejected(lambda:c.DomainData(data,td,'val',evaluator=True,expected=ex))
            tests.append('single_student_deploy_and_current_domain_capability')
            assert op.counts['optimizer_steps']==16 and op.counts['sample_val']==0
        result=dict(status='PASS',source=source,tests=tests,counts=dict(op.counts),synthetic_updates=16,real_updates=0,device=device,torch=torch.__version__,seconds=time.time()-started,overfit_loss=losses)
    except BaseException as err:
        c.write_json(out/'qualification.json',dict(status='FAIL',source=source,tests=tests,error=repr(err),counts=dict(op.counts) if 'op' in locals() else {}));raise
    c.write_json(out/'qualification.json',result);print(result,flush=True)

def smoke(output,data,reference,device):
    source=ct.verify();ct.neutral_subprocess_paths();out=Path(output);out.mkdir();dev=torch.device(device);peak=0;rows=[]
    with Operations(out/'operations',update_cap=8) as op:
        for domain in c.DOMAINS:
            with c.training_access(domain):
                ds=c.DomainData(data,domain,'train_labeled');patients=c.patients_for(data,domain,(c.MANIFEST_SHA,c.SPLIT_SHA))
                for arm in c.ARMS:
                    torch.cuda.reset_peak_memory_stats(dev);m=c.parent.build(reference,dev,71,domain,'SUP_CE');c.configure(m,'F_CONV',None);ema=c.ema_from(m);opt=c.optimizer_for(m)
                    l=c.batch(ds,[0,1],dev,71,40,0,'labeled');u={k:l[k] for k in ('image','geometry')} if arm==c.MAIN else None
                    task=dict(seed=71,domain=domain,arm=arm)
                    for i in range(2):
                        row,_=c.step(m,ema,opt,l,u,task,40,i,Counter(),patients[:2]);assert all(row['checks'].values())
                    peak=max(peak,torch.cuda.max_memory_reserved(dev));rows.append(dict(arm=arm,domain=domain,checks=row['checks'],loss=row['loss']))
                    del m,ema,opt,l,u;gc.collect();torch.cuda.empty_cache()
    import math
    r=dict(status='PASS',source=source,real_updates=8,discarded=True,source_models='fresh initialization only; no historical checkpoint',counts=dict(op.counts),rows=rows,peak_reserved=peak,admission_mib=max(3072,math.ceil(peak/2**20*1.25+768)),U_accesses=0,val_GT=0)
    c.write_json(out/'receipt.json',r);print(r,flush=True)

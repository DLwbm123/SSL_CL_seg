"""Bounded exact-source qualification; synthetic data only except12 discarded L smoke steps."""
import gc,os,time,tempfile,platform
from pathlib import Path
from collections import Counter
from unittest.mock import patch
import torch
from experiments.lcrseg.cist_plasticity_v0_2 import core as c,engine as e,contract as ct
from experiments.lcrseg.ssl_anchored_mix_v0_1.telemetry import Operations
from experiments.lcrseg.tests.cist_v0_1.qualification import rejected

def run(output,reference,device):
    source=ct.verify();ct.neutral_subprocess_paths();b=Path(output);b.mkdir();dev=torch.device(device)
    os.environ['SF_REFERENCE']=reference;os.environ['SF_DEVICE']=device
    from experiments.lcrseg.tests.ams_seq_transfer_v0_1.test_contract import fixture
    tests=[];started=time.time()
    try:
        with Operations(b/'operations',update_cap=32) as op,tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);data=base/'data';expected=fixture(data)
            def kw(tid,arm,epochs=(20,21)):
                t=dict(task_id=tid,source_task_id='synthetic',seed=61,order='O1',source_domain=c.DOMAINS[0],domain=c.DOMAINS[1],arm=arm,updates=len(epochs),steps_per_epoch=1)
                return dict(base=base,task_id=tid,data=data,reference=reference,device=dev,fixture=dict(task=t,epochs=list(epochs),steps=1,expected=expected))
            ds=c.DomainData(data,c.DOMAINS[1],'train_labeled',expected=expected,shape=(24,24));lb=c.batch(ds,[0,1],dev,61,21,0,'labeled')
            for arm in c.ARMS:
                args=kw('manual',arm);m,opt,entry,bound=e.initial(base,args['fixture']['task'],reference,dev,args['fixture'])
                with torch.no_grad():
                    native,h=m.core(lb['image'],stochastic_classifier=False);z,(h2,hp,dt)=m(lb['image']);assert torch.equal(h,hp) and torch.equal(native,z)
                assert sum(p.numel() for p in m.parameters() if p.requires_grad)==c.SIZES[arm]
                initial=m.dynamic_state();initial={n:p.clone() for n,p in initial.items()}
                for i,epoch in enumerate((1,21)):
                    r=c.step(m,opt,lb,args['fixture']['task'],epoch,i,Counter(),['p0','p1'])
                    assert r['labeled_source_scoring_multiplicity']==1 and r['U_image_records']==r['donor_extra_image_reads']==r['EMA_updates']==0
                    assert any(v and v>0 for n,v in r['gradient_norms'].items() if n.startswith('core.'))
                assert any(not torch.equal(v,m.dynamic_state()[n]) for n,v in initial.items() if n.startswith('core.'))
                assert e.diagnose(m,bound)['status']=='PASS'
                # A learned controller still passes a gradient through context and a frozen readout into current H.
                if m.transport is not None:
                    h=torch.randn(2,16,8,8,device=dev,requires_grad=True);hp,detail=m.transport(h);hp.square().mean().backward();assert h.grad.norm()>0
                    assert c.mechanism(h,hp,detail)['distance_relative_max']<=c.DISTANCE32_LIMIT
                tests.append('identity_whitelist_gradient_geometry_'+arm);del m,opt,initial;gc.collect()
                with c.training_access(c.DOMAINS[1]):
                    rejected(lambda:c.DomainData(data,c.DOMAINS[1],'train_unlabeled',expected=expected))
                    rejected(lambda:c.DomainData(data,c.DOMAINS[0],'train_labeled',expected=expected))
                    rejected(lambda:c.DomainData(data,c.DOMAINS[1],'val',evaluator=True,expected=expected))
                full=e.train(**kw('full_'+arm,arm));gc.collect()
                with patch.object(e,'pending',side_effect=RuntimeError('injected post-save pause')):rejected(lambda:e.train(**kw('split_'+arm,arm)))
                gc.collect();root=base/'tasks'/('split_'+arm);p=torch.load(root/'latest.pt',map_location='cpu',weights_only=False)
                assert p['position']==1 and p['state']=='POST_UPDATE_PENDING_DIAGNOSTICS' and any(k.startswith('core.') for k in p['dynamic'])
                n=op.counts['optimizer_steps']
                with patch.object(e,'diagnose',return_value=dict(status='FAIL',checks={'injected':False})):rejected(lambda:e.train(**kw('split_'+arm,arm),resume=True))
                gc.collect();assert op.counts['optimizer_steps']==n
                resumed=e.train(**kw('split_'+arm,arm),resume=True);gc.collect()
                for k in ('student_hash','optimizer_hash','controller_hash','label_order_hash','L_opens','U_opens'):assert full[k]==resumed[k],(arm,k)
                assert (base/'tasks'/('full_'+arm)/'steps.jsonl').read_text()==(root/'steps.jsonl').read_text()
                payload=torch.load(root/'deploy_student.pt',map_location=dev,weights_only=False)
                core=c.from_state(reference,dev,payload.pop('core'),c.LINEAR);tensors,_=c.readout_basis(core.decoder.conv_logit.mu.weight,61,c.DOMAINS[0])
                reloaded=c.Student(core,tensors,args['fixture']['task'])
                if reloaded.transport is not None:reloaded.transport.load_state_dict(payload.pop('controller'))
                assert c.state_hash(reloaded)==resumed['student_hash'] and c.state_hash(core)!=bound['source_student_hash']
                del reloaded,core,p,payload;gc.collect();tests.append('full_dynamic_pending_resume_and_deploy_'+arm)
            # Compare a truly fixed identity bypass against the unmodified native F_CONV optimizer step.
            from experiments.lcrseg.lctx_weight_memory_v0_1 import core as native
            args=kw('parity',c.MAIN);task=args['fixture']['task'];m,opt,entry,bound=e.initial(base,task,reference,dev,args['fixture']);m.bypass=True;m.transport.requires_grad_(False);opt=c.optimizer(m)
            r=c.step(m,opt,lb,task,21,0,Counter(),['p0','p1']);result=(c.state_hash(m.core),c.optimizer_hash(opt),r['loss'],r['CE'],r['GT_Dice']);del m,opt;gc.collect()
            m=c.parent.build(reference,dev,61,c.DOMAINS[0],'SUP_CE');native.configure(m,'F_CONV',None);teacher=c.ema_from(m);opt=c.optimizer_for(m)
            r=native.step(m,teacher,opt,lb,61,c.DOMAINS[1],21,0,Counter(),['p0','p1'],'F_CONV')
            assert result[:2]==(c.state_hash(m),c.optimizer_hash(opt));assert result[2]==r['loss'],(result,r)
            assert r['labeled_source_scoring_multiplicity']==1 and r['donor_extra_image_reads']==0
            del m,teacher,opt;gc.collect();tests.append('fixed_identity_bypass_native_F_CONV_student_Adam_loss_exact')
            args=kw('overfit',c.MAIN);m,opt,_,_=e.initial(base,args['fixture']['task'],reference,dev,args['fixture']);losses=[]
            for i in range(8):losses.append(c.step(m,opt,lb,args['fixture']['task'],1,i,Counter(),['p0','p1'])['loss'])
            assert losses[-1]<losses[0];del m,opt;gc.collect();tests.append('two_case_overfit')
            assert op.counts['optimizer_steps']==28 and op.counts['sample_train_unlabeled']==op.counts['sample_val']==0
        result=dict(status='PASS',source=source,tests=tests,counts=dict(op.counts),synthetic_updates=28,real_updates=0,
                    baseline_parity_EMA_only=op.counts['ema_updates'],python=platform.python_version(),torch=torch.__version__,device=device,seconds=time.time()-started,overfit_loss=losses)
    except BaseException as exc:
        c.write_json(b/'qualification.json',dict(status='FAIL',source=source,tests=tests,error=repr(exc),counts=dict(op.counts),real_updates=0));raise
    c.write_json(b/'qualification.json',result);print(dict(status='PASS',updates=28,tests=len(tests)),flush=True)

def smoke(output,base,data,reference,device):
    source=ct.verify();ct.neutral_subprocess_paths();b=Path(output);b.mkdir();dev=torch.device(device);rows=[];peak=reserved=0
    with Operations(b/'operations',update_cap=12) as op:
        for task in [t for t in ct.protocol()['tasks'] if t['seed']==61]:
            with c.training_access(task['domain']):
                ds=c.DomainData(data,task['domain'],'train_labeled');patients=c.patients_for(data,task['domain'],(c.MANIFEST_SHA,c.SPLIT_SHA))
                torch.cuda.init();torch.cuda.reset_peak_memory_stats(dev);m,opt,entry,bound=e.initial(base,task,reference,dev)
                lb=c.batch(ds,[0,1],dev,61,21,0,'labeled')
                for i in range(2):r=c.step(m,opt,lb,task,21,i,Counter(),patients[:2])
                diag=e.diagnose(m,bound);assert diag['status']=='PASS'
                peak=max(peak,torch.cuda.max_memory_allocated(dev));reserved=max(reserved,torch.cuda.max_memory_reserved(dev))
                rows.append(dict(arm=task['arm'],domain=task['domain'],loss=r['loss'],trainable_parameters=bound['trainable_parameters'],whitelist=bound['whitelist'],fixed_subset_unchanged=True))
                del m,opt,lb;gc.collect();torch.cuda.empty_cache()
    import math
    result=dict(status='PASS',source=source,counts=dict(op.counts),real_updates=12,discarded=True,rows=rows,peak_allocated=peak,peak_reserved=reserved,admission_mib=math.ceil(max(peak,reserved)/2**20*1.25+768),U_images=0,val_GT=0)
    c.write_json(b/'receipt.json',result);print(dict(status='PASS',updates=12,admission_mib=result['admission_mib']),flush=True)

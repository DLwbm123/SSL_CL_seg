"""Rebuild image-held-out states, then conditionally test a frozen conflict rule."""
import copy
import fcntl
import importlib.util
import json
import os
import random
import time
import traceback
from collections import Counter
from pathlib import Path
import numpy as np
import torch

CAPS=dict(entries=3200,qualification=24,prior=32000)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def configure(cfg):
    global N,c,b,S,P,M,A,Q
    v=load('parent',cfg['v111_entry']);v.configure(cfg);N=v.N;c=N.c;b=N.b;N.STAGE='V123';c.CAPS.update(CAPS)
    S=load('selector',cfg['selector_entry']);P=load('target',cfg['target_entry'])
    M=load('metrics',cfg['metrics_entry']);A=load('analysis',cfg['analysis_entry']);Q=load('quarter',cfg['quarter_entry'])


def fold_roles(roles,fold,n):
    out=copy.deepcopy(roles);a=roles['A_fit'];assert len(a)==8 and fold in (0,1) and n in (2,4)
    fit=a[4*fold:4*fold+4];held=a[4*(1-fold):4*(1-fold)+4]
    out['A_fit']=fit[:n];out['A_hold']=held
    assert len(out['A_fit'])==n and len(held)==4 and not set(held)&set(out['A_fit']+out['M_fit']+out['U_adapt']+out['U_memory'])
    return out


def conflict(base,q,mask):
    old=base.argmax(1);new=q.detach().argmax(1)
    return mask.to(base.device)&(((old==1)&(new==0))|((old==2)&(new==1)))


def intervention(p,q,target,valid,mask,mode):
    selected=mask.to(p.device);chosen=conflict(target,q,selected)
    out=P.permute(target,q.argmax(1),chosen) if mode=='RULE' else q.detach() if mode=='EMA_ONLY' else target.detach()
    if mode in ('BASE','GOLD','RULE','EMA_ONLY'):loss=S.selected_loss(p,q,out,valid,mask)
    else:
        assert mode in ('IGNORE_C','OFF')
        admitted=valid.bool()&(q.detach().max(1).values>.7)
        raw=q.new_tensor([.5,1.,1.5])[q.detach().argmax(1)]
        weight=raw*selected;weight=weight*admitted.sum()/weight.sum().clamp_min(1e-8)
        # Keep pre-removal normalization so ignored mass is not reassigned elsewhere.
        keep=~chosen if mode=='IGNORE_C' else torch.zeros_like(chosen)
        loss=((out*(out.clamp_min(1e-8).log()-p.float().clamp_min(1e-8).log())).sum(1)*weight*keep).sum()/valid.sum().clamp_min(1)
    raw=q.new_tensor([.5,1.,1.5])[q.detach().argmax(1)]
    mass=float((raw*selected).sum());removed=float((raw*chosen).sum()) if mode=='IGNORE_C' else mass if mode=='OFF' else 0.
    detail=dict(selected=int(selected.sum()),conflict=int(chosen.sum()),selected_raw_weight=mass,removed_raw_weight=removed,
                removed_normalized_weight=removed*int((valid&(q.detach().max(1).values>.7)).sum())/max(mass,1e-8))
    return loss,out,detail


def selfcheck():
    A.selfcheck();Q.selfcheck()
    roles=dict(A_fit=list(range(8)),M_fit=list(range(8,24)),U_adapt=list(range(24,104)),U_memory=list(range(104,184)))
    for f in (0,1):
        for n in (2,4):assert len(fold_roles(roles,f,n)['A_fit'])==n
    logits=torch.tensor([.2,.3,.5]).reshape(1,3,1,1).expand(1,3,4,4).clone().requires_grad_();p=logits.softmax(1)
    q=torch.tensor([.8,.15,.05]).reshape(1,3,1,1).expand_as(p).clone().requires_grad_()
    target=torch.tensor([.15,.8,.05]).reshape(1,3,1,1).expand_as(p).clone().requires_grad_()
    valid=torch.ones_like(p[:,0],dtype=torch.bool);mask=valid.clone();mask[:,:,2:]=False
    gold=S.selected_loss(p,q,target,valid,mask);base=intervention(p,q,target,valid,mask,'BASE')[0]
    assert torch.equal(gold,base) and torch.equal(torch.autograd.grad(gold,logits,retain_graph=True)[0],torch.autograd.grad(base,logits,retain_graph=True)[0])
    for mode in A.METHODS:
        loss,out,detail=intervention(p,q,target,valid,mask,mode)
        assert torch.isfinite(loss) and not out.requires_grad
        gradients=torch.autograd.grad(loss,(logits,q,target),retain_graph=True,allow_unused=True)
        assert torch.isfinite(gradients[0]).all() and gradients[1:]==(None,None)
        if mode=='RULE':assert torch.equal(out.sort(1).values,target.detach().sort(1).values) and torch.equal(out.masked_select(~mask[:,None]),target.detach().masked_select(~mask[:,None]))
        if mode in ('IGNORE_C','OFF'):assert float(loss)==0 and not gradients[0].any() and detail['removed_raw_weight']==detail['selected_raw_weight']
        assert intervention(p,q,target,valid&False,mask&False,mode)[0]==0
    partial=target.detach().clone();partial[:,:,:,0]=q.detach()[:,:,:,0]
    ignore=intervention(p,q,partial,valid,mask,'IGNORE_C')[0]
    remain=mask&~conflict(partial,q,mask)
    renormalized=S.selected_loss(p,q,partial,valid,remain)
    assert torch.allclose(ignore,renormalized*.5) and not torch.allclose(ignore,renormalized)
    assert not conflict(q,q,mask).any()
    return dict(status='PASS',model_forwards=0,image_reads=0,optimizer_calls=0,checks=['fold exclusion and true n','BASE loss and gradient exact','RULE permutation and unselected identity','teacher detach','IGNORE_C fixed denominator','OFF and empty differentiable zero','quality and strong-control gates'])


def worker(root,cfg):
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.worker import scores
    torch.set_num_threads(2);torch.cuda.set_device(0)
    original=N.E.install_actions(c);N.E.action_check(c,original);transfer=c.transfer_loss
    roles=c.split_roles(cfg['data']);assert roles==N.D.read(Path(cfg['action_root'])/'ROLES.private.json')
    ctx=A.contexts()[cfg.get('context_index',0)];stream=cfg.get('stream',3);r=fold_roles(roles,ctx['fold'],ctx['labeled_images'])
    ledger=b.JobLedger(root,cfg['caps']);counts=Counter();allowed=set(r['A_fit']);u_allowed=True
    sample=c.e.primitive.CurrentData.__getitem__
    def guarded(ds,i):
        assert ds.role in ('train_labeled','train_unlabeled')
        assert ds.rows[i]['case_id'] in (allowed if ds.role=='train_labeled' else set(r['U_adapt']) if u_allowed else set())
        counts['image_attempts_'+ds.role]+=1;v=sample(ds,i);counts['image_success_'+ds.role]+=1;return v
    c.e.primitive.CurrentData.__getitem__=guarded
    random.seed(123000+stream);np.random.seed(123000+stream);torch.manual_seed(123000+stream);torch.cuda.manual_seed_all(123000+stream)
    condition=(ctx['condition'],ctx['factor']);native_ctx=(ctx['source_step'],ctx['labeled_images'],condition,'heldout')
    t=b.make(cfg,r,ledger,native_ctx,300);t.provider.seed=10168+10000*stream
    random.seed(123000+stream);np.random.seed(123000+stream);torch.manual_seed(123000+stream);torch.cuda.manual_seed_all(123000+stream)
    assert t.step==0 and set(t.provider.semantic_metadata()['fit_cases'])==set(r['A_fit'])
    initial_memory=c.e.cpu(t.memory.state_dict());profile={'mode':'WARM'}
    for name,model in [('student',t.model),('ema',t.ema),('memory',t.memory)]:
        def hook(module,args,name=name):counts[name+'_image_forwards']+=len(args[0])
        model.register_forward_pre_hook(hook)
    step=t.optimizer.step
    def rate_step(*args,**kwargs):
        rates=[g['lr'] for g in t.optimizer.param_groups];scale=1. if t.step<100 else .25
        counts['rate_records']+=1;c.e.append(root/'RATE_LEDGER.jsonl',dict(ordinal=counts['rate_records'],key=t.key,category=t.category,step=t.step,scale=scale,base=rates,effective=[a*scale for a in rates]))
        try:
            for g,lr in zip(t.optimizer.param_groups,rates):g['lr']=lr*scale
            return step(*args,**kwargs)
        finally:
            for g,lr in zip(t.optimizer.param_groups,rates):g['lr']=lr
    rate_step._wrapped_by_lr_sched=True;t.optimizer.step=rate_step
    def selected(p,q,valid,action,qm=None,qflip=None):
        if profile['mode']=='WARM':assert action==0;return transfer(p,q,valid,action,qm,qflip)
        assert action==11
        _,stats,target=transfer(p,q,valid,action,qm,qflip)
        features,eligible,classes=S.features(p,q,qm,qflip,target,valid,t.step);mask,selection=S.choose(features,eligible,classes,'COVERAGE',None,None)
        loss,out,detail=intervention(p,q,target,valid,mask,profile['mode'])
        if profile['mode']=='GOLD':loss=S.selected_loss(p,q,target,valid,mask)
        counts['selector_calls']+=1;c.e.append(root/'SELECTION_LEDGER.jsonl',dict(ordinal=counts['selector_calls'],key=t.key,category=t.category,step=t.step,mode=profile['mode'],image_budgets=[(v['eligible'],v['selected']) for v in selection],**detail))
        return loss,stats,out
    c.transfer_loss=selected
    campaign=Path(cfg['campaign']);entryfile=campaign/f"jobs/entry{ctx['context']}_{stream}/ENTRY100.private.pt"
    def restore(path):
        state=torch.load(path,map_location='cpu',weights_only=False);c.restore(t,state);return state
    def freeze_check():
        assert c.e.same(initial_memory,t.memory.state_dict()) and all(p.grad is None for model in (t.memory,t.ema) for p in model.parameters())
    job=cfg['job']
    if job=='entry':
        t.action=0;t.category='entries';t.key=f"entry{ctx['context']}_{stream}"
        for _ in range(100):t.update()
        freeze_check();assert t.step==100 and dict(ledger.count)=={'entries':100}
        c.e.atomic_save(c.snapshot(t),root/'ENTRY100.private.pt')
        N.D.write(root/'PARTITION.private.json',dict(fit=r['A_fit'],held=r['A_hold'],source=r['M_fit']))
    elif job=='qualification':
        entry=restore(entryfile);gold=None
        for mode in ('GOLD','BASE'):
            c.restore(t,entry);t.action=11;t.category='qualification';t.key=mode;profile['mode']=mode
            for _ in range(2):t.update()
            now=c.snapshot(t)
            if gold is None:gold=now
            else:assert c.e.same(gold,now)
        for mode in A.METHODS:
            expected=None
            for repeat in range(2):
                c.restore(t,entry);t.action=11;t.category='qualification';t.key=mode;profile['mode']=mode
                for _ in range(2):t.update()
                freeze_check();now=c.snapshot(t)
                if repeat==0:
                    c.e.atomic_save(now,root/(mode+'_CHECK.private.pt'));expected=torch.load(root/(mode+'_CHECK.private.pt'),map_location='cpu',weights_only=False)
                else:assert c.e.same(expected,now),mode
        assert dict(ledger.count)=={'qualification':24}
        N.D.write(root/'QUALIFICATION.json',dict(status='PASS',native_updates=24,BASE_golden_exact=True,all_five_checkpoint_replays_exact=True,frozen_memory=True,teacher_gradients_none=True))
    elif job=='quality':
        assert N.D.read(campaign/'ENTRY_LOCK.json')['entries']==32
        entry=restore(entryfile);allowed=set(r['A_hold']);u_allowed=False
        ds=c.subset(c.e.primitive.CurrentData(cfg['data'],0,'train_labeled'),r['A_hold']);details=[]
        with t.readonly(),torch.no_grad():
            for j in range(len(ds)):
                item=ds[j];x=c.photo(item['image'][None].cuda(),condition);y=item['label'][None].cuda();valid=item['geometry'][None].cuda()&(y!=255)
                def forward(kind,fn):
                    counts['diagnostic_forward_attempts']+=1;rec=dict(ordinal=counts['diagnostic_forward_attempts'],image_index=j,kind=kind)
                    c.e.append(root/'FORWARD_LEDGER.jsonl',dict(event='attempt',**rec))
                    result=fn();c.e.append(root/'FORWARD_LEDGER.jsonl',dict(event='success',**rec));counts['diagnostic_forward_success']+=1;return result
                q=forward('ema',lambda:t.ema(x,mode='teacher').softmax(1));qm=forward('memory',lambda:t.memory(x,mode='teacher').softmax(1));qf=forward('memory_flip',lambda:t.memory(x.flip(-1),mode='teacher').softmax(1).flip(-1));p=forward('student',lambda:t.clean(x))
                _,_,target=transfer(p,q,valid,11,qm,qf);features,eligible,classes=S.features(p,q,qm,qf,target,valid,100)
                mask,selection=S.choose(features,eligible,classes,'COVERAGE',None,None);mask=mask.to(q.device)
                changed=conflict(target,q,mask);out=P.permute(target,q.argmax(1),changed);weight=q.new_tensor([.5,1.,1.5])[q.argmax(1)]
                assert int(changed.sum())==int(((out.argmax(1)!=target.argmax(1))&mask).sum()) and torch.equal(out.sort(1).values,target.sort(1).values)
                details.append(M.measure(target,out,y,mask,weight))
        assert c.e.same(entry,c.snapshot(t)) and not ledger.count
        row=dict(ctx,stream=stream,**{k:sum(v[k] for v in details) for k in details[0]});row.update(M.rates(row))
        assert counts['diagnostic_forward_success']==counts['diagnostic_forward_attempts']==16
        N.D.write(root/'DETAILS.private.json',details);N.D.write(root/'RESULT.json',row)
    elif job=='train':
        assert N.D.read(campaign/'QUALITY_DECISION.json')['passed'] and N.D.read(campaign/'QUALITY_LOCK.json')['rows']==32
        entry=restore(entryfile)
        for mode in A.METHODS:
            c.restore(t,entry);t.action=11;t.category='prior';t.key=f"ctx{ctx['context']}/stream{stream}/{mode}";profile['mode']=mode
            for _ in range(200):
                t.update()
                if t.step in (150,300):c.e.atomic_save(c.snapshot(t),root/f'{mode}_{t.step}.private.pt')
                if t.step%50==0:N.B.write(root/'STATUS.json',dict(status='RUNNING',mode=mode,step=t.step,physical=dict(ledger.count)))
            freeze_check();N.D.write(root/(mode+'_TRAINING.json'),dict(status='SEALED',updates=200,time=time.time()))
        assert dict(ledger.count)=={'prior':1000}
    elif job=='evaluate':
        assert N.D.read(campaign/'ENDPOINT_LOCK.json')['snapshots']==320
        u_allowed=False
        qr=copy.deepcopy(r);qr['Q_train_new']=r['A_hold']
        def query(role,cond):
            nonlocal allowed
            allowed=set(qr[role]);counts['query_attempts']+=1;rec=dict(ordinal=counts['query_attempts'],role=role,step=t.step)
            c.e.append(root/'QUERY_LEDGER.jsonl',dict(event='attempt',**rec));value=scores(t,qr,role,cond)
            c.e.append(root/'QUERY_LEDGER.jsonl',dict(event='success',**rec));counts['query_success']+=1;counts['query_images']+=4;return value
        restore(entryfile);new_entry=query('Q_train_new',condition);old_entry=query('Q_train_old',('identity',1.));rows=[]
        src=campaign/f"jobs/train{ctx['context']}_{stream}"
        for mode in A.METHODS:
            for at in (150,300):
                restore(src/f'{mode}_{at}.private.pt');new=query('Q_train_new',condition);old=query('Q_train_old',('identity',1.))
                rows.append(dict(ctx,stream=stream,method=mode,step=at,new=new['macro'],old=old['macro'],new_rim=new['rim'],new_cup=new['cup'],old_rim=old['rim'],old_cup=old['cup'],new_entry=new_entry['macro'],old_entry=old_entry['macro']))
        assert not ledger.count and counts['query_success']==counts['query_attempts']==22 and counts['query_images']==88
        N.D.write(root/'RESULTS.json',rows)
    else:raise ValueError(job)
    N.D.write(root/'COUNTS.json',dict(counts));N.D.write(root/'FINAL.json',dict(status='COMPLETE',physical=dict(ledger.count),time=time.time()))


def coordinator(root,cfg):
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.run_b import Accounting
    lock=(root/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=Path(cfg['v122']);assert N.D.read(old/'COMPLETION_AUDIT.json')['status']=='PASS' and N.D.read(old/'FINAL_PUBLICATION.json')['anonymous_http']=='200'
    receipt=N.D.read(Path(cfg['action_root'])/'auxiliary/receipt.json');assert receipt['roles']=='M_fit only; native first task does not consume U'
    (root/'jobs').mkdir(exist_ok=False);(root/'PHYSICAL_LEDGER.jsonl').touch(exist_ok=False)
    jobs=lambda kind,caps:[dict(id=f'{kind}{i}_{s}',job=kind,context_index=i,stream=s,caps=caps) for i in range(16) for s in A.STREAMS]
    entries=jobs('entry',dict(entries=100));qual=[dict(id='qualification',job='qualification',context_index=0,stream=3,caps=dict(qualification=24))]
    quality=jobs('quality',{});train=jobs('train',dict(prior=1000));evaluate=jobs('evaluate',{})
    account=Accounting(root,caps=CAPS,jobs=entries+qual+quality+train+evaluate)
    N.schedule(root,cfg,entries,account)
    assert dict(account.count)=={'entries':3200} and not account.failure
    N.D.write(root/'ENTRY_LOCK.json',dict(status='SEALED_BEFORE_HELDOUT_READ',entries=32,time=time.time()))
    N.schedule(root,cfg,qual,account);N.schedule(root,cfg,quality,account)
    rows=[N.D.read(root/'jobs'/j['id']/'RESULT.json') for j in quality];decision=A.quality(rows)
    N.D.write(root/'QUALITY_RESULTS.json',rows);N.D.table(root/'QUALITY_RESULTS.csv',rows);N.D.write(root/'QUALITY_DECISION.json',decision)
    N.D.write(root/'QUALITY_LOCK.json',dict(rows=32,passed=decision['passed'],time=time.time()))
    selected_jobs=entries+qual+quality
    if decision['passed']:
        N.schedule(root,cfg,train,account)
        assert dict(account.count)==CAPS and not account.failure
        assert all((root/'jobs'/j['id']/f'{m}_{at}.private.pt').is_file() for j in train for m in A.METHODS for at in (150,300))
        N.D.write(root/'ENDPOINT_LOCK.json',dict(status='SEALED_BEFORE_PERFORMANCE_READOUT',trajectories=160,snapshots=320,time=time.time()))
        N.schedule(root,cfg,evaluate,account);selected_jobs+=train+evaluate
        results=sum((N.D.read(root/'jobs'/j['id']/'RESULTS.json') for j in evaluate),[])
        N.D.write(root/'RESULTS.json',results);N.D.table(root/'RESULTS.csv',results)
        for name,value in A.training(results).items():N.D.write(root/(name+'.json'),value)
    else:N.D.write(root/'DECISION.json',dict(status=decision['status'],training_stage='NOT_RUN_GATE_FAILED',passed=False,independent_confirmation=False,campaign_success=False))
    counts=Counter()
    for j in selected_jobs:counts.update(N.D.read(root/'jobs'/j['id']/'COUNTS.json'))
    assert counts['diagnostic_forward_success']==counts['diagnostic_forward_attempts']==512
    assert counts.get('query_images',0)==(2816 if decision['passed'] else 0)
    N.B.write(root/'COSTS.json',dict(native_updates=sum(account.count.values()),physical=dict(account.count),success=dict(account.success),failures=dict(account.failure),counts=dict(counts),actor_optimizer_updates=0,linear_solves=0,Q_dev_images=0,new_annotation_cases=0,held_out_diagnostic_images=128,diagnostic_image_forwards=512,stage_B_executed=decision['passed']))
    N.D.write(root/'FINAL.json',dict(status='COMPLETE',stage_B_executed=decision['passed'],decision=N.D.read(root/'DECISION.json')['status'],time=time.time()));N.B.write(root/'STATUS.json',N.D.read(root/'FINAL.json'))


if __name__=='__main__':
    root=Path(os.environ['EXEC_RUN']);cfg=json.loads(Path(os.environ['EXEC_CONFIG']).read_text());configure(cfg)
    if os.environ.get('EXEC_SELFCHECK')=='1':print(json.dumps(selfcheck()))
    else:
        N.D.write(root/'STARTED.json',dict(pid=os.getpid(),time=time.time(),commit=cfg['commit']))
        try:
            if os.environ.get('EXEC_MODE')=='job':worker(root,cfg)
            else:coordinator(root,cfg)
        except BaseException as exc:N.D.write(root/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()));raise

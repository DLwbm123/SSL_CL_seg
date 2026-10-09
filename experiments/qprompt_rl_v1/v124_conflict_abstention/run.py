"""Frozen conflict abstention against matched weight-removal controls."""
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

CAPS=dict(qualification=24,prior=32000)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def configure(cfg):
    global N,c,b,S,A,Q,H,fold_roles,conflict
    H=load('heldout',str(Path(cfg['v123'])/'run.py'));H.configure(cfg)
    N=H.N;c=H.c;b=H.b;S=H.S;A=H.A;Q=H.Q;fold_roles=H.fold_roles;conflict=H.conflict
    N.STAGE='V124';c.CAPS.update(CAPS)


def random_removal(selected,classes,chosen,seed):
    # A local CPU generator leaves the paired model/provider RNG stream untouched.
    selected=selected.cpu();classes=classes.cpu();chosen=chosen.cpu()
    generator=torch.Generator().manual_seed(seed);removed=torch.zeros_like(selected)
    for image in range(len(selected)):
        for cls in range(3):
            count=int((chosen[image]&(classes[image]==cls)).sum())
            candidates=torch.where((selected[image]&(classes[image]==cls)).flatten())[0]
            assert count<=len(candidates)
            ids=candidates[torch.randperm(len(candidates),generator=generator)[:count]]
            removed[image].flatten()[ids]=True
    return removed


def intervention(p,q,target,valid,mask,mode,seed=124000000):
    assert mode in (*A.METHODS,'GOLD')
    selected=mask.to(p.device);chosen=conflict(target,q,selected);classes=q.detach().argmax(1)
    out=target.detach();raw=q.new_tensor([.5,1.,1.5])[classes]
    admitted=valid.bool()&(q.detach().max(1).values>.7)
    weight=raw*selected;weight=weight*admitted.sum()/weight.sum().clamp_min(1e-8)
    mass=float((raw*selected).sum());conflict_mass=float((raw*chosen).sum())
    removed_mask=torch.zeros_like(chosen);scale=1.
    if mode=='IGNORE_C':removed_mask=chosen
    elif mode=='RANDOM_MATCHED':removed_mask=random_removal(selected,classes,chosen,seed).to(p.device)
    elif mode=='OFF':removed_mask=selected
    elif mode=='UNIFORM_MATCHED':scale=1.-conflict_mass/max(mass,1e-8)
    if mode in ('BASE','GOLD'):loss=S.selected_loss(p,q,out,valid,mask)
    else:
        # Compute normalization before removing weight; never redistribute it.
        kl=(out*(out.clamp_min(1e-8).log()-p.float().clamp_min(1e-8).log())).sum(1)
        loss=(kl*weight*(~removed_mask)).sum()/valid.sum().clamp_min(1)*scale
    removed=conflict_mass if mode=='UNIFORM_MATCHED' else float((raw*removed_mask).sum())
    quotas=[[int((chosen[i]&(classes[i]==k)).sum()) for k in range(3)] for i in range(len(selected))]
    actual=[[int((removed_mask[i]&(classes[i]==k)).sum()) for k in range(3)] for i in range(len(selected))]
    if mode=='RANDOM_MATCHED':assert actual==quotas and removed==conflict_mass
    detail=dict(selected=int(selected.sum()),conflict=int(chosen.sum()),selected_raw_weight=mass,conflict_raw_weight=conflict_mass,
        removed_raw_weight=removed,removed_normalized_weight=removed*int(admitted.sum())/max(mass,1e-8),
        uniform_scale=scale,conflict_class_counts=quotas,removed_class_counts=actual,
        removed_pixels=int(removed_mask.sum()),removed_conflict_overlap=int((removed_mask&chosen).sum()))
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
        assert torch.isfinite(loss) and not out.requires_grad and torch.equal(out,target.detach())
        gradients=torch.autograd.grad(loss,(logits,q,target),retain_graph=True,allow_unused=True)
        assert torch.isfinite(gradients[0]).all() and gradients[1:]==(None,None)
        if mode!='BASE':assert float(loss)==0 and not gradients[0].any() and detail['removed_raw_weight']==detail['selected_raw_weight']
        assert intervention(p,q,target,valid&False,mask&False,mode)[0]==0
    partial=target.detach().clone();partial[:,:,:,0]=q.detach()[:,:,:,0]
    ignore=intervention(p,q,partial,valid,mask,'IGNORE_C')[0]
    remain=mask&~conflict(partial,q,mask)
    renormalized=S.selected_loss(p,q,partial,valid,remain)
    assert torch.allclose(ignore,renormalized*.5) and not torch.allclose(ignore,renormalized)
    uniform=intervention(p,q,partial,valid,mask,'UNIFORM_MATCHED')[0]
    assert torch.allclose(uniform,S.selected_loss(p,q,partial,valid,mask)*.5)
    for mode in A.METHODS[:-1]:assert torch.equal(intervention(p,q,q,valid,mask,mode)[0],S.selected_loss(p,q,q,valid,mask))
    sel=torch.ones(2,4,8,dtype=torch.bool);cls=torch.arange(64).reshape(2,4,8)%3;chosen=sel&False;chosen[:,0,:3]=True
    rng=torch.get_rng_state().clone();removed=random_removal(sel,cls,chosen,124000000)
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(removed,random_removal(sel,cls,chosen,124000000))
    assert (removed&~chosen).any() and not (removed&~sel).any()
    for i in range(2):
        for k in range(3):assert int((removed[i]&(cls[i]==k)).sum())==int((chosen[i]&(cls[i]==k)).sum())
    return dict(status='PASS',model_forwards=0,image_reads=0,optimizer_calls=0,checks=['fold exclusion','BASE exact loss and gradient','unchanged detached target','fixed IGNORE denominator','random per-image/class mass and deterministic local RNG','uniform same-weight scaling','OFF/empty differentiable zero','strong-control and absolute gates'])


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
    assert cfg['job'] in ('qualification','train','evaluate')
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
        loss,out,detail=intervention(p,q,target,valid,mask,profile['mode'],124000000+100000*ctx['context']+10000*stream+t.step)
        if profile['mode']=='GOLD':loss=S.selected_loss(p,q,target,valid,mask)
        counts['selector_calls']+=1;c.e.append(root/'SELECTION_LEDGER.jsonl',dict(ordinal=counts['selector_calls'],key=t.key,category=t.category,step=t.step,mode=profile['mode'],image_budgets=[(v['eligible'],v['selected']) for v in selection],**detail))
        return loss,stats,out
    c.transfer_loss=selected
    campaign=Path(cfg['campaign']);entryfile=Path(cfg['v123'])/f"jobs/entry{ctx['context']}_{stream}/ENTRY100.private.pt"
    def restore(path):
        state=torch.load(path,map_location='cpu',weights_only=False);c.restore(t,state);return state
    def freeze_check():
        assert c.e.same(initial_memory,t.memory.state_dict()) and all(p.grad is None for model in (t.memory,t.ema) for p in model.parameters())
    job=cfg['job']
    if job=='qualification':
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
    elif job=='train':
        assert N.D.read(campaign/'ENTRY_REUSE.json')['entries']==32 and N.D.read(campaign/'jobs/qualification/QUALIFICATION.json')['status']=='PASS'
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
    old=Path(cfg['v123']);assert N.D.read(old/'COMPLETION_AUDIT.json')['status']=='PASS' and N.D.read(old/'FINAL_PUBLICATION.json')['anonymous_http']=='200'
    assert N.D.read(old/'DECISION.json')['training_stage']=='NOT_RUN_GATE_FAILED'
    receipt=N.D.read(Path(cfg['action_root'])/'auxiliary/receipt.json');assert receipt['roles']=='M_fit only; native first task does not consume U'
    for ctx in A.contexts():
        for stream in A.STREAMS:
            source=old/f"jobs/entry{ctx['context']}_{stream}"
            assert (source/'ENTRY100.private.pt').is_file()
            partition=N.D.read(source/'PARTITION.private.json')
            assert len(partition['held'])==4 and len(partition['fit'])==ctx['labeled_images'] and not set(partition['held'])&set(partition['fit']+partition['source'])
    N.D.write(root/'ENTRY_REUSE.json',dict(entries=32,source=old.name,source_audit='PASS',new_entry_updates=0,time=time.time()))
    (root/'jobs').mkdir(exist_ok=False);(root/'PHYSICAL_LEDGER.jsonl').touch(exist_ok=False)
    jobs=lambda kind,caps:[dict(id=f'{kind}{i}_{s}',job=kind,context_index=i,stream=s,caps=caps) for i in range(16) for s in A.STREAMS]
    qual=[dict(id='qualification',job='qualification',context_index=0,stream=3,caps=dict(qualification=24))]
    train=jobs('train',dict(prior=1000));evaluate=jobs('evaluate',{})
    account=Accounting(root,caps=CAPS,jobs=qual+train+evaluate)
    N.schedule(root,cfg,qual,account)
    assert N.D.read(root/'jobs/qualification/QUALIFICATION.json')['status']=='PASS'
    N.schedule(root,cfg,train,account)
    assert dict(account.count)==CAPS and not account.failure
    assert all((root/'jobs'/j['id']/f'{m}_{at}.private.pt').is_file() for j in train for m in A.METHODS for at in (150,300))
    N.D.write(root/'ENDPOINT_LOCK.json',dict(status='SEALED_BEFORE_PERFORMANCE_READOUT',trajectories=160,snapshots=320,time=time.time()))
    N.schedule(root,cfg,evaluate,account)
    results=sum((N.D.read(root/'jobs'/j['id']/'RESULTS.json') for j in evaluate),[])
    N.D.write(root/'RESULTS.json',results);N.D.table(root/'RESULTS.csv',results)
    for name,value in A.training(results).items():N.D.write(root/(name+'.json'),value)
    counts=Counter()
    for j in qual+train+evaluate:counts.update(N.D.read(root/'jobs'/j['id']/'COUNTS.json'))
    assert counts['query_images']==2816 and counts['query_attempts']==counts['query_success']==704
    N.B.write(root/'COSTS.json',dict(native_updates=sum(account.count.values()),physical=dict(account.count),success=dict(account.success),failures=dict(account.failure),counts=dict(counts),actor_optimizer_updates=0,linear_solves=0,Q_dev_images=0,new_annotation_cases=0,new_entry_updates=0,diagnostic_image_forwards=0,performance_query_calls=704,performance_query_images=2816))
    N.D.write(root/'FINAL.json',dict(status='COMPLETE',decision=N.D.read(root/'DECISION.json')['status'],time=time.time()));N.B.write(root/'STATUS.json',N.D.read(root/'FINAL.json'))


if __name__=='__main__':
    root=Path(os.environ['EXEC_RUN']);cfg=json.loads(Path(os.environ['EXEC_CONFIG']).read_text());configure(cfg)
    if os.environ.get('EXEC_SELFCHECK')=='1':print(json.dumps(selfcheck()))
    else:
        N.D.write(root/'STARTED.json',dict(pid=os.getpid(),time=time.time(),commit=cfg['commit']))
        try:
            if os.environ.get('EXEC_MODE')=='job':worker(root,cfg)
            else:coordinator(root,cfg)
        except BaseException as exc:N.D.write(root/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()));raise

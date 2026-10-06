"""Finite V8 qualification, clean auxiliary source and paired action screen."""
import copy
import json
import os
import random
import time
import traceback
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from experiments.lcrseg.five_frameworks_v1 import native_runner as native
from . import core as c
from .core import e


def scores(t,roles,role,condition):
    if t.provider.domain!='REFUGE' or role not in ('Q_train_old','Q_train_new','Q_dev_old','Q_dev_new'):raise PermissionError('query role/domain denied')
    ds=c.subset(e.primitive.CurrentData(t.provider._l.data,0,'train_labeled'),roles[role]);values=[]
    with t.readonly(),torch.no_grad():
        for i in range(len(ds)):
            item=ds[i];y=item['label'].to('cuda:0');p=t.clean(c.photo(item['image'][None].to('cuda:0'),condition))[0];valid=y!=255
            values.append([float((2*(p[k][valid]*(y[valid]==k)).sum()+1)/(p[k][valid].sum()+(y[valid]==k).sum()+1)) for k in (1,2)])
    return dict(macro=float(np.mean(values)),rim=float(np.mean([v[0] for v in values])),cup=float(np.mean([v[1] for v in values])))


def qualify(root,config,roles,ledger):
    checks={};torch.manual_seed(8601)
    logits=torch.randn(2,3,5,5,requires_grad=True);p=logits.softmax(1);q=torch.randn_like(p).softmax(1);q[:,0]=.9;q[:,1:]=.05;valid=torch.ones(2,5,5,dtype=torch.bool)
    a=c.transfer_loss(p,q,valid,0)[0];b=e.u_loss(p,q,valid,2)[0]
    assert torch.equal(a,b) and torch.equal(torch.autograd.grad(a,logits,retain_graph=True)[0],torch.autograd.grad(b,logits)[0]);checks['native_loss_gradient']='PASS'
    for action in range(9):
        m=q.clone().requires_grad_();loss,stats,target=c.transfer_loss(p,q,valid,action,m,m)
        assert torch.isfinite(loss) and not target.requires_grad and torch.allclose(target.sum(1),torch.ones_like(valid,dtype=torch.float32))
        assert abs(stats['weight_normalized']-stats['admitted'])<1e-4
        zero=c.transfer_loss(p,q,valid&False,action,m,m)[0];assert zero.requires_grad and float(zero)==0
    checks['target_detach_normalization_and_empty_support']='PASS'
    try:c.transfer_loss(p,q,valid,3)
    except PermissionError:pass
    else:raise AssertionError('random memory protected')
    actor=c.Actor(601);z=torch.zeros(24);g=torch.Generator().manual_seed(123);h=torch.Generator().manual_seed(123);rng=e.cpu(e.rng_state())
    for _ in range(100):assert actor.sample(z,g)==int(torch.multinomial(torch.ones(9)/9,1,generator=h))
    assert e.same(rng,e.rng_state())
    for _ in range(100):assert actor.sample(z,g,False)<3
    checks['private_rng_zero_uniform_legal_mask']='PASS'
    source=Path(config['source'])/'SOURCE_S168';receipt=e.read(source/'receipt.json');payload=torch.load(source/'student.pt',map_location='cpu',weights_only=False)
    assert receipt['status']=='SEALED' and payload['step']==8000 and payload['identity']==receipt['identity']
    assert e.tensor_fingerprint(payload['student'])==receipt['student_hash']
    t=c.create(config,roles,payload,ledger);del payload
    entry=c.snapshot(t);t.action=0;t.update();ours=c.snapshot(t);c.restore(t,entry)
    def native_loss(self):
        l,p,q,v=self.components();ul=e.u_loss(p,q,v,2)[0];return l,.5*ul,None
    with patch.object(c.Trainer,'losses',native_loss):t.update()
    assert e.same(ours['student'],t.model.state_dict()) and e.same(ours['optimizer'],t.optimizer.state_dict())
    checks['native_update']='PASS';c.restore(t,entry)
    for _ in range(4):t.update()
    continuous=c.snapshot(t);c.restore(t,entry)
    for _ in range(2):t.update()
    mid=c.snapshot(t);e.atomic_save(mid,root/'QUALIFICATION_ENTRY.private.pt');loaded=torch.load(root/'QUALIFICATION_ENTRY.private.pt',map_location='cpu',weights_only=False);c.restore(t,loaded)
    for _ in range(2):t.update()
    assert e.same(continuous,c.snapshot(t));checks['continuous_vs_2_plus_2_and_full_restore']='PASS'
    before=c.snapshot(t);z=t.extract();assert e.same(before,c.snapshot(t));assert float(z[0])==float(torch.tensor(t.step/t.options['total_steps']))
    mem=e.cpu(t.memory.state_dict());t.action=8;t.update();assert e.same(mem,t.memory.state_dict()) and all(p.grad is None for p in t.memory.parameters())
    checks['isolated_features_and_frozen_memory']='PASS'
    before=c.snapshot(t);opt=torch.optim.Adam(actor.parameters(),lr=.001)
    result=c.actor_update(actor,opt,z,[0,1,2,3],[0.,.01,.02,.03],1e-4,ledger,'actor_qualification','full_group')
    assert len(result['updates'])<=4 and e.same(before,c.snapshot(t));checks['actor_isolation_full_group']='PASS'
    count=ledger.count.copy();c.restore(t,entry);assert ledger.count==count and ledger.count['qualification']==11;checks['physical_ledger_not_rolled_back']='PASS'
    fit={r['case_id'] for r in t.provider._l.rows};u={r['case_id'] for r in t.provider._u.rows};queries=set(x for k,v in roles.items() if k.startswith('Q_') for x in v)
    assert not (fit|u)&queries and not any('label' in k for r in t.provider._u.rows for k in r)
    try:c.subset(t.provider._l,roles['Q_train_old'])
    except PermissionError:pass
    else:raise AssertionError('query bypass')
    t.provider.domain='RIM_ONE_r3'
    try:scores(t,roles,'Q_train_old',('identity',1.))
    except PermissionError:pass
    else:raise AssertionError('old query reachable')
    t.provider.domain='REFUGE';checks['role_scoping_and_history_free_query']='PASS'
    try:e.ExecutionPermit({},(),{},object()).validate()
    except (PermissionError,RuntimeError,ValueError):pass
    else:raise AssertionError('permit seal bypass')
    checks['permit_required']='PASS'
    with t.readonly(),torch.no_grad():
        t.model.eval();x,_,_=t.provider.labeled(0);expected=t.clean(x);deploy=t.model.deploy();actual=deploy(x).softmax(1)
        assert torch.allclose(expected,actual,atol=2e-6,rtol=2e-5)
        for ad in t.model.parent.adapters:
            v=getattr(t.model.parent,'V64_'+str(ad.index));res=(ad.a.double()@v.double()).norm()/ad.a.double().norm().clamp_min(1e-12)
            assert float(res)<1e-4
    checks['A_only_and_single_student_deployment']='PASS'
    result=dict(status='PASS',checks=checks,physical_counts=dict(ledger.count),scope='image-level exploratory; not patient-independent')
    e.write(root/'QUALIFICATION.json',result);del t;torch.cuda.empty_cache()


def auxiliary(root,config,roles,ledger):
    dest=root/'auxiliary';dest.mkdir(exist_ok=False)
    original=e.atomic_save;original_step=torch.optim.Adam.step
    def scoped(*args,**kw):return c.Provider(*args,**kw,roles=roles,fit='M_fit',n=16)
    def save(value,path):
        original(value,path)
        if value['step'] in (2000,8000) and Path(path).name=='latest.pt':
            original(dict(value,rng=e.rng_state(),cursor=value['step']),dest/f"step{value['step']}.pt")
    def step(opt,*args,**kw):return ledger.call('auxiliary','clean_source',lambda:original_step(opt,*args,**kw))
    cfg=dict(config,execution_commit=config['commit']);node=dict(seed=168,id='V8_CLEAN_AUXILIARY')
    with patch.object(native,'NativeCurrentDomain',scoped),patch.object(native,'evaluate',lambda *a,**kw:({},{})),patch.object(native.checkpoint,'atomic_save',save),patch.object(torch.optim.Adam,'step',step):
        receipt=native.source_task(cfg,node,c.permit(config),dest,torch.device('cuda:0'))
    receipt.update(roles='M_fit only; native first task does not consume U',query_exposure=False,independence_unit='image',evaluation='NOT_RUN')
    e.write(dest/'receipt.json',receipt)


def action_screen(root,config,roles,ledger):
    allrows=[]
    for state in (2000,8000):
        payload=torch.load(root/'auxiliary'/f'step{state}.pt',map_location='cpu',weights_only=False)
        for n in (2,8):
            for condition in (('brightness',.8),('contrast',1.2)):
                key=f'm{state}_n{n}_{condition[0]}';dest=root/key;dest.mkdir(exist_ok=False)
                t=c.create(config,roles,payload,ledger,n,condition);t.key=key;t.category='entries'
                reference=scores(t,roles,'Q_train_old',('identity',1.))['macro']
                for _ in range(100):t.update()
                entry=c.snapshot(t);e.atomic_save(entry,dest/'ENTRY.private.pt');entrynew=scores(t,roles,'Q_train_new',condition)['macro']
                for stream in (1,2):
                    for action in range(9):
                        c.restore(t,entry);t.provider.seed=168+stream*10000
                        random.seed(860100+stream);np.random.seed(860100+stream);torch.manual_seed(860100+stream);torch.cuda.manual_seed_all(860100+stream)
                        t.category='audit';t.key=f'{key}/s{stream}/a{action}';t.action=action;mid=None;stats=[];grads=[]
                        for step in range(100):
                            t.update();stats.append(copy.deepcopy(t.last['transfer']))
                            grads.append(sum(float(p.grad.square().sum()) for p in t.model.parameters() if p.grad is not None)**.5)
                            if step==24:mid=scores(t,roles,'Q_train_new',condition)['macro']
                        new=scores(t,roles,'Q_train_new',condition);old=scores(t,roles,'Q_train_old',('identity',1.));gain=.25*(mid-entrynew)+.75*(new['macro']-entrynew);forget=max(0.,reference-old['macro']-.005)
                        difference=sum(float((p.detach().cpu()-entry['student'][name]).square().sum()) for name,p in t.model.named_parameters())**.5
                        row=dict(entry_parameter_delta=difference,context=key,stream=stream,action=action,new=new,old=old,new25=mid,entry_new=entrynew,memory_old=reference,reward=gain-forget,gain=gain,forget_penalty=forget,gate=float(np.mean([s['gate'] for s in stats])),admitted=float(np.mean([s['admitted'] for s in stats])),gradient_norm=float(np.mean(grads)),update_norm=float(sum(v*v for v in t.last['actual_update_norms'].values())**.5))
                        allrows.append(row);e.append(root/'ACTION_ROWS.jsonl',row);e.write(root/'STATUS.json',dict(status='RUNNING',phase='ACTION_AUDIT',completed=len(allrows),total=144,physical=dict(ledger.count)))
                        t.provider.seed=168
                del t;torch.cuda.empty_cache()
        del payload
    contexts=sorted(set(r['context'] for r in allrows));passed=[];noise=[]
    for context in contexts:
        index={(r['stream'],r['action']):r for r in allrows if r['context']==context}
        hits=[]
        for a in range(9):
            noise.append(abs(index[1,a]['reward']-index[2,a]['reward']))
            if a and index[1,a]['new']['macro']-index[1,0]['new']['macro']>=.002 and index[2,a]['new']['macro']>index[2,0]['new']['macro'] and all(index[s,a]['old']['macro']-index[s,0]['old']['macro']>=-.005 for s in (1,2)):hits.append(a)
        if hits:passed.append(dict(context=context,actions=hits))
    means={a:float(np.mean([r['reward'] for r in allrows if r['action']==a])) for a in range(9)}
    best=sorted(means,key=lambda a:(-means[a],a!=0,c.ACTIONS[a][0],a))[0]
    result=dict(status='PASS_ACTION_SIGNAL' if len(passed)>=4 else 'STOP_ACTION_SIGNAL_WEAK',passed_contexts=passed,required=4,fixed_best=best,action_reward_means=means,sigma_floor=max(1e-4,float(np.median(noise))/2**.5),physical=dict(ledger.count),B='NOT_RUN',C='NOT_RUN',D='NOT_RUN')
    e.write(root/'ACTION_DECISION.json',result);e.write(root/'STATUS.json',result)


def main():
    root=Path(os.environ['EXEC_RUN']);config=e.read(os.environ['EXEC_CONFIG']);ledger=c.Ledger(root)
    torch.set_num_threads(2);torch.cuda.set_device(0);random.seed(168);np.random.seed(168);torch.manual_seed(168)
    if (root/'STARTED.json').exists():raise RuntimeError('create-only; no automatic retry')
    e.write(root/'STARTED.json',dict(pid=os.getpid(),time=time.time(),commit=config['commit']))
    roles=c.split_roles(config['data']);e.write(root/'ROLES.private.json',roles)
    try:
        e.write(root/'STATUS.json',dict(status='RUNNING',phase='QUALIFICATION'));qualify(root,config,roles,ledger)
        e.write(root/'STATUS.json',dict(status='RUNNING',phase='CLEAN_AUXILIARY'));auxiliary(root,config,roles,ledger)
        e.write(root/'STATUS.json',dict(status='RUNNING',phase='ACTION_AUDIT'));action_screen(root,config,roles,ledger)
    except BaseException as exc:
        e.write(root/'STATUS.json',dict(status='ENGINEERING_STOP',error=repr(exc),traceback=traceback.format_exc(),physical=dict(ledger.count)));raise

if __name__=='__main__':main()

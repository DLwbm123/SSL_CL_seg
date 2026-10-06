"""First-domain GRPO and held-out development; no target-domain reads."""
import copy
import json
import os
import random
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from . import core as c
from .core import e
from .worker import scores


class JobLedger(c.Ledger):
    def __init__(self,root,caps):super().__init__(root);self.caps=caps
    def call(self,category,key,fn):
        if category not in self.caps or self.count[category]>=self.caps[category]:raise RuntimeError('job physical cap '+category)
        return super().call(category,key,fn)


def contexts():
    return [(m,n,(kind,factor),f'm{m}_n{n}_{kind}') for m in (2000,8000) for n in (2,8) for kind,factor in (('brightness',.8),('contrast',1.2))]


def make(config,roles,ledger,context,horizon=900):
    m,n,condition,_=context
    payload=torch.load(Path(config['action_root'])/'auxiliary'/f'step{m}.pt',map_location='cpu',weights_only=False)
    assert payload['step']==m
    return c.create(config,roles,payload,ledger,n,condition,horizon=horizon)


def rollout(t,roles,condition,steps=100,category='prior'):
    assert (category=='prior' and steps==100) or (category=='qualification' and steps==2)
    short=25 if steps==100 else 1;mid=None;t.category=category
    gradient=[];gate=[];losses=[]
    for step in range(steps):
        t.update();gradient.append(sum(float(p.grad.square().sum()) for p in t.model.parameters() if p.grad is not None)**.5)
        gate.append(t.last['transfer']['gate']);losses.append((t.last['labeled_loss'],t.last['unlabeled_loss']))
        if step+1==short:mid=scores(t,roles,'Q_train_new',condition)
    return dict(new_short=mid,new=scores(t,roles,'Q_train_new',condition),old=scores(t,roles,'Q_train_old',('identity',1.)),gradient_norm=float(np.mean(gradient)),memory_gate=float(np.mean(gate)),labeled_loss=float(np.mean([x[0] for x in losses])),unlabeled_loss_weighted=float(np.mean([x[1] for x in losses])))


def group(t,roles,actor,opt,g,floor,reference,ledger,key,steps=100,category='prior',actor_category='actor_prior'):
    entry=c.snapshot(t);z=t.extract();entry_new=scores(t,roles,'Q_train_new',t.provider.condition)['macro']
    behavior=actor(z).detach().softmax(-1).tolist()
    actions=[actor.sample(z,g) for _ in range(4)];branches=[];retained=None
    for i,action in enumerate(actions):
        c.restore(t,entry);t.action=action;t.key=key+f'/branch{i}'
        row=rollout(t,roles,t.provider.condition,steps,category)
        gain=.25*(row['new_short']['macro']-entry_new)+.75*(row['new']['macro']-entry_new)
        penalty=max(0.,reference-row['old']['macro']-.005)
        branches.append(dict(branch=i,action=action,gain=gain,forget_penalty=penalty,reward=gain-penalty,**row))
        if i==0:retained=c.snapshot(t)
    c.restore(t,entry);before=c.snapshot(t)
    result=c.actor_update(actor,opt,z,actions,[r['reward'] for r in branches],floor,ledger,actor_category,key)
    assert e.same(before,c.snapshot(t)), 'actor mutated student/EMA/provider/RNG'
    c.restore(t,retained)
    assert e.same(retained,c.snapshot(t)), 'retained branch zero state differs'
    assert t.step==entry['step']+steps
    return dict(actions=actions,behavior_probabilities=behavior,after_probabilities=actor(z).detach().softmax(-1).tolist(),state=z.tolist(),entry_step=entry['step'],retained_branch=0,new_entry=entry_new,old_frozen_memory_reference=reference,branches=branches,actor=result)


def qualify(root,config,roles,ledger):
    ctx=contexts()[0];t=make(config,roles,ledger,ctx)
    v=torch.load(Path(config['action_root'])/ctx[3]/'ENTRY.private.pt',map_location='cpu',weights_only=False);c.restore(t,v)
    actor=c.Actor(601);opt=torch.optim.Adam(actor.parameters(),lr=.001);g=torch.Generator().manual_seed(601)
    before=c.snapshot(t);e.atomic_save(before,root/'ENTRY.private.pt')
    reference=next(r['memory_old'] for r in config['action_rows'] if r['context']==ctx[3])
    out=group(t,roles,actor,opt,g,config['sigma_floor'],reference,ledger,'qualification',2,'qualification','actor_qualification')
    # group() checks the complete restored branch-zero state, not its reward rank.
    assert out['retained_branch']==0 and t.step==before['step']+2 and ledger.count['qualification']==8
    student=c.snapshot(t);c.actor_update(actor,opt,t.extract(),[0,1,2,3],[0.,.02,.04,.06],1e-4,ledger,'actor_qualification','synthetic_full_group')
    assert e.same(student,c.snapshot(t))
    e.write(root/'QUALIFICATION.json',dict(status='PASS',real_student_updates=8,actor_updates=ledger.count['actor_qualification'],checks=['four paired restored branches','branch zero retained','fixed memory reference','query readout isolation','actor update isolation','low-signal skip and full-group core reused'],source_qualification_reused=True,commit=config['commit']))


def learn(root,config,roles,ledger):
    assert e.read(Path(config['campaign'])/'jobs/qualification/QUALIFICATION.json')['status']=='PASS'
    seed=config['controller'];actor=c.Actor(seed);opt=torch.optim.Adam(actor.parameters(),lr=.001);g=torch.Generator().manual_seed(seed)
    states={ctx[3]:torch.load(Path(config['action_root'])/ctx[3]/'ENTRY.private.pt',map_location='cpu',weights_only=False) for ctx in contexts()}
    references={}
    for ctx in contexts():
        values=[r['memory_old'] for r in config['action_rows'] if r['context']==ctx[3]]
        assert max(values)-min(values)<1e-8;references[ctx[3]]=values[0]
    for k in range(64):
        ctx=contexts()[k%8];key=ctx[3];t=make(config,roles,ledger,ctx);c.restore(t,states[key]);assert t.step==100+100*(k//8)
        e.atomic_save(dict(student=states[key],actor=actor.state_dict(),optimizer=opt.state_dict(),private_rng=g.get_state(),group=k,context=key),root/'GROUP_ENTRY.private.pt')
        try:result=group(t,roles,actor,opt,g,config['sigma_floor'],references[key],ledger,f'g{k:02d}/{key}')
        except BaseException:
            e.atomic_save(c.snapshot(t),root/'FAILED_STATE.private.pt');raise
        states[key]=c.snapshot(t);e.atomic_save(states[key],root/(key+'.private.pt'))
        e.append(root/'GROUPS.jsonl',dict(group=k,context=key,controller=seed,**result))
        e.atomic_save(dict(actor=actor.state_dict(),optimizer=opt.state_dict(),rng=g.get_state(),completed_groups=k+1,controller=seed),root/'actor_latest.pt')
        e.write(root/'STATUS.json',dict(status='RUNNING',phase='PRIOR_TRAINING',controller=seed,groups=k+1,total=64,physical=dict(ledger.count),time=time.time()))
        del t;torch.cuda.empty_cache()
    assert ledger.count['prior']==25600 and all(v['step']==900 for v in states.values())
    e.atomic_save(dict(actor=actor.state_dict(),controller=seed,groups=64,commit=config['commit']),root/'actor_final.pt')
    e.write(root/'FINAL.json',dict(status='COMPLETE',controller=seed,groups=64,physical=dict(ledger.count),selection='final checkpoint, both seeds retained',time=time.time()))


def development(root,config,roles,ledger):
    campaign=Path(config['campaign']);actors={}
    for seed in (601,602):
        source=campaign/f'jobs/prior_{seed}';assert e.read(source/'FINAL.json')['status']=='COMPLETE'
        payload=torch.load(source/'actor_final.pt',map_location='cpu',weights_only=False);actor=c.Actor(seed);actor.load_state_dict(payload['actor']);actors[seed]=actor
    registered=e.read(Path(config['code'])/'experiments/qprompt_rl_v1/v8_grpo_transfer_prior/ACTION_AUDIT_CONFIG.json')['dev_contexts']
    cc=[(m,n,(kind,factor),f'dev{i}') for i,(m,n,kind,factor) in enumerate(registered)]
    methods=('NATIVE','FIXED_BEST','UNIFORM_ACTION','PRE_FROZEN_601','PRE_FROZEN_602')
    for j,ctx in enumerate(cc):
        dest=root/ctx[3];dest.mkdir(exist_ok=False);t=make(config,roles,ledger,ctx,300);t.category='entries';t.key=ctx[3];t.action=0
        for _ in range(100):t.update()
        entry=c.snapshot(t);e.atomic_save(entry,dest/'ENTRY.private.pt')
        for method in methods:
            c.restore(t,entry);t.category='development';t.key=ctx[3]+'/'+method;g=torch.Generator().manual_seed(860301+j);trajectory=[]
            for step in range(200):
                if step%100==0:
                    if method=='NATIVE':action=0
                    elif method=='FIXED_BEST':action=config['fixed_best']
                    elif method=='UNIFORM_ACTION':action=int(torch.multinomial(torch.ones(9)/9,1,generator=g))
                    else:action=actors[int(method.rsplit('_',1)[1])].sample(t.extract(),g)
                    t.action=action;trajectory.append(action)
                t.update()
                if step==49:e.atomic_save(c.snapshot(t),dest/(method+'_MID.private.pt'))
            e.atomic_save(c.snapshot(t),dest/(method+'_FINAL.private.pt'))
            e.write(dest/(method+'_TRAINING.json'),dict(status='SEALED',steps=200,actions=trajectory,method=method))
            e.write(root/'STATUS.json',dict(status='RUNNING',phase='DEVELOPMENT_TRAINING',context=j,method=method,physical=dict(ledger.count),time=time.time()))
        del t;torch.cuda.empty_cache()
    assert ledger.count['entries']==400 and ledger.count['development']==4000
    e.write(root/'ENDPOINT_LOCK.json',dict(all_twenty_sealed=True,time=time.time()))
    results=[]
    for ctx in cc:
        dest=root/ctx[3];t=make(config,roles,ledger,ctx,300)
        reference=scores(t,roles,'Q_dev_old',('identity',1.))['macro']
        c.restore(t,torch.load(dest/'ENTRY.private.pt',map_location='cpu',weights_only=False));entrynew=scores(t,roles,'Q_dev_new',ctx[2])['macro']
        for method in methods:
            c.restore(t,torch.load(dest/(method+'_MID.private.pt'),map_location='cpu',weights_only=False));mid=scores(t,roles,'Q_dev_new',ctx[2])
            c.restore(t,torch.load(dest/(method+'_FINAL.private.pt'),map_location='cpu',weights_only=False));new=scores(t,roles,'Q_dev_new',ctx[2]);old=scores(t,roles,'Q_dev_old',('identity',1.))
            gain=.25*(mid['macro']-entrynew)+.75*(new['macro']-entrynew);penalty=max(0.,reference-old['macro']-.005)
            results.append(dict(context=ctx[3],method=method,new=new,old=old,new50=mid,new_entry=entrynew,old_memory_reference=reference,gain=gain,forget_penalty=penalty,utility=gain-penalty))
        del t;torch.cuda.empty_cache()
    means={m:{key:float(np.mean([r[key]['macro'] if key in ('new','old') else r[key] for r in results if r['method']==m])) for key in ('new','old','utility')} for m in methods}
    gates={}
    for seed in (601,602):
        row=means[f'PRE_FROZEN_{seed}'];uniform=means['UNIFORM_ACTION'];dn=row['new']-uniform['new'];do=row['old']-uniform['old']
        gates[str(seed)]=dict(utility_improved=row['utility']>uniform['utility'],learning_retention_improved=(dn>=.002 and do>=-.005) or (do>=.005 and dn>=-.0025),new_vs_uniform=dn,old_vs_uniform=do)
    passed=any(v['utility_improved'] or v['learning_retention_improved'] for v in gates.values())
    e.write(root/'DEVELOPMENT_RESULTS.json',dict(results=results,means=means,gates=gates))
    e.write(root/'FINAL.json',dict(status='PASS_PRIOR_DEV_SCREEN' if passed else 'STOP_PRIOR_NOT_TRANSFERABLE_IN_D1_SCREEN',gates=gates,physical=dict(ledger.count),C='NOT_RUN_IMPLEMENTATION_PENDING',D='NOT_RUN',time=time.time()))


def main():
    root=Path(os.environ['EXEC_RUN']);config=e.read(os.environ['EXEC_CONFIG']);ledger=JobLedger(root,config['caps']);torch.set_num_threads(2);torch.cuda.set_device(0);random.seed(168);np.random.seed(168);torch.manual_seed(168)
    assert not (root/'STARTED.json').exists(),'create-only; no automatic retry'
    e.write(root/'STARTED.json',dict(time=time.time(),pid=os.getpid(),commit=config['commit']))
    decision=e.read(Path(config['action_root'])/'ACTION_DECISION.json');assert decision['status']=='PASS_ACTION_SIGNAL'
    roles=c.split_roles(config['data']);assert roles==e.read(Path(config['action_root'])/'ROLES.private.json')
    config.update(sigma_floor=decision['sigma_floor'],fixed_best=decision['fixed_best'],action_rows=[json.loads(s) for s in (Path(config['action_root'])/'ACTION_ROWS.jsonl').read_text().splitlines()])
    if 'sigma_floor_override' in config:
        from experiments.qprompt_rl_v1.v81_paired_noise.round import paired_floor
        assert config['protocol']=='V81_PAIRED_NOISE'
        assert e.read(Path(config['prior_campaign'])/'DECISION.json')['status']=='STOP_PRIOR_NOT_TRANSFERABLE_IN_D1_SCREEN'
        assert abs(config['sigma_floor_override']-paired_floor(config['action_rows']))<1e-12
        config['sigma_floor']=config['sigma_floor_override']
    try:
        {'qualification':qualify,'prior':learn,'development':development}[config['job']](root,config,roles,ledger)
        if config['job']=='qualification':e.write(root/'FINAL.json',dict(status='COMPLETE',physical=dict(ledger.count)))
    except BaseException as exc:
        e.write(root/'STATUS.json',dict(status='ENGINEERING_STOP',error=repr(exc),traceback=traceback.format_exc(),physical=dict(ledger.count)));raise

if __name__=='__main__':main()

"""Native paired complete-action readouts and fixed dense policy supervision."""
import json
import os
import random
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from experiments.qprompt_rl_v1.v8_grpo_transfer_prior import core as c,stage_b as b
from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.core import e
from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.worker import scores
from .round import dataset,preferences


def loss(actor,z,target):
    log=actor(z).log_softmax(-1)
    return -(target*log).sum(-1).mean()+.01*(log.exp()*log).sum(-1).mean()


def qualify(root,config,roles,ledger):
    b.qualify(root,config,roles,ledger)
    actor=c.Actor(601);opt=torch.optim.Adam(actor.parameters(),lr=.001)
    z=torch.stack((torch.ones(24),-torch.ones(24)))
    targets=torch.tensor([preferences([1.]+[0.]*8,1e-4),preferences([0.]*8+[1.],1e-4)])
    before=float(loss(actor,z,targets).detach());opt.zero_grad(set_to_none=True)
    value=loss(actor,z,targets);value.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),1.)
    assert all(torch.isfinite(p.grad).all() for p in actor.parameters())
    ledger.call('actor_qualification','dense_synthetic',opt.step)
    assert float(loss(actor,z,targets).detach())<before
    receipt=e.read(root/'QUALIFICATION.json');receipt.update(actor_updates=ledger.count['actor_qualification'],dense_objective='PASS',qualification_horizon=300)
    e.write(root/'QUALIFICATION.json',receipt)


def entries(root,config,roles,ledger):
    assert e.read(Path(config['campaign'])/'jobs/qualification/QUALIFICATION.json')['status']=='PASS'
    prior=Path(config['prior_campaign'])/'jobs/training_entries'
    assert e.read(prior/'FINAL.json')['entries']==8
    for ctx in b.contexts():
        dest=root/ctx[3];dest.mkdir(exist_ok=False);original=prior/ctx[3]/'ENTRY.private.pt'
        (dest/'ENTRY100.private.pt').symlink_to(original)
        t=b.make(config,roles,ledger,ctx);c.restore(t,torch.load(original,map_location='cpu',weights_only=False))
        assert t.step==100 and t.options['total_steps']==300
        t.category='entries';t.key=ctx[3];t.action=0
        for _ in range(100):t.update()
        assert t.step==200 and abs(float(t.extract()[0])-2/3)<1e-6
        e.atomic_save(c.snapshot(t),dest/'ENTRY200.private.pt');del t;torch.cuda.empty_cache()
    assert ledger.count['entries']==800
    e.write(root/'FINAL.json',dict(status='COMPLETE',physical=dict(ledger.count),time=time.time()))


def audit(root,config,roles,ledger):
    campaign=Path(config['campaign']);assert e.read(campaign/'jobs/decision_entries/FINAL.json')['status']=='COMPLETE'
    stream=config['stream'];completed=0
    source_rows=[json.loads(l) for l in (Path(config['action_root'])/'ACTION_ROWS.jsonl').read_text().splitlines()]
    for ctx in b.contexts():
        t=b.make(config,roles,ledger,ctx)
        reference=next(x['memory_old'] for x in source_rows if x['context']==ctx[3])
        for step in (100,200):
            entry=torch.load(campaign/f'jobs/decision_entries/{ctx[3]}/ENTRY{step}.private.pt',map_location='cpu',weights_only=False)
            c.restore(t,entry);state=t.extract().tolist();entry_new=scores(t,roles,'Q_train_new',ctx[2])['macro']
            e.atomic_save(entry,root/'PAIRED_ENTRY.private.pt')
            for action in range(9):
                c.restore(t,entry);t.provider.seed=168+stream*10000
                random.seed(860100+stream);np.random.seed(860100+stream);torch.manual_seed(860100+stream);torch.cuda.manual_seed_all(860100+stream)
                t.action=action;t.key=f'{ctx[3]}/step{step}/stream{stream}/action{action}'
                try:row=b.rollout(t,roles,ctx[2],category='audit')
                except BaseException:
                    e.atomic_save(c.snapshot(t),root/'FAILED_STATE.private.pt');raise
                finally:t.provider.seed=168
                gain=.25*(row['new_short']['macro']-entry_new)+.75*(row['new']['macro']-entry_new)
                penalty=max(0.,reference-row['old']['macro']-.005)
                e.append(root/'ACTION_ROWS.jsonl',dict(context=ctx[3],entry_step=step,stream=stream,action=action,state=state,reward=gain-penalty,gain=gain,forget_penalty=penalty,old_frozen_memory_reference=reference,new_entry=entry_new,**row))
                completed+=1;e.write(root/'STATUS.json',dict(status='RUNNING',phase='DENSE_REWARD_AUDIT',completed=completed,total=144,physical=dict(ledger.count),time=time.time()))
        del t;torch.cuda.empty_cache()
    assert completed==144 and ledger.count['audit']==14400
    e.write(root/'FINAL.json',dict(status='COMPLETE',rows=completed,physical=dict(ledger.count),time=time.time()))


def fit(root,config,roles,ledger):
    campaign=Path(config['campaign']);rows=[]
    for stream in (1,2):
        source=campaign/f'jobs/audit_{stream}';assert e.read(source/'FINAL.json')['status']=='COMPLETE'
        rows.extend(json.loads(l) for l in (source/'ACTION_ROWS.jsonl').read_text().splitlines())
    table=dataset(rows,config['sigma_floor_override']);e.write(root/'DENSE_TARGETS.json',table)
    z=torch.tensor([r['state'] for r in table]);target=torch.tensor([r['preferences'] for r in table])
    actor=c.Actor(config['controller']);opt=torch.optim.Adam(actor.parameters(),lr=.001)
    for epoch in range(64):
        value=loss(actor,z,target);assert torch.isfinite(value);opt.zero_grad(set_to_none=True);value.backward()
        gradient=float(torch.nn.utils.clip_grad_norm_(actor.parameters(),1.));ledger.call('actor_prior',f'dense_epoch{epoch}',opt.step)
        e.append(root/'FIT.jsonl',dict(epoch=epoch,loss=float(value.detach()),gradient_norm=gradient))
    e.atomic_save(dict(actor=actor.state_dict(),controller=config['controller'],epochs=64,kind='DENSE_REWARD_SUPERVISION',commit=config['commit']),root/'actor_final.pt')
    e.write(root/'FINAL.json',dict(status='COMPLETE',kind='DENSE_REWARD_SUPERVISION',actor_epochs=64,physical=dict(ledger.count),grpo_success_claim=False,time=time.time()))


def main():
    root=Path(os.environ['EXEC_RUN']);config=e.read(os.environ['EXEC_CONFIG']);ledger=b.JobLedger(root,config['caps'])
    torch.set_num_threads(2);torch.cuda.set_device(0);random.seed(168);np.random.seed(168);torch.manual_seed(168)
    assert config['protocol']=='V83_DENSE_REWARD' and config['training_horizon']==300
    assert e.read(Path(config['prior_campaign'])/'DECISION.json')['status']=='STOP_EPISODE_ALIGNMENT_NO_PRACTICAL_GAIN'
    assert not (root/'STARTED.json').exists(),'create-only; no automatic retry'
    e.write(root/'STARTED.json',dict(pid=os.getpid(),commit=config['commit'],time=time.time()))
    roles=c.split_roles(config['data']);assert roles==e.read(Path(config['action_root'])/'ROLES.private.json')
    config['fixed_best']=e.read(Path(config['action_root'])/'ACTION_DECISION.json')['fixed_best']
    if config['job']=='development':config['fixed_best']=e.read(Path(config['campaign'])/'FIXED_BEST.json')['action']
    config['sigma_floor']=config['sigma_floor_override']
    config['action_rows']=[json.loads(l) for l in (Path(config['action_root'])/'ACTION_ROWS.jsonl').read_text().splitlines()]
    try:
        {'qualification':qualify,'entries':entries,'audit':audit,'fit':fit,'development':b.development}[config['job']](root,config,roles,ledger)
        if config['job']=='qualification':e.write(root/'FINAL.json',dict(status='COMPLETE',physical=dict(ledger.count)))
    except BaseException as exc:
        e.write(root/'STATUS.json',dict(status='ENGINEERING_STOP',error=repr(exc),traceback=traceback.format_exc(),physical=dict(ledger.count)));raise

if __name__=='__main__':main()

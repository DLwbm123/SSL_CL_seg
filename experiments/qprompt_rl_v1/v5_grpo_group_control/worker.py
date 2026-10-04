"""V5 jobs; all student updates use the unmodified native V4 path."""
import copy
import fcntl
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from unittest.mock import patch
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from experiments.qprompt_rl_v1.v4_rl_block_control import worker as v4
from experiments.qprompt_rl_v1.v3b_lrref_endpoint import engine as e, runner as io
from experiments.lcrseg.five_frameworks_v1.native_operations import NativeOperations
from . import protocol as p, controller as q, qualification

ROOT,C=io.ROOT,io.C
CAMPAIGN=Path(C['campaign'])
V4=Path(C['v4_root'])
COST=Counter()

@contextmanager
def measured(name):
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record();wall=time.monotonic()
    try:yield
    finally:
        end.record();end.synchronize()
        COST[name+'_calls']+=1;COST[name+'_cuda_interval_seconds']+=start.elapsed_time(end)/1000
        COST[name+'_wall_seconds']+=time.monotonic()-wall

def datasets(t):return dict(l=t.provider._l,u=t.provider._u,**t.provider.feedback)

def extra(t):
    return dict(options=copy.deepcopy(t.options),physical=t.physical_optimizer_updates,teacher_modes=[m.training for m in t.ema.modules()],provider_identity=t.provider.semantic_metadata(),roles=copy.deepcopy(t.provider.roles),checked={k:copy.deepcopy(d.checked) for k,d in datasets(t).items() if d is not None},requires_restore=t.requires_restore)

def restore_extra(t,s):
    assert t.provider.semantic_metadata()==s['provider_identity'] and t.provider.roles==s['roles']
    t.options=copy.deepcopy(s['options']);t.physical_optimizer_updates=s['physical'];t.requires_restore=s['requires_restore']
    for m,mode in zip(t.ema.modules(),s['teacher_modes']):m.training=mode
    for k,checked in s['checked'].items():datasets(t)[k].checked=copy.deepcopy(checked)

def snapshot(t):return dict(native=e.snapshot(t),extra=extra(t))
def restore(t,s):e.restore(t,s['native']);restore_extra(t,s['extra'])

def save_student(t,path,**metadata):
    e.atomic_save(dict(state=snapshot(t),commit=C['commit'],**metadata),path)

def features(t):
    prior=extra(t)
    with measured('features'):
        z=v4.features(t)
    restore_extra(t,prior)
    return z

def feedback(t,role):
    prior=extra(t)
    with measured('feedback_'+role):value=t.feedback_score(role)
    COST[role+'_label_sample_accesses']+=len(t.provider.feedback[role])
    restore_extra(t,prior)
    return value

def advance(t,ledger,category,key,action,steps=25):
    with measured('student'):
        v4.advance(t,ledger,category,key,action,steps)

def create(seed,domain,development,ledger):
    # Native creation resets all student RNG using the segmentation seed only.
    return v4.create(seed,domain,development,ledger,reused=seed in p.DEVELOPMENT_SEEDS)

def panel_data(domain):
    rows=[]
    for s in p.DEVELOPMENT_SEEDS:rows.extend(e.read(V4/'jobs'/f'panel_{s}_{domain}'/'PANELS.private.json'))
    return rows

def old_initial(domain):
    saved=torch.load(V4/'jobs'/f'fit_{domain}'/'OFFLINE_25.pt',map_location='cpu',weights_only=False)
    rows=panel_data(domain);x=q.normalize([r['x'] for r in rows],saved['scaler'])
    m=q.Policy(saved['state']);softening=q.soften(m,x)
    state=copy.deepcopy(saved['state'])
    state.update({'actor.'+k:v for k,v in m.actor.state_dict().items()})
    return state,saved['scaler'],x,rows,softening

def load_init(domain,method=None):
    value=torch.load(CAMPAIGN/'jobs'/('init_'+domain)/'INITIAL.private.pt',map_location='cpu',weights_only=False)
    model=q.Policy(value['state'],method in p.LEARNERS[:2]);model.eval()
    return model,value

def initialize(ledger):
    d=C['domain'];state,scaler,x,rows,softening=old_initial(d);model=q.Policy(state)
    e.atomic_save(dict(state=state,scaler=scaler,panel=x,initial_logits=model.actor(x).detach()),ROOT/'INITIAL.private.pt')
    e.write(ROOT/'INITIALIZATION.json',dict(domain=d,panel_source='all V4 h25 development states, seed then step',contexts=len(rows),**softening))
    e.write(ROOT/'POLICY_PANEL.private.json',q.panel(model,x,model.actor(x).detach()))
    # Existing action panel recentered to the same-state U0 outcome, diagnostic only.
    diagnostics=[]
    for r in rows:
        on=np.array([a['25']['online'] for a in r['outcomes']]);audit=np.array([a['25']['audit'] for a in r['outcomes']])
        action=int(on.argmax())
        diagnostics.append(dict(seed=r['seed'],step=r['step'],online_relative_U0=(on-on[0]).tolist(),audit_relative_U0=(audit-audit[0]).tolist(),selected=action,audit_best=int(audit.argmax()),selected_audit_gain=float(audit[action]-audit[0])))
    e.write(ROOT/'V4_REWARD_DIAGNOSTICS.json',dict(rows=diagnostics,selection_used=False))

def qualification_job(ledger):
    prior=C.get('qualification_prefix')
    if prior:
        prior=Path(prior)
        assert e.read(prior/'status.json')['physical_calls']=={}, 'no replay of prior student calls'
        passed=e.read(prior/'SYNTHETIC_CHECK.json');assert passed['status']=='PASS'
        e.write(ROOT/'SYNTHETIC_CHECK.json',dict(passed,reused_from=str(prior),no_repeat=True))
        legacy=e.read(prior/'CONTROLLER_SELF_CHECK.json');assert legacy['status']=='PASS'
        with patch.object(v4.policy,'self_check',lambda:legacy):
            v4.smoke(ledger)
    else:
        initials={}
        for d in p.DOMAINS:
            state,_,x,_,_=old_initial(d);initials[d]=(state,x[0])
        e.write(ROOT/'SYNTHETIC_CHECK.json',qualification.run(initials))
        v4.smoke(ledger)
    # Fixed additional CPU correction check, before any performance stage.
    torch.manual_seed(516);model=q.Policy(q.ActorCritic().state_dict());opts=q.optimizers(model)
    x=torch.zeros(40)
    with torch.no_grad():
        dist=q.distribution(model.actor(x))
        row=dict(x=x,action=0,logp=float(dist.log_prob(torch.tensor(0))),probs=dist.probs.tolist(),value=0.,reward=0.)
    steps=[]
    def checked_step(opt,kind,i):steps.append((kind,i));opt.step()
    diag=q.update(model,opts,[[copy.deepcopy(row) for _ in range(4)] for _ in range(4)],[0.,1.,2.,3.],'GRPO_FS_SHUFFLE',1.,517,checked_step)
    assert diag['permutation']==np.random.RandomState(517 ^ 0x5A17).permutation(4).tolist()
    assert len(steps)==16 and sorted(diag['permutation'])==list(range(4))
    e.write(ROOT/'RNG_QUALIFICATION.json',dict(status='PASS',optimizer_calls=16,independent_reward_permutation_stream=True,student_calls=0))
    checks={}
    for d in p.DOMAINS:
        t=create(168,d,True,ledger);entry=snapshot(t)
        features(t);assert e.same(entry,snapshot(t)), 'full feature state mutation'
        for role in ('online','audit'):
            feedback(t,role);assert e.same(entry,snapshot(t))
        actual=[]
        for repeat in range(2):
            restore(t,entry);ledger.smoke_context=f'full/{d}/{repeat}'
            for block,action in enumerate((0,1,2)):advance(t,ledger,'smoke',str(block),action)
            actual.append(snapshot(t))
        assert e.same(*actual),'mixed full-state replay mismatch'
        checks[d]=dict(full_state_exact=True,three_action_blocks=75,feature_immutable=True,stateless_provider=True)
        del t,entry,actual;gc.collect();torch.cuda.empty_cache()
    assert ledger.count['smoke']==p.P0_CALLS
    e.write(ROOT/'QUALIFICATION.json',dict(status='PASS',native_calls=p.P0_CALLS,full_replay=checks,baseline_qualification='V4 unchanged 390-step suite',legacy_synthetic_calls=268))

def reference_identity(t,group):
    return dict(group=group,seed=t.provider.seed,domain=t.provider.domain,provider=t.provider.semantic_metadata(),source=t.provider.stage_source,implementation=C['commit'],baseline=p.BASELINE,development=True)

def reference(ledger):
    d,s,g=C['domain'],C['seed'],C['group'];t=create(s,d,True,ledger)
    entry=snapshot(t);identity=reference_identity(t,g)
    e.atomic_save(dict(state=entry,identity=identity),ROOT/'ENTRY.private.pt')
    before=feedback(t,'online');audit_before=feedback(t,'audit')
    for b in range(p.HORIZONS[d]//25):advance(t,ledger,'reference',str(b),0)
    e.write(ROOT/'REFERENCE.json',dict(identity=identity,online=feedback(t,'online'),audit=feedback(t,'audit'),online_start=before,audit_start=audit_before,student_calls=p.HORIZONS[d],on_policy=False))

def rollout(t,entry,model,init,ledger,category,key,action_seed):
    restore(t,entry);trajectory=[];begin=feedback(t,'online');previous=begin;audit_start=feedback(t,'audit')
    rng=torch.Generator().manual_seed(action_seed);outside=clipped=0
    for b in range(p.HORIZONS[t.provider.domain]//25):
        z=features(t);x=q.normalize(z,init['scaler']).detach();array=np.asarray(z)
        outside+=int(((array<init['scaler']['minimum'])|(array>init['scaler']['maximum'])).sum())
        raw=(array-init['scaler']['mean'])/init['scaler']['sd'];raw[np.asarray(init['scaler']['constant'])]=0
        clipped+=int((np.abs(raw)>10).sum())
        with torch.no_grad():
            logits,value=model(x);dist=q.distribution(logits);action=int(torch.multinomial(dist.probs,1,generator=rng))
            row=dict(x=x,action=action,logp=float(dist.log_prob(torch.tensor(action))),probs=dist.probs.tolist(),value=float(value),reference=False)
        advance(t,ledger,category,f'{key}/{b}',action)
        current=feedback(t,'online');row['reward']=current-previous;previous=current;trajectory.append(row)
        if t.step%100==0 or t.step==p.HORIZONS[t.provider.domain]:
            save_student(t,ROOT/'student_latest.private.pt',key=key)
            e.atomic_save(dict(trajectory=trajectory,action_rng=rng.get_state(),key=key,behavior=e.cpu(model.state_dict())),ROOT/'trajectory_latest.private.pt')
        v4.status(category,key=key,step=t.step)
    assert abs(sum(r['reward'] for r in trajectory)-(previous-begin))<1e-8
    return trajectory,dict(online=previous,audit=feedback(t,'audit'),online_start=begin,audit_start=audit_start,actions=[r['action'] for r in trajectory],out_of_range_fraction=outside/(40*len(trajectory)),normalization_clipped_fraction=clipped/(40*len(trajectory)))

def calibration(ledger):
    d,s=C['domain'],C['seed'];model,init=load_init(d);t=create(s,d,True,ledger);entry=snapshot(t)
    trajectories=[];summaries=[]
    for i in range(4):
        trajectory,summary=rollout(t,entry,model,init,ledger,'calibration',str(i),e.stable('V5/action',400,d,s,i))
        trajectories.append(trajectory);summaries.append(summary)
    restore(t,entry)
    for b in range(p.HORIZONS[d]//25):advance(t,ledger,'calibration',f'U0/{b}',0)
    ref=dict(online=feedback(t,'online'),audit=feedback(t,'audit'))
    e.atomic_save(dict(trajectories=trajectories),ROOT/'CALIBRATION.private.pt')
    e.write(ROOT/'CALIBRATION.json',dict(seed=s,domain=d,trajectories=summaries,reference=ref,relative_rewards=[r['online']-ref['online'] for r in summaries],policy_updated=False))

def scale(ledger):
    d=C['domain'];values=[];trajectories=[]
    for s in p.DEVELOPMENT_SEEDS:
        root=CAMPAIGN/'jobs'/f'cal_{s}_{d}';values.append(e.read(root/'CALIBRATION.json')['relative_rewards'])
        trajectories.extend(torch.load(root/'CALIBRATION.private.pt',map_location='cpu',weights_only=False)['trajectories'])
    a=np.array(values);sr=max(1e-4,float(np.sqrt(np.mean((a-a.mean(1,keepdims=True))**2))))
    e.write(ROOT/'SCALE.json',dict(domain=d,S_R=sr,groups=3,trajectories=12,online_only=True,dynamic_updates=0))
    for method in p.LEARNERS[:2]:
        model,_=load_init(d,method)
        diagnostics=q.warmup(model,trajectories,method,sr,lambda opt,i:ledger.step(opt,'critic_warmup',f'{method}/{i}'))
        e.atomic_save(dict(critic=e.cpu(model.critic.state_dict())),ROOT/(method+'.private.pt'))
        e.write(ROOT/(method+'_WARMUP.json'),diagnostics)

def learn(ledger):
    d,m,c=C['domain'],C['method'],C['controller'];model,init=load_init(d,m)
    sr=e.read(CAMPAIGN/'jobs'/('scale_'+d)/'SCALE.json')['S_R']
    if model.critic is not None:
        warm=torch.load(CAMPAIGN/'jobs'/('scale_'+d)/(m+'.private.pt'),map_location='cpu',weights_only=False)
        model.critic.load_state_dict(warm['critic'])
    opts=q.optimizers(model);summaries=[];panels=[]
    for g in range(6):
        s=p.DEVELOPMENT_SEEDS[g%3];t=create(s,d,True,ledger);entry=snapshot(t)
        refroot=CAMPAIGN/'jobs'/f'ref_{g}_{d}';ref=e.read(refroot/'REFERENCE.json')
        cached=torch.load(refroot/'ENTRY.private.pt',map_location='cpu',weights_only=False)
        assert ref['identity']==reference_identity(t,g)==cached['identity'] and e.same(entry,cached['state']), 'U0 cache state/provenance mismatch'
        trajectories=[];out=[];behavior=e.cpu(model.state_dict());old_logits=model.actor(init['panel']).detach()
        for i in range(4):
            traj,summary=rollout(t,entry,model,init,ledger,'development',f'{g}/{i}',e.stable('V5/action',c,d,g,i))
            trajectories.append(traj);out.append(summary)
            assert e.same(behavior,e.cpu(model.state_dict())),'behavior changed inside group'
        rewards=[r['online']-ref['online'] for r in out]
        # Preserve the complete pre-update batch before any optimizer call.
        e.atomic_save(dict(trajectories=trajectories,rewards=rewards,behavior=behavior,optimizers=[o.state_dict() if o else None for o in opts],group=g),ROOT/'group_latest.private.pt')
        student_before=snapshot(t)
        diag=q.update(model,opts,trajectories,rewards,m,sr,e.stable('V5/update',c,d,g),lambda opt,kind,k:ledger.step(opt,kind,f'{g}/{k}'),clip_high=.28 if m=='GRPO_FS_CLIPHI' else .2)
        assert e.same(student_before,snapshot(t)), 'controller update mutated student state'
        del student_before
        record=dict(group=g,seed=s,domain=d,controller=c,method=m,trajectories=out,reference=ref,relative_rewards=rewards,reward_mean=float(np.mean(rewards)),reward_population_std=float(np.std(rewards)),unique_sequences=len({tuple(r['actions']) for r in out}),audit_order=np.argsort([r['audit'] for r in out]).tolist(),online_order=np.argsort([r['online'] for r in out]).tolist(),**diag)
        summaries.append(record);e.write(ROOT/'GROUP_DIAGNOSTICS.json',summaries)
        panels.append(dict(group=g,**q.panel(model,init['panel'],init['initial_logits'],old_logits)))
        e.write(ROOT/'POLICY_PANEL.private.json',panels)
        e.atomic_save(dict(state=e.cpu(model.state_dict()),optimizers=[e.cpu(o.state_dict()) if o else None for o in opts],group=g,scaler=init['scaler'],S_R=sr,commit=C['commit']),ROOT/'latest_policy.private.pt')
        del t,entry,cached,trajectories;gc.collect();torch.cuda.empty_cache()
    e.atomic_save(dict(state=e.cpu(model.state_dict()),scaler=init['scaler'],method=m,controller=c,groups=6,commit=C['commit']),ROOT/'FROZEN_POLICY.private.pt')

def load_final(d,m,c):
    raw=torch.load(CAMPAIGN/'jobs'/f'learn_{c}_{d}_{m}'/'FROZEN_POLICY.private.pt',map_location='cpu',weights_only=False)
    base,_=load_init(d,m);base.load_state_dict(raw['state']);base.eval()
    return base,raw['scaler']

def endpoint(ledger):
    d,s,stage=C['domain'],C['seed'],C['stage'];t=create(s,d,False,ledger);entry=snapshot(t)
    save_student(t,ROOT/'ENTRY.private.pt');io.export(t,ROOT/'entry.pt')
    schedules={};endpoints=[]
    for m,c in p.endpoint_methods(stage):
        restore(t,entry);key=m+(f'_{c}' if c else '');actions=[];outside=clipped=0;model=scaler=None
        if m=='OFFLINE_25':
            old=torch.load(V4/'jobs'/f'fit_{d}'/'OFFLINE_25.pt',map_location='cpu',weights_only=False);model=q.Policy(old['state']);scaler=old['scaler']
        elif m in p.LEARNERS or m=='GRPO_FS_CLIPHI':model,scaler=load_final(d,m,c)
        if model is not None:policy_before=e.cpu(model.state_dict())
        if m=='PHASE_SHUFFLE_FS':
            original=schedules['GRPO_FS_'+str(c)];shuffled=list(original)
            for l,r in p.phases(d):
                order=np.random.RandomState(e.stable('V5/phase',s,d,c,l)).permutation(r-l)
                shuffled[l:r]=[original[l+i] for i in order];assert Counter(shuffled[l:r])==Counter(original[l:r])
        # Loading CPU controllers must not change the restored student random streams.
        e.rng_restore(entry['native']['rng']);started=time.time()
        for b in range(p.HORIZONS[d]//25):
            if m in ('ORIGINAL','FINE_05'):action=0 if m=='ORIGINAL' else 2
            elif m=='PHASE_SHUFFLE_FS':action=shuffled[b]
            else:
                z=features(t);array=np.asarray(z);x=q.normalize(z,scaler)
                with torch.no_grad():logits=model.actor(x)
                action=next(a for a in (2,1,0) if float(logits.max()-logits[a])<=1e-8)
                outside+=int(((array<scaler['minimum'])|(array>scaler['maximum'])).sum())
                raw=(array-scaler['mean'])/scaler['sd'];raw[np.asarray(scaler['constant'])]=0;clipped+=int((abs(raw)>10).sum())
            actions.append(action);advance(t,ledger,'endpoint',f'{key}/{b}',action)
            e.append(ROOT/'ACTIONS.jsonl',dict(method=m,controller=c,block=b,action=action))
            if t.step%100==0 or t.step==p.HORIZONS[d]:save_student(t,ROOT/'student_latest.private.pt',method=m,controller=c)
            v4.status('ENDPOINT_TRAINING',stage=stage,seed=s,domain=d,method=m,controller=c,step=t.step)
        if model is not None:assert e.same(policy_before,e.cpu(model.state_dict()))
        io.export(t,ROOT/(key+'.pt'));schedules[key]=actions
        endpoints.append(dict(seed=s,domain=d,method=m,controller=c,step=t.step,path=str(ROOT/(key+'.pt')),training_wall_seconds=time.time()-started,action_counts=[actions.count(i) for i in range(3)],out_of_range_fraction=outside/(40*len(actions)) if model else None,normalization_clipped_fraction=clipped/(40*len(actions)) if model else None))
        e.write(ROOT/'ENDPOINT_REGISTRY.json',endpoints)
    e.write(ROOT/'FREEZE.json',dict(status='FROZEN',time=time.time(),controller_updates=0,reward_accesses=0,endpoints=len(endpoints),commit=C['commit']))

def evaluate_file(path):
    env=dict(os.environ,EVAL_INPUT=str(path),EXEC_MODULE='experiments.qprompt_rl_v1.v5_grpo_group_control.evaluator')
    with (ROOT/'evaluator.log').open('a') as log:
        subprocess.run([sys.executable,'-c','import os,runpy;runpy.run_module(os.environ["EXEC_MODULE"],run_name="__main__")'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)

def evaluate(ledger):
    stage=C['stage'];assert (CAMPAIGN/(stage.upper()+'_ENDPOINT_LOCK.json')).exists()
    target=Path(C['target']);registry=e.read(target/'ENDPOINT_REGISTRY.json')
    evaluate_file(target/'entry.pt');start=e.read(target/'entry.scores.json')['scores'];rows=[]
    for endpoint in registry:
        path=Path(endpoint['path']);evaluate_file(path);scores=e.read(path.with_suffix('.scores.json'))['scores'];d=endpoint['domain']
        rows.append(dict({k:v for k,v in endpoint.items() if k!='path'},macro_Dice=scores[d]['macro_Dice'],rim=scores[d]['rim'],cup=scores[d]['cup'],disc_union=scores[d]['disc_union'],old_REFUGE=scores['REFUGE']['macro_Dice'],old_change=scores['REFUGE']['macro_Dice']-start['REFUGE']['macro_Dice']))
    e.write(ROOT/'RESULTS.json',rows)

def audit(ledger):
    path=ROOT/'PHYSICAL_LEDGER.jsonl'
    rows=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    a=Counter((r['category'],r['key']) for r in rows if r['event']=='attempt');b=Counter((r['category'],r['key']) for r in rows if r['event']=='success')
    assert a==b and all(n==1 for n in a.values()) and not any(r['event']=='failure' for r in rows)
    assert all(ledger.count[k]==v for k,v in C['caps'].items())
    assert not set(ledger.count)-set(C['caps'])
    e.write(ROOT/'COMPLETION_AUDIT.json',dict(status='PASS',physical_calls=dict(ledger.count),duplicate_keys=0,failures=0,commit=C['commit']))

def main():
    p.budget();torch.set_num_threads(2);torch.cuda.set_device(0)
    lock=(ROOT/'EXECUTOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'PHYSICAL_LEDGER.jsonl').exists(),'no automatic retry'
    ledger=v4.Ledger(ROOT);ledger.caps=dict(C['caps']);ledger.seen=set();v4.status('STARTING',job=C['job'])
    functions=dict(qualification=qualification_job,initialize=initialize,reference=reference,calibration=calibration,scale=scale,learn=learn,endpoint=endpoint,evaluate=evaluate,source=lambda l:v4.source(C['seed'],l,{}))
    try:
        functions[C['job']](ledger);audit(ledger)
        e.write(ROOT/'FINAL.json',dict(status='COMPLETE',commit=C['commit'],time=time.time(),physical_calls=dict(ledger.count),peak_cuda_allocated=torch.cuda.max_memory_allocated()))
        v4.status('COMPLETE',job=C['job'])
    except BaseException as exc:
        v4.status('FAILED',error=repr(exc),traceback=traceback.format_exc(),physical_calls=dict(ledger.count));raise
    finally:e.write(ROOT/'MEASURED_COSTS.json',dict(COST))

if __name__=='__main__':
    with NativeOperations(ROOT/'operations'):main()

"""One finite, serial GPU queue. No score gates and no automatic retries."""
import csv, gc, math, os, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from .engine import *
from . import policy

ROOT=Path(os.environ['EXEC_RUN'])
C=read(os.environ['EXEC_CONFIG'])

def export(t,path):
    model=t.model.deploy()
    atomic_save(dict(student=cpu(model.parent.native.state_dict()),seed=t.provider.seed,domain=t.provider.domain,step=t.step),path)
    del model

def evaluate_file(path):
    assert time.time()<read(ROOT/'SESSION.json')['hard_deadline'],'HARD_DEADLINE'
    env=dict(os.environ,EVAL_INPUT=str(path),EXEC_MODULE='experiments.qprompt_rl_v1.v3b_lrref_endpoint.evaluator')
    start=time.time()
    with (ROOT/'evaluator.log').open('a') as log:
        subprocess.run([sys.executable,'-c','import os,runpy;runpy.run_module(os.environ["EXEC_MODULE"],run_name="__main__")'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,read(ROOT/'SESSION.json')['hard_deadline']-time.time()))
    return time.time()-start

def extract(t,full,cost):
    start=time.time();append(ROOT/'FEATURE_LEDGER.jsonl',dict(event='attempt',full=full,time=start,step=t.step));z=t.extract(full);torch.cuda.synchronize();append(ROOT/'FEATURE_LEDGER.jsonl',dict(event='success',full=full,time=time.time(),step=t.step))
    kind='effect' if full else 'base';cost[kind+'_extractions']=cost.get(kind+'_extractions',0)+1
    cost['feature_seconds']=cost.get('feature_seconds',0)+time.time()-start
    if full:
        cost['VJP']=cost.get('VJP',0)+4;cost['virtual_previews']=cost.get('virtual_previews',0)+3
    return z

def save_state(t,path,**extra):atomic_save(dict(state=snapshot(t),commit=C['commit'],**extra),path)

def check_native(t,ledger,cost):
    root=snapshot(t);z=extract(t,True,cost);assert same(root,snapshot(t)), 'feature mutation'
    # Actual optimizer and native A-side operation vs scratch previews, each action.
    rows=[]
    for action in range(3):
        restore(t,root);t.action=action
        with t.readonly():
            l,u,_=t.losses();params=[p for p in t.model.parameters() if p.requires_grad]
            grads=torch.autograd.grad(l if u is None else l+u,params,allow_unused=True)
            values,states=effect.preview(t.optimizer,params,grads)
        ledger.update(t,'smoke',f'parity/{t.provider.domain}/{action}',action)
        error=max(float((p-v).abs().max()) for p,v in zip(params,values))
        assert error<1e-7,(action,error)
        for i,p in enumerate(params):
            if i in states:
                for k,v in states[i].items():assert torch.allclose(t.optimizer.state[p][k].cpu(),v.cpu(),atol=1e-7,rtol=1e-6)
        t.model.parent.apply_constraints();rows.append(dict(action=action,max_abs_error=error))
    # Nonempty moments, snapshot ownership and exact repeated continuation.
    branch=snapshot(t);stored=copy.deepcopy(branch);extract(t,True,cost);assert same(branch,snapshot(t))
    ledger.update(t,'smoke',f'restore/{t.provider.domain}/a',2);a=snapshot(t)
    restore(t,branch);ledger.update(t,'smoke',f'restore/{t.provider.domain}/b',2);b=snapshot(t)
    assert same(a,b),'resume trajectory mismatch';assert same(branch,stored),'snapshot alias'
    restore(t,root)
    return dict(parity=rows,feature_immutable=True,resume_exact=True,snapshot_owned=True,head_trainable=False,body_names=[n for n,p in t.model.named_parameters() if p.requires_grad])

def smoke(ledger,cost):
    times={};audit={};eval_seconds=0.;entry_seconds=0.
    for domain in DOMAINS:
        start=time.time();t=create(C,161,domain,True,ledger);entry_seconds+=time.time()-start
        write(ROOT/f'DATA_ROLES_{domain}.private.json',t.provider.roles)
        audit[domain]=check_native(t,ledger,cost);entry=snapshot(t)
        # Every method measured as a full 5-step block, including restore and durable save.
        for method in METHODS:
            start=time.time();restore(t,entry)
            if method in ('EFFECT_POLICY','EFFECT_RULE'):extract(t,True,cost)
            elif method.startswith('BASE'):extract(t,False,cost)
            for k in range(5):ledger.update(t,'smoke',f'timing/{domain}/{method}/{k}',-1 if method=='ORIGINAL' else 2)
            save_state(t,ROOT/'smoke_latest.pt',method=method);torch.cuda.synchronize()
            times[domain+'/'+method]=time.time()-start
        restore(t,entry);path=ROOT/f'smoke_entry_{domain}.pt';export(t,path);del t;gc.collect();torch.cuda.empty_cache()
        eval_seconds+=evaluate_file(path)
    write(ROOT/'INTEGRATION_CHECK.json',dict(status='PASS',checks=audit,smoke_calls=ledger.count['smoke']))
    remaining=ledger.session['optimizer_deadline']-time.time()
    # Two pilot seeds per domain. Include measured expensive per-block saving even
    # though production saves every 100 steps; use serial GPU timings, not GPU count.
    estimates={}
    for h in (1200,600,300):
        training=2*(h/5)*sum(times.values())
        development=1080/5*max(times.values())+32*cost.get('feature_seconds',0)/max(1,cost.get('effect_extractions',0))+1024*.02
        overhead=entry_seconds*4+eval_seconds*16+600
        estimates[h]=1.25*(training+development)+overhead
    h=next((h for h in (1200,600,300) if estimates[h]<=remaining),None)
    write(ROOT/'THROUGHPUT.json',dict(block_seconds=times,entry_seconds=entry_seconds,evaluation_seconds=eval_seconds,estimates=estimates,remaining=remaining,chosen=h))
    if h is None:raise RuntimeError('complete 300-step matrix does not fit measured resource window')
    lock=dict(protocol='V3B-LRREF-ENDPOINT',commit=C['commit'],H=h,methods=METHODS,domains=DOMAINS,development_seed=161,pilot_seeds=[162,163],queue='seed then domain, round-robin methods in five-step blocks',session=ledger.session,original_KI_identity='unverified',native_schedule={'RIM_ONE_r3':3200,'Drishti_GS':2100},bridge='native L-only nonU + 0.5 U from step 0; action-specific .7 mask; dense EMA .99; original sixth control',feature_blocks='all trainable native A/B BODY; frozen HEAD empty, flags zero',student_cap=24*h+1080,smoke_cap=240,replay_cap=math.ceil(.05*(24*h+1080+240)),checkpoints=[x for x in (0,300,600,1200) if x<=h],inference='merged final student; deterministic native linear readout',metric='case-mean rim/cup Dice macro; domains/seeds equal',validation='independent evaluator; never feedback',push_authorized=False)
    write(ROOT/'RUN_LOCK.json',lock);ledger.caps['main']=24*h;ledger.caps['replay']=lock['replay_cap'];return lock

def development(ledger,cost):
    fit={};states=[5*math.floor(59*j/15) for j in range(16)]
    for domain in DOMAINS:
        t=create(C,161,domain,True,ledger);rows=[]
        for step in range(300):
            if step in states:
                root=snapshot(t);z=extract(t,True,cost);out=[]
                for action in range(3):
                    restore(t,root)
                    for k in range(5):ledger.update(t,'development',f'{domain}/branch/{step}/{action}/{k}',action if k==0 else 2)
                    out.append([t.feedback_score('online'),t.feedback_score('audit')])
                restore(t,root);assert same(root,snapshot(t));out=np.array(out)
                rows.append(dict(step=step,x=z,online=(out[:,0]-out[2,0]).tolist(),audit=(out[:,1]-out[2,1]).tolist()))
                write(ROOT/f'development_{domain}.private.json',rows)
            ledger.update(t,'development',f'{domain}/retained/{step}',2)
            if t.step==1 or t.step%25==0:save_state(t,ROOT/f'development_{domain}.pt',phase='retained_development')
        x=np.array([r['x'] for r in rows]);raw=np.array([r['online'] for r in rows]);scale=max(1e-4,float(np.sqrt(np.mean(raw[:,:2]**2))))
        fit[domain]={m:policy.fit(m,x,raw/scale,stable('V3B',domain),ledger,domain+'/'+m) for m in ('BASE_POLICY','EFFECT_POLICY','BASE_RIDGE')}
        fit[domain].update(FINE=dict(method='FINE'),EFFECT_RULE=dict(method='EFFECT_RULE'),reward_scale=scale)
        del t;gc.collect();torch.cuda.empty_cache()
    assert ledger.count['development']==1080 and ledger.count['controller']==1024
    write(ROOT/'FROZEN_CONTROLLERS.private.json',fit);write(ROOT/'CONTROLLER_FREEZE.json',dict(time=time.time(),student_development_calls=1080,controller_calls=1024,main_controller_updates=0,commit=C['commit']))
    return fit

def main_matrix(ledger,cost,lock,fits):
    endpoints=[]
    for seed in (162,163):
        for domain in DOMAINS:
            t=create(C,seed,domain,False,ledger);entry=snapshot(t);states={m:entry for m in METHODS};stats={m:dict(actions=[0,0,0],out_of_range=0,features=0) for m in METHODS}
            cell=ROOT/f'S{seed}_{domain}';cell.mkdir(exist_ok=True)
            export(t,cell/'entry.pt');save_state(t,cell/'entry_state.pt',phase='main_entry')
            for step in range(0,lock['H'],5):
                for method in METHODS:
                    restore(t,states[method]);assert t.step==step
                    action=-1 if method=='ORIGINAL' else 2
                    if method not in ('ORIGINAL','FINE'):
                        z=extract(t,method.startswith('EFFECT'),cost);f=fits[domain][method]
                        rr=cpu(rng_state());action=policy.predict(f,[z])['actions'][0];rng_restore(rr)
                        if 'scaler' in f:
                            zz=np.array(z)
                            if method.startswith('BASE'):zz[16:]=0
                            s=f['scaler'];stats[method]['out_of_range']+=int(((zz<np.array(s['train_min']))|(zz>np.array(s['train_max']))).sum());stats[method]['features']+=40
                        append(cell/(method+'_actions.private.jsonl'),dict(step=step,action=action))
                    for k in range(5):
                        a=(-1 if method=='ORIGINAL' else action if k==0 else 2)
                        ledger.update(t,'main',f'S{seed}/{domain}/{method}/{step+k}',a)
                        if a>=0:stats[method]['actions'][a]+=1
                    states[method]=snapshot(t)
                    if t.step%100==0 or t.step==lock['H']:save_state(t,cell/(method+'_latest.pt'),method=method,stats=stats[method],controller_commit=C['commit'])
                    if t.step in lock['checkpoints']:export(t,cell/f'{method}_{t.step}.pt')
                write(ROOT/'MATRIX_PROGRESS.json',dict(seed=seed,domain=domain,all_methods_step=step+5,H=lock['H'],physical=dict(ledger.count)))
            for method in METHODS:endpoints.append(dict(seed=seed,domain=domain,method=method,step=lock['H'],path=str(cell/f'{method}_{lock["H"]}.pt'),entry=str(cell/'entry.pt'),**stats[method]))
            write(ROOT/'ENDPOINT_REGISTRY.json',endpoints);del t,states,entry;gc.collect();torch.cuda.empty_cache()
    return endpoints

def report(endpoints,ledger,cost,lock):
    files=list(dict.fromkeys([r['entry'] for r in endpoints]+[r['path'] for r in endpoints]));seconds=0.
    for p in files:seconds+=evaluate_file(Path(p))
    rows=[]
    for r in endpoints:
        out=read(Path(r['path']).with_suffix('.scores.json'))['scores'];entry=read(Path(r['entry']).with_suffix('.scores.json'))['scores'];d=r['domain']
        rows.append({k:r[k] for k in ('seed','domain','method','step')}|dict(macro_Dice=out[d]['macro_Dice'],rim=out[d]['rim'],cup=out[d]['cup'],disc_union=out[d]['disc_union'],entry_macro=entry[d]['macro_Dice'],old_REFUGE=out['REFUGE']['macro_Dice'],old_change=out['REFUGE']['macro_Dice']-entry['REFUGE']['macro_Dice'],skip=r['actions'][0],coarse=r['actions'][1],fine=r['actions'][2],out_of_range_fraction=r['out_of_range']/max(1,r['features']),missing=False))
    lookup={(r['seed'],r['domain'],r['method']):r for r in rows}
    for r in rows:
        for ref in ('ORIGINAL','FINE','BASE_POLICY','BASE_RIDGE','EFFECT_RULE'):r['delta_vs_'+ref]=r['macro_Dice']-lookup[r['seed'],r['domain'],ref]['macro_Dice']
    with (ROOT/'ENDPOINTS.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    means={m:float(np.mean([r['macro_Dice'] for r in rows if r['method']==m])) for m in METHODS}
    seedmeans={m:{str(s):float(np.mean([r['macro_Dice'] for r in rows if r['method']==m and r['seed']==s])) for s in (162,163)} for m in METHODS}
    comparisons={}
    for ref in ('ORIGINAL','FINE','BASE_POLICY','BASE_RIDGE','EFFECT_RULE'):
        diffs=[seedmeans['EFFECT_POLICY'][str(s)]-seedmeans[ref][str(s)] for s in (162,163)]
        comparisons[ref]=dict(seed_deltas=diffs,mean=float(np.mean(diffs)),sample_std=float(np.std(diffs,ddof=1)))
    write(ROOT/'SUMMARY.json',dict(means=means,seed_means=seedmeans,paired_comparisons=comparisons,negative_cells=[r for r in rows if r['method']=='EFFECT_POLICY' and r['delta_vs_ORIGINAL']<0]))
    cost.update(physical_calls=dict(ledger.count),evaluation_seconds=seconds,peak_cuda_allocated=torch.cuda.max_memory_allocated(),wall_seconds=time.time()-ledger.session['start'])
    write(ROOT/'ALL_COSTS.json',cost)
    text='# V3B-LRREF-ENDPOINT\n\n24/24 fixed-budget student endpoints completed. Original KI historical identity remains unverified. This is a local native-reference pilot, not a complete CL sequence.\n\n'
    text+='\n'.join(f'- {m}: {v:.6f}' for m,v in means.items())
    text+='\n\nEFFECT_POLICY paired comparisons (2 optimization seeds, domain means):\n'+json.dumps(comparisons,indent=2)
    text+='\n\nNo significance or RL-specific superiority claim. Old-domain change is not full-sequence BWT. Negative cells are retained. No automatic follow-up experiment and no push authorization.\n'
    (ROOT/'FINAL_INTERPRETATION.md').write_text(text);write(ROOT/'status.json',dict(status='COMPLETE_PRIVATE',time=time.time(),endpoints=len(rows),H=lock['H']))

def main():
    torch.set_num_threads(2);torch.cuda.set_device(0);ledger=Ledger(ROOT);cost={}
    # Finite process has exclusive ownership; no silent continuation of partial runs.
    import fcntl
    lockfile=(ROOT/'EXECUTOR.lock').open('a');fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'RUN_LOCK.json').exists(),'existing run requires explicit recovery binding'
    try:
        write(ROOT/'status.json',dict(status='INTEGRATION_SMOKE',pid=os.getpid(),time=time.time()))
        lock=smoke(ledger,cost);write(ROOT/'status.json',dict(status='DEVELOPMENT',pid=os.getpid(),H=lock['H'],time=time.time()))
        fits=development(ledger,cost);write(ROOT/'status.json',dict(status='MAIN_MATRIX',pid=os.getpid(),H=lock['H'],time=time.time()))
        endpoints=main_matrix(ledger,cost,lock,fits);write(ROOT/'status.json',dict(status='EVALUATION',pid=os.getpid(),time=time.time()))
        report(endpoints,ledger,cost,lock)
    except BaseException as e:
        cost.update(physical_calls=dict(ledger.count),wall_seconds=time.time()-ledger.session['start']);write(ROOT/'ALL_COSTS.json',cost)
        write(ROOT/'status.json',dict(status='STOPPED',error=repr(e),traceback=traceback.format_exc(),time=time.time(),pid=os.getpid()));raise
if __name__=='__main__':
    from experiments.lcrseg.five_frameworks_v1.native_operations import NativeOperations
    with NativeOperations(ROOT/'operations'):main()

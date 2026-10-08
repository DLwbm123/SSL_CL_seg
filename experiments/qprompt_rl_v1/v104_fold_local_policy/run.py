"""Fold-local matched objectives using frozen states and reward tables only."""
import copy
import fcntl
import importlib.util
import json
import os
import runpy
import time
from pathlib import Path
import numpy as np
import torch

CAPS=dict(qualification=2,warmup=20480,CE=20480,RL=20480)


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def fresh(v,seed):
    rng=torch.get_rng_state();actor=v.model12(seed)
    torch.nn.init.zeros_(actor.net[-1].weight);torch.nn.init.zeros_(actor.net[-1].bias)
    assert torch.equal(rng,torch.get_rng_state())
    return actor


def prepare(x,returns):
    mean=x.mean(0);std=x.std(0,unbiased=False);std=torch.where(std<1e-6,torch.ones_like(std),std)
    center=returns[:,:9].mean(-1,keepdim=True);scale=(returns[:,:9]-center).square().mean().sqrt().clamp_min(1e-4)
    return mean,std,((returns-center)/scale).clamp(-3,3),scale


def update(actor,opt,z,adv,prior,rl,b,budget,category,key):
    loss=b.actor_loss(actor,z,adv,prior,rl);assert torch.isfinite(loss)
    opt.zero_grad(set_to_none=True);loss.backward();assert torch.isfinite(torch.nn.utils.clip_grad_norm_(actor.parameters(),1.))
    budget.step(category,key,opt);return float(loss.detach())


def selfcheck(v,b,budget):
    a=fresh(v,601);other=fresh(v,601);assert all(torch.equal(x,other.state_dict()[k]) for k,x in a.state_dict().items())
    x=torch.stack([torch.zeros(24),torch.ones(24)]);x[:,0]=1
    reward=torch.arange(24,dtype=torch.float32).reshape(2,12)/100;mean,std,adv,scale=prepare(x,reward);assert std[0]==1 and scale>=1e-4 and torch.isfinite(adv).all()
    heldout=torch.ones((1,12));heldout.fill_(1e9);assert all(torch.equal(u,w) for u,w in zip((mean,std,adv,scale),prepare(x,reward)))
    for rl in (False,True):
        actor=copy.deepcopy(a);z=(x-mean)/std;prior=actor(z).detach().log_softmax(-1);opt=torch.optim.Adam(actor.parameters(),lr=.001)
        before=update(actor,opt,z,adv,prior,rl,b,budget,'qualification','RL' if rl else 'CE')
        assert float(b.actor_loss(actor,z,adv,prior,rl).detach())<before


def train_pair(x,returns,seed,key,v,b,d,budget,root):
    # Only the train fold is passed here, including its normalization targets.
    mean,std,adv,scale=prepare(x,returns);z=(x-mean)/std;warm=fresh(v,seed);prior=warm(z).detach().log_softmax(-1);logs=[]
    def fit(actor,rl,category):
        opt=torch.optim.Adam(actor.parameters(),lr=.001)
        for step in range(512):
            loss=update(actor,opt,z,adv,prior,rl,b,budget,category,key+f'/{step}')
            if step in (0,127,511):logs.append(dict(key=key,arm=category,step=step+1,loss_before_update=loss))
    fit(warm,False,'warmup');prior=warm(z).detach().log_softmax(-1)
    models={'WARM':warm}
    for arm in ('CE','RL'):
        actor=copy.deepcopy(warm);assert all(torch.equal(t,warm.state_dict()[k]) for k,t in actor.state_dict().items())
        fit(actor,arm=='RL',arm);models[arm]=actor
    metrics=[];unique,inverse=torch.unique(z,dim=0,return_inverse=True)
    with torch.no_grad():
        lp0=prior.double();lp0-=lp0.logsumexp(-1,keepdim=True)
        for arm,actor in models.items():
            log=actor(z).log_softmax(-1).double();log-=log.logsumexp(-1,keepdim=True);prob=log.exp();gap=0.;best=0.;value=0.
            for i in range(len(unique)):
                ids=(inverse==i).nonzero().flatten();a=adv[ids].double().mean(0,keepdim=True);p0=lp0[ids[:1]];lp=log[ids[:1]]
                assert torch.equal(lp0[ids],p0.expand(len(ids),-1)) and torch.equal(log[ids],lp.expand(len(ids),-1))
                star=d.optimum(a,p0);jstar=float(d.objective(star,a,p0));j=float(d.objective(lp,a,p0));kl=float(.26*(lp.exp()*(lp-star)).sum());assert abs((jstar-j)-kl)<2e-7
                w=len(ids)/len(z);gap+=w*(jstar-j);best+=w*jstar;value+=w*j
            metrics.append(dict(key=key,arm=arm,train_groups=len(x),unique_train_states=len(unique),reward_scale=float(scale),clipped_values=int((adv.abs()==3).sum()),expected_dense=float((prob*returns.double()).sum(-1).mean()),expected_advantage=float((prob*adv.double()).sum(-1).mean()),entropy=float(-(prob*log).sum(-1).mean()),KL_to_warm=float((prob*(log-lp0)).sum(-1).mean()),CE_loss=float(b.actor_loss(actor,z,adv,prior,False)),RL_objective=value,optimal_RL_objective=best,RL_objective_gap=gap))
            with (root/(key.replace('/','_')+'_'+arm+'.private.pt')).open('xb') as f:torch.save(dict(actor=actor.state_dict(),mean=mean,std=std,scale=scale,seed=seed),f)
    return metrics,logs


def main(root,cfg):
    torch.set_num_threads(1);v=load(cfg['v99_entry'],'actor12');b=load(cfg['base_entry'],'losses');d=load(cfg['diagnostic_entry'],'diagnostic')
    budget=runpy.run_path(cfg['budget_helper'])['Budget'](root,CAPS);selfcheck(v,b,budget)
    d.write(root/'SYNTHETIC_QUALIFICATION.json',dict(status='PASS',optimizer_updates=2,model_seed_identity=True,constructor_rng_unchanged=True,constant_feature_handling=True,CE_RL_finite_decreasing=True))
    keys,original,reward,gain,old=d.dataset(cfg,'V101');saved=np.load(Path(cfg['v103'])/'FEATURES.private.npz');assert np.array_equal(saved['original'],original)
    features={'ORIGINAL':original,'FIXED_MEAN4':saved['probes'].mean(1)};assert all(x.shape==(32,24) and np.isfinite(x).all() for x in features.values())
    folds=[('STREAM_SWAP',f'stream{s}',[i for i,k in enumerate(keys) if k[1]==s]) for s in (1,2)]+[('CONDITION_LOCO',ctx,[i for i,k in enumerate(keys) if k[0]==ctx]) for ctx in sorted({k[0] for k in keys})]
    plans=[];fitrows=[];fitlogs=[];returns=torch.tensor(reward,dtype=torch.float32)
    for representation,x in features.items():
        x=torch.tensor(x,dtype=torch.float32)
        for protocol,fold,test in folds:
            train=[i for i in range(32) if i not in test];assert not set(train)&set(test)
            for seed in (601,602):
                key=f'{representation}/{protocol}/{fold}/{seed}';metrics,logs=train_pair(x[train],returns[train],seed,key,v,b,d,budget,root)
                fitrows+=metrics;fitlogs+=logs;plans.append(dict(representation=representation,protocol=protocol,fold=fold,seed=seed,key=key,train=train,test=test))
                b.write(root/'STATUS.json',dict(status='RUNNING',phase='fit',completed_pairs=len(plans),total_pairs=40,physical=dict(budget.count)))
    assert dict(budget.count)==CAPS and len(plans)==40 and len(fitrows)==120
    d.write(root/'FIT_LOCK.json',dict(status='SEALED_BEFORE_HELDOUT_SCORING',models=120,pairs=40,physical=dict(budget.count),time=time.time()))
    d.write(root/'FIT_SUMMARY.json',fitrows);d.table(root/'FIT_SUMMARY.csv',fitrows);d.table(root/'FIT_LOG.csv',fitlogs)
    rows=[]
    for plan in plans:
        x=torch.tensor(features[plan['representation']],dtype=torch.float32)
        for arm in ('WARM','CE','RL'):
            checkpoint=torch.load(root/(plan['key'].replace('/','_')+'_'+arm+'.private.pt'),map_location='cpu',weights_only=False);actor=fresh(v,plan['seed']);actor.load_state_dict(checkpoint['actor']);test=plan['test']
            with torch.no_grad():prob=actor((x[test]-checkpoint['mean'])/checkpoint['std']).softmax(-1).double().numpy();prob/=prob.sum(-1,keepdims=True)
            for offset,i in enumerate(test):
                p=prob[offset];action=int(p.argmax())
                for mode in ('EXACT_EXPECTATION','ARGMAX'):
                    weight=p if mode=='EXACT_EXPECTATION' else np.eye(12)[action]
                    r=dict(representation=plan['representation'],protocol=plan['protocol'],fold=plan['fold'],seed=plan['seed'],arm=arm,mode=mode,context=keys[i][0],stream=keys[i][1],step=keys[i][2],argmax_action=action,train_groups=len(plan['train']),test_groups=len(test))
                    for metric,values in [('dense',reward),('gain',gain),('old_change',old)]:r[metric]=float(weight@values[i])
                    r['regret']=float(reward[i].max()-r['dense']);r.update({f'p{a}':float(p[a]) for a in range(12)});rows.append(r)
    assert len(rows)==1536
    controls=[r for r in d.read(Path(cfg['v103'])/'HELDOUT_RESULTS.json') if r['dataset'] in features];assert len(controls)==640
    folded=d.aggregate(rows,['representation','protocol','fold','seed','arm','mode'],['dense','gain','old_change','regret']);comparisons=[]
    for plan in plans:
        control={r['method']:r for r in d.read(Path(cfg['v103'])/'HELDOUT_FOLDS.json') if r['dataset']==plan['representation'] and r['protocol']==plan['protocol'] and r['fold']==plan['fold']}
        for mode in ('EXACT_EXPECTATION','ARGMAX'):
            block={r['arm']:r for r in folded if all(r[k]==plan[k] for k in ['representation','protocol','fold','seed']) and r['mode']==mode}
            r={k:plan[k] for k in ['representation','protocol','fold','seed']};r['mode']=mode
            for metric in ('dense','gain','old_change'):
                for arm in ('CE','WARM'):r[f'RL_minus_{arm}_{metric}']=block['RL'][metric]-block[arm][metric]
                for name,c in control.items():r[f'RL_minus_{name}_{metric}']=block['RL'][metric]-c[metric]
            comparisons.append(r)
    consistency=[]
    for representation in features:
        primary=[r for r in comparisons if r['representation']==representation and r['protocol']=='STREAM_SWAP' and r['mode']=='EXACT_EXPECTATION']
        consistency.append(dict(representation=representation,RL_minus_CE_positive_both_seeds_both_directions=all(r['RL_minus_CE_dense']>0 for r in primary),deltas=[{k:r[k] for k in ('fold','seed','RL_minus_CE_dense')} for r in primary]))
    for name,values in [('HELDOUT_RESULTS',rows),('HELDOUT_FOLDS',folded),('COMPARISONS',comparisons),('CONTROLS_REUSED',controls)]:d.write(root/(name+'.json'),values);d.table(root/(name+'.csv'),values)
    d.write(root/'DIAGNOSTIC.json',dict(status='FOLD_LOCAL_POLICY_DIAGNOSTIC_COMPLETE',direction_consistency=consistency,summary=d.aggregate(rows,['representation','protocol','seed','arm','mode'],['dense','gain','old_change','regret']),independent_source_validation='NA_SHARED_SOURCE_QUERY_ROLES',sequential_deployment='NOT_EVALUATED'))
    d.write(root/'COSTS.json',dict(native_updates=0,actor_optimizer_updates=sum(budget.count.values()),physical=dict(budget.count),new_ridge_solves=0,image_inference=0,training_query_image_evaluations=0,development_query_image_evaluations=0,new_annotation_cases=0,models=120,heldout_rows=1536,reused_control_rows=640))
    d.write(root/'FINAL.json',dict(status='COMPLETE',decision='FOLD_LOCAL_POLICY_DIAGNOSTIC_COMPLETE',time=time.time(),physical=dict(budget.count)))


if __name__=='__main__':
    cfg=json.loads(Path(os.environ['EXEC_CONFIG']).read_text());root=Path(os.environ['EXEC_RUN']);d=load(cfg['diagnostic_entry'],'diagnostic_io');lock=(root/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    d.write(root/'STARTED.json',dict(pid=os.getpid(),time=time.time(),commit=cfg['commit']))
    try:main(root,cfg)
    except BaseException as exc:d.write(root/'FAILURE.json',dict(error=repr(exc),time=time.time()));raise

"""Training-only cross-stream transfer, leakage-free simple predictors and objective gaps."""
import csv
import fcntl
import json
import math
import os
import statistics as st
import time
from pathlib import Path
import numpy as np

SOLVES = {"synthetic":0,"diagnostic":0}
PHASE = "synthetic"
SOLVE_LEDGER = None


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def table(path,rows):
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def ranks(x):
    x=np.asarray(x);return np.array([1+np.sum(x<v)+.5*(np.sum(x==v)-1) for v in x])


def corr(x,y):
    a=ranks(x);b=ranks(y);a-=a.mean();b-=b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den>0 else None


def predict(x,y,test):
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-6]=1;z=(x-mean)/scale;t=(test-mean)/scale
    centered=y-y.mean(1,keepdims=True);intercept=centered.mean(0)
    SOLVES[PHASE]+=1
    if SOLVE_LEDGER:
        with SOLVE_LEDGER.open('a') as log:log.write(json.dumps(dict(event='attempt',phase=PHASE,ordinal=SOLVES[PHASE]))+'\n')
    weight=np.linalg.solve(z.T@z/len(z)+np.eye(x.shape[1]),z.T@(centered-intercept)/len(z))
    if SOLVE_LEDGER:
        with SOLVE_LEDGER.open('a') as log:log.write(json.dumps(dict(event='success',phase=PHASE,ordinal=SOLVES[PHASE]))+'\n')
    ridge=intercept+t@weight
    nearest=np.argmin(((t[:,None]-z[None])**2).sum(-1),axis=1)
    return ridge,y[nearest]


def optimum(adv,logprior):
    import torch
    logits=(adv+.25*logprior)/.26;return logits-logits.logsumexp(-1,keepdim=True)


def objective(logp,adv,logprior):
    return (logp.exp()*(adv+.25*logprior-.26*logp)).sum(-1)


def selfcheck():
    import torch
    assert ranks([1,1,3]).tolist()==[1.5,1.5,3.] and abs(corr([1,2,3],[3,2,1])+1)<1e-12
    x=np.array([[0.,1.],[1.,1.],[2.,1.]]);y=np.array([[1.,0.],[0.,1.],[1.,0.]]);test=np.array([[.5,1.]])
    a=predict(x,y,test);heldout=np.zeros((1,2));heldout[:]=1e6;b=predict(x,y,test)
    assert all(np.array_equal(u,v) and np.isfinite(u).all() for u,v in zip(a,b))
    adv=torch.tensor([[1.,-1.,0.],[-1.,2.,0.]],dtype=torch.float64);prior=torch.tensor([[.2,.3,.5]],dtype=torch.float64).log();bar=adv.mean(0,keepdim=True);star=optimum(bar,prior)
    p=torch.tensor([[.4,.2,.4]],dtype=torch.float64).log();gap=objective(star,bar,prior)-objective(p,bar,prior);kl=.26*(p.exp()*(p-star)).sum(-1)
    assert torch.allclose(gap,kl,atol=1e-12) and (gap>=0).all()
    assert not torch.allclose(star.exp(),optimum(adv,prior).exp().mean(0,keepdim=True))


def dataset(cfg,name):
    rows=[]
    for i in range(8):
        for stream in (1,2):
            if name=='V101':part=read(Path(cfg['v101'])/f'jobs/collect{i}_{stream}/TRAINING_TABLE.private.json')
            else:part=read(Path(cfg['v92'])/f'jobs/collect{i}_{stream}/TRAINING_TABLE.private.json')+read(Path(cfg['v99'])/f'jobs/collect{i}_{stream}/TRAINING_TABLE.private.json')
            assert len(part)==24;rows+=part
    keys=sorted({(r['context'],r['stream'],r['step']) for r in rows});groups=[]
    for key in keys:
        g=sorted((r for r in rows if (r['context'],r['stream'],r['step'])==key),key=lambda r:r['action'])
        assert [r['action'] for r in g]==list(range(12)) and all(r['state']==g[0]['state'] for r in g)
        groups.append(g)
    x=np.array([g[0]['state'] for g in groups]);reward=np.array([[r['dense_reward'] for r in g] for g in groups]);gain=np.array([[r['gain'] for r in g] for g in groups]);old=np.array([[r['old_final']-r['old_entry'] for r in g] for g in groups])
    assert x.shape==(32,24) and reward.shape==gain.shape==old.shape==(32,12) and all(np.isfinite(v).all() for v in (x,reward,gain,old)) and np.allclose(reward,gain+old,atol=1e-14,rtol=0)
    return keys,x,reward,gain,old


def reliability(cfg,name,data):
    import torch
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior import core as c,stage_b as b
    keys,x,y,g,o=data;out=[];transfer=[];contexts=[v[3] for v in b.contexts()]
    for context in sorted({k[0] for k in keys}):
        i=contexts.index(context)
        for step in (100,200):
            ids=[keys.index((context,s,step)) for s in (1,2)];a,z=ids;diff1=y[a,:,None]-y[a,None,:];diff2=y[z,:,None]-y[z,None,:]
            valid=np.triu((abs(diff1)>1e-12)&(abs(diff2)>1e-12),1);agreements=(diff1*diff2>0)[valid]
            checkpoints=[]
            for stream in (1,2):
                base=cfg['v92'] if step==100 or name=='V99' else cfg['v101'];filename='ENTRY100_STREAM.private.pt' if step==100 else 'ON_POLICY_ENTRY200.private.pt'
                checkpoints.append(torch.load(Path(base)/f'jobs/collect{i}_{stream}'/filename,map_location='cpu',weights_only=False))
            same_student=c.e.same(checkpoints[0]['student'],checkpoints[1]['student']);same_prefix=c.e.same(checkpoints[0],checkpoints[1]);same_state=np.array_equal(x[a],x[z])
            out.append(dict(dataset=name,context=context,step=step,state_exact=bool(same_state),student_weights_exact=bool(same_student),full_snapshot_exact=bool(same_prefix),spearman=corr(y[a],y[z]),pairwise_agreement=float(agreements.mean()) if len(agreements) else None,comparable_action_pairs=int(valid.sum()),top1_equal=bool(y[a].argmax()==y[z].argmax()),top_gap_stream1=float(np.sort(y[a])[-1]-np.sort(y[a])[-2]),top_gap_stream2=float(np.sort(y[z])[-1]-np.sort(y[z])[-2])))
            for source,target in ((a,z),(z,a)):
                stream=keys[source][1];train=[j for j,k in enumerate(keys) if k[1]==stream];timetrain=[j for j in train if keys[j][2]==step]
                chosen=int(y[source].argmax());global_action=int(y[train].mean(0).argmax());time_action=int(y[timetrain].mean(0).argmax())
                r=dict(dataset=name,context=context,step=step,source_stream=stream,target_stream=keys[target][1],state_exact=bool(same_state),action=chosen,global_action=global_action,time_action=time_action)
                for metric,values in [('dense',y),('gain',g),('old_change',o)]:
                    r[metric]=float(values[target,chosen]);r[metric+'_minus_global']=float(values[target,chosen]-values[target,global_action]);r[metric+'_minus_time']=float(values[target,chosen]-values[target,time_action])
                r['source_advantage_over_uniform']=float(y[source,chosen]-y[source].mean());r['target_advantage_over_uniform']=float(y[target,chosen]-y[target].mean());transfer.append(r)
    return out,transfer


def crossvalidate(name,data):
    keys,x,y,g,o=data;rows=[];folds=[('STREAM_SWAP',f'stream{s}',[i for i,k in enumerate(keys) if k[1]==s]) for s in (1,2)]+[('CONDITION_LOCO',context,[i for i,k in enumerate(keys) if k[0]==context]) for context in sorted({k[0] for k in keys})]
    for protocol,fold,test in folds:
        train=[i for i in range(len(keys)) if i not in test];assert not set(train)&set(test)
        ridge,nearest=predict(x[train],y[train],x[test]);global_action=int(y[train].mean(0).argmax())
        for offset,i in enumerate(test):
            timeids=[j for j in train if keys[j][2]==keys[i][2]];actions={'UNIFORM_EXPECTATION':None,'TRAIN_GLOBAL':global_action,'TRAIN_TIME':int(y[timeids].mean(0).argmax()),'STATE_RIDGE':int(ridge[offset].argmax()),'STATE_1NN':int(nearest[offset].argmax())}
            for method,action in actions.items():
                r=dict(dataset=name,protocol=protocol,fold=fold,context=keys[i][0],stream=keys[i][1],step=keys[i][2],method=method,action=-1 if action is None else action,train_groups=len(train),test_groups=len(test))
                for metric,values in [('dense',y),('gain',g),('old_change',o)]:r[metric]=float(values[i].mean() if action is None else values[i,action])
                r['regret']=float(y[i].max()-r['dense']);rows.append(r)
    assert len(rows)==320
    return rows


def analytical(cfg,data):
    import torch,importlib.util
    torch.set_num_threads(2);spec=importlib.util.spec_from_file_location('v99',cfg['v99_entry']);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
    keys,x,rewards,_,_=data;x=torch.tensor(x,dtype=torch.float32);returns=torch.tensor(rewards,dtype=torch.float32);scale=.0026099849492311478
    adv=((returns-returns[:,:9].mean(-1,keepdim=True))/scale).clamp(-3,3);statekeys=[tuple(z.tolist()) for z in x];unique=sorted(set(statekeys));groups=[[i for i,k in enumerate(statekeys) if k==u] for u in unique];out=[];summary=[]
    for seed in (601,602):
        initial,mean,std=v.load12(Path(cfg['v99'])/f'jobs/fit/EXPANDED_CE_{seed}.private.pt');z=(x-mean)/std
        with torch.no_grad():prior=initial(z).log_softmax(-1).double();prior-=prior.logsumexp(-1,keepdim=True);logs={'INITIAL':prior}
        for arm in ('REFRESH_CE','REFRESH_RL'):
            actor,am,astd=v.load12(Path(cfg['v101'])/f'jobs/fit/{arm}_{seed}.private.pt');assert torch.equal(am,mean) and torch.equal(astd,std)
            with torch.no_grad():logs[arm]=actor(z).log_softmax(-1).double();logs[arm]-=logs[arm].logsumexp(-1,keepdim=True)
        for group_id,ids in enumerate(groups):
            a=adv[ids].double().mean(0,keepdim=True);p0=prior[ids[:1]];assert torch.allclose(prior[ids],p0,atol=0,rtol=0)
            star=optimum(a,p0);best=float(objective(star,a,p0))
            for arm,log in logs.items():
                lp=log[ids[:1]];assert torch.allclose(log[ids],lp,atol=0,rtol=0)
                value=float(objective(lp,a,p0));gap=best-value;kl=float(.26*(lp.exp()*(lp-star)).sum())
                assert abs(gap-kl)<2e-7 and gap>=-2e-7
                out.append(dict(seed=seed,state_group=group_id,rows=len(ids),arm=arm,objective=value,unconstrained_objective=best,gap=gap,scaled_KL_to_optimum=kl,raw_expected_dense=float((lp.exp()*returns[ids].double().mean(0)).sum()),entropy=float(-(lp.exp()*lp).sum())))
        for arm in logs:
            selected=[r for r in out if r['seed']==seed and r['arm']==arm];summary.append(dict(seed=seed,arm=arm,**{k:sum(r[k]*r['rows'] for r in selected)/32 for k in ('objective','unconstrained_objective','gap','raw_expected_dense','entropy')}))
    return out,summary,len(groups)


def aggregate(rows,groupkeys,metrics):
    keys=sorted({tuple(r[k] for k in groupkeys) for r in rows});result=[]
    for key in keys:
        values=[r for r in rows if tuple(r[k] for k in groupkeys)==key];result.append(dict(zip(groupkeys,key),n=len(values),**{metric:st.mean(r[metric] for r in values if r[metric] is not None) if any(r[metric] is not None for r in values) else None for metric in metrics}))
    return result


def main(root,cfg):
    global PHASE,SOLVE_LEDGER
    SOLVE_LEDGER=root/'LINEAR_SOLVE_LEDGER.jsonl'
    selfcheck();write(root/'SYNTHETIC_QUALIFICATION.json',dict(status='PASS',optimizer_updates=0,queries=0,synthetic_ridge_solves=SOLVES['synthetic']));PHASE='diagnostic'
    allrank=[];alltransfer=[];allcv=[];integrity=[];primary=None
    for name in ('V101','V99'):
        data=dataset(cfg,name);rank,transfer=reliability(cfg,name,data);cv=crossvalidate(name,data)
        allrank+=rank;alltransfer+=transfer;allcv+=cv;integrity.append(dict(dataset=name,groups=32,actions=384,unique_states=len({tuple(r) for r in data[1]}),exact_crossstream_state_pairs=sum(r['state_exact'] for r in rank),same_student_weight_pairs=sum(r['student_weights_exact'] for r in rank),same_full_snapshot_pairs=sum(r['full_snapshot_exact'] for r in rank)))
        if name=='V101':primary=data
    analytic,an_summary,unique=analytical(cfg,primary)
    rank_summary=aggregate(allrank,['dataset','step'],['spearman','pairwise_agreement','top1_equal','top_gap_stream1','top_gap_stream2'])
    transfer_summary=aggregate(alltransfer,['dataset','source_stream'],['dense_minus_global','dense_minus_time','gain_minus_global','old_change_minus_global','target_advantage_over_uniform'])
    cv_summary=aggregate(allcv,['dataset','protocol','method'],['dense','gain','old_change','regret']);fold_summary=aggregate(allcv,['dataset','protocol','fold','method'],['dense','gain','old_change','regret'])
    consistency=[]
    for dataset_name in ('V101','V99'):
        for method in ('STATE_RIDGE','STATE_1NN'):
            delta=[]
            for fold in ('stream1','stream2'):
                block={r['method']:r for r in fold_summary if r['dataset']==dataset_name and r['protocol']=='STREAM_SWAP' and r['fold']==fold};delta.append(dict(fold=fold,vs_global=block[method]['dense']-block['TRAIN_GLOBAL']['dense'],vs_time=block[method]['dense']-block['TRAIN_TIME']['dense']))
            consistency.append(dict(dataset=dataset_name,method=method,positive_both_directions_vs_both_controls=all(r[k]>0 for r in delta for k in ('vs_global','vs_time')),deltas=delta))
    result=dict(status='TRAINING_DIAGNOSTIC_COMPLETE_NO_DEPLOYMENT_CLAIM',integrity=integrity,rank_summary=rank_summary,transfer_summary=transfer_summary,cv_summary=cv_summary,state_prediction_consistency=consistency,analytic_summary=an_summary,unique_V101_state_groups=unique,independent_source_validation='NA_SHARED_SOURCE_QUERY_ROLES',same_state_noise_identification='Only exact state/full-prefix pairs qualify; crossstream comparisons otherwise describe transfer.',neural_refits=0)
    for name,rows in [('RANK_PAIRS',allrank),('CROSS_STREAM_TRANSFER',alltransfer),('HELDOUT_RESULTS',allcv),('HELDOUT_FOLDS',fold_summary),('ANALYTICAL_GAPS',analytic)]:write(root/(name+'.json'),rows);table(root/(name+'.csv'),rows)
    assert SOLVES=={'synthetic':2,'diagnostic':20}
    write(root/'DIAGNOSTIC.json',result);write(root/'COSTS.json',dict(synthetic_ridge_solves=2,native_updates=0,actor_optimizer_updates=0,training_image_evaluations=0,development_image_evaluations=0,new_unique_images=0,ridge_solves=20,nearest_neighbor_reference_sets=20,frozen_actor_evaluations=6,failed_updates=0))
    write(root/'FINAL.json',dict(status='COMPLETE',decision=result['status'],physical={},time=time.time()))


if __name__=='__main__':
    cfg=read(os.environ['EXEC_CONFIG']);root=Path(os.environ['EXEC_RUN']);lock=(root/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(root/'STARTED.json',dict(pid=os.getpid(),time=time.time(),commit=cfg['commit']))
    try:main(root,cfg)
    except BaseException as exc:write(root/'FAILURE.json',dict(error=repr(exc),time=time.time()));raise

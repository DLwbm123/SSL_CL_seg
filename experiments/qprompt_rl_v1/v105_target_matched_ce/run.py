"""Forward-KL control for the frozen fold-local reverse-KL objective."""
import copy
import fcntl
import json
import os
import runpy
import time
from pathlib import Path
import numpy as np
import torch

CAPS = dict(qualification=1, distillation=20480)


def target(z, adv, prior, d):
    unique, inverse = torch.unique(z, dim=0, return_inverse=True)
    result = torch.empty_like(adv, dtype=torch.float64)
    prior = prior.double().log_softmax(-1)
    for i in range(len(unique)):
        ids = (inverse == i).nonzero().flatten()
        assert torch.equal(prior[ids], prior[ids[:1]].expand(len(ids), -1))
        result[ids] = d.optimum(adv[ids].double().mean(0, keepdim=True), prior[ids[:1]])
    return result


def loss(actor, z, logtarget):
    return -(logtarget.exp().float() * actor(z).log_softmax(-1)).sum(-1).mean()


def update(actor, opt, z, logtarget, budget, category, key):
    value = loss(actor, z, logtarget)
    assert torch.isfinite(value)
    opt.zero_grad(set_to_none=True)
    value.backward()
    assert torch.isfinite(torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.))
    budget.step(category, key, opt)
    return float(value.detach())


def selfcheck(h, v, d, budget, actions=12):
    z = torch.zeros((3, 24)); z[2] = 1
    adv = torch.linspace(-2, 2, 3*actions).reshape(3, actions)
    actor = h.fresh(v, 601); prior = actor(z).detach().log_softmax(-1)
    star = target(z, adv, prior, d)
    assert torch.equal(star[0], star[1])
    assert torch.allclose(star[:1], d.optimum(adv[:2].double().mean(0, keepdim=True), prior[:1].double().log_softmax(-1)))
    perm = torch.tensor([2, 0, 1])
    assert torch.allclose(star[perm], target(z[perm], adv[perm], prior[perm], d))
    assert torch.allclose(star, target(z, adv+1., prior, d), atol=1e-6)
    lp = prior.double().log_softmax(-1); avg = adv.double().clone(); avg[:2] = avg[:2].mean(0)
    assert torch.allclose(d.objective(star, avg, lp)-d.objective(lp, avg, lp), .26*(lp.exp()*(lp-star)).sum(-1), atol=1e-12)
    ce = -(star.exp()*lp).sum(-1); entropy = -(star.exp()*star).sum(-1)
    assert torch.allclose(ce-entropy, (star.exp()*(star-lp)).sum(-1), atol=1e-12)
    opt = torch.optim.Adam(actor.parameters(), lr=.001)
    before = update(actor, opt, z, star, budget, 'qualification', 'synthetic')
    assert float(loss(actor, z, star).detach()) < before


def main(root, cfg):
    torch.set_num_threads(1)
    h = runpy.run_path(cfg['v104_entry'])
    from types import SimpleNamespace
    h = SimpleNamespace(**h)
    v = h.load(cfg['v99_entry'], 'actor12'); d = h.load(cfg['diagnostic_entry'], 'diagnostic')
    budget = runpy.run_path(cfg['budget_helper'])['Budget'](root, CAPS)
    selfcheck(h, v, d, budget)
    d.write(root/'SYNTHETIC_QUALIFICATION.json', dict(status='PASS', optimizer_updates=1, grouping_permutation_shift=True, forward_reverse_KL_identities=True, finite_decreasing_loss=True))
    keys, original, reward, gain, old = d.dataset(cfg, 'V101')
    saved = np.load(Path(cfg['v103'])/'FEATURES.private.npz')
    assert np.array_equal(saved['original'], original)
    features = {'ORIGINAL': original, 'FIXED_MEAN4': saved['probes'].mean(1)}
    historical = d.read(Path(cfg['v104'])/'HELDOUT_RESULTS.json')
    folds = [('STREAM_SWAP', f'stream{s}', [i for i,k in enumerate(keys) if k[1] == s]) for s in (1,2)]
    folds += [('CONDITION_LOCO', ctx, [i for i,k in enumerate(keys) if k[0] == ctx]) for ctx in sorted({k[0] for k in keys})]
    plans = []; replayed = 0
    returns = torch.tensor(reward, dtype=torch.float32)
    for rep, values in features.items():
        x = torch.tensor(values, dtype=torch.float32)
        for protocol, fold, test in folds:
            train = [i for i in range(32) if i not in test]
            for seed in (601,602):
                key = f'{rep}/{protocol}/{fold}/{seed}'; stem = key.replace('/', '_')
                mean, std, adv, scale = h.prepare(x[train], returns[train])
                for arm in ('WARM','CE','RL'):
                    cp = torch.load(Path(cfg['v104'])/(stem+'_'+arm+'.private.pt'), map_location='cpu', weights_only=False)
                    assert all(torch.equal(cp[n], value) for n,value in [('mean',mean),('std',std),('scale',scale)])
                    actor = h.fresh(v,seed); actor.load_state_dict(cp['actor'])
                    with torch.no_grad(): prob = actor((x[test]-mean)/std).softmax(-1).double().numpy(); prob /= prob.sum(-1,keepdims=True)
                    for offset,i in enumerate(test):
                        rows = [r for r in historical if (r['representation'],r['protocol'],r['fold'],r['seed'],r['arm'],r['context'],r['stream'],r['step']) == (rep,protocol,fold,seed,arm,*keys[i])]
                        assert len(rows) == 2
                        for r in rows:
                            assert np.array_equal(prob[offset], [r[f'p{a}'] for a in range(12)]) and int(prob[offset].argmax()) == r['argmax_action']
                            replayed += 1
                plans.append(dict(representation=rep,protocol=protocol,fold=fold,seed=seed,key=key,stem=stem,train=train,test=test))
    assert replayed == 1536 and len(plans) == 40
    d.write(root/'REPLAY_QUALIFICATION.json', dict(status='PASS', models=120, probability_rows=replayed, normalizers_and_scales_exact=True, optimizer_updates=0))
    logs = []; fits = []
    for plan in plans:
        x = torch.tensor(features[plan['representation']], dtype=torch.float32); ids = plan['train']
        cp = torch.load(Path(cfg['v104'])/(plan['stem']+'_WARM.private.pt'), map_location='cpu', weights_only=False)
        mean,std,adv,scale = h.prepare(x[ids],returns[ids]); z = (x[ids]-mean)/std
        actor = h.fresh(v,plan['seed']); actor.load_state_dict(cp['actor'])
        with torch.no_grad(): prior = actor(z).log_softmax(-1); star = target(z,adv,prior,d)
        opt = torch.optim.Adam(actor.parameters(), lr=.001)
        for step in range(512):
            value = update(actor,opt,z,star,budget,'distillation',plan['key']+f'/{step}')
            if step in (0,127,511): logs.append(dict(key=plan['key'],step=step+1,loss_before_update=value))
        with torch.no_grad():
            lp = actor(z).log_softmax(-1).double().log_softmax(-1)
            fits.append(dict(key=plan['key'],forward_KL=float((star.exp()*(star-lp)).sum(-1).mean()),reverse_KL=float((lp.exp()*(lp-star)).sum(-1).mean()),RL_objective_gap=float(.26*(lp.exp()*(lp-star)).sum(-1).mean()),expected_dense=float((lp.exp()*returns[ids].double()).sum(-1).mean())))
        with (root/(plan['stem']+'_DISTILL.private.pt')).open('xb') as f:
            torch.save(dict(actor=actor.state_dict(),mean=mean,std=std,scale=scale,seed=plan['seed']),f)
        (root/'STATUS.json').write_text(json.dumps(dict(status='RUNNING',phase='fit',completed_models=len(fits),total_models=40,physical=dict(budget.count))))
    assert dict(budget.count) == CAPS
    d.write(root/'FIT_LOCK.json', dict(status='SEALED_BEFORE_HELDOUT_SCORING',models=40,time=time.time()))
    rows = []
    for plan in plans:
        cp = torch.load(root/(plan['stem']+'_DISTILL.private.pt'),map_location='cpu',weights_only=False)
        actor = h.fresh(v,plan['seed']);actor.load_state_dict(cp['actor']);test=plan['test']
        x = torch.tensor(features[plan['representation']],dtype=torch.float32)
        with torch.no_grad(): prob = actor((x[test]-cp['mean'])/cp['std']).softmax(-1).double().numpy();prob/=prob.sum(-1,keepdims=True)
        for offset,i in enumerate(test):
            p=prob[offset];action=int(p.argmax())
            for mode in ('EXACT_EXPECTATION','ARGMAX'):
                weight=p if mode=='EXACT_EXPECTATION' else np.eye(12)[action]
                r={k:plan[k] for k in ('representation','protocol','fold','seed')}
                r.update(arm='DISTILL',mode=mode,context=keys[i][0],stream=keys[i][1],step=keys[i][2],argmax_action=action,train_groups=len(plan['train']),test_groups=len(test))
                for metric,values in [('dense',reward),('gain',gain),('old_change',old)]:r[metric]=float(weight@values[i])
                r['regret']=float(reward[i].max()-r['dense']);r.update({f'p{a}':float(p[a]) for a in range(12)});rows.append(r)
    assert len(rows)==512
    folded=d.aggregate(historical+rows,['representation','protocol','fold','seed','arm','mode'],['dense','gain','old_change','regret']);comparisons=[]
    controls=d.read(Path(cfg['v103'])/'HELDOUT_FOLDS.json')
    for plan in plans:
        for mode in ('EXACT_EXPECTATION','ARGMAX'):
            block={r['arm']:r for r in folded if all(r[k]==plan[k] for k in ('representation','protocol','fold','seed')) and r['mode']==mode}
            r={k:plan[k] for k in ('representation','protocol','fold','seed')};r['mode']=mode
            for metric in ('dense','gain','old_change'):
                for a,b in [('RL','DISTILL'),('DISTILL','CE'),('DISTILL','WARM')]:r[f'{a}_minus_{b}_{metric}']=block[a][metric]-block[b][metric]
                for c in controls:
                    if c['dataset']==plan['representation'] and c['protocol']==plan['protocol'] and c['fold']==plan['fold']:r[f'DISTILL_minus_{c["method"]}_{metric}']=block['DISTILL'][metric]-c[metric]
            comparisons.append(r)
    for name,values in [('HELDOUT_RESULTS',rows),('HELDOUT_FOLDS',folded),('COMPARISONS',comparisons),('FIT_SUMMARY',fits),('FIT_LOG',logs)]:d.write(root/(name+'.json'),values);d.table(root/(name+'.csv'),values)
    primary=[r for r in comparisons if r['representation']=='FIXED_MEAN4' and r['protocol']=='STREAM_SWAP' and r['mode']=='EXACT_EXPECTATION']
    d.write(root/'DIAGNOSTIC.json',dict(status='TARGET_MATCHED_DIAGNOSTIC_COMPLETE',RL_increment_positive_all4=all(r['RL_minus_DISTILL_dense']>0 for r in primary),primary=primary,sequential_deployment='NOT_EVALUATED'))
    d.write(root/'COSTS.json',dict(native_updates=0,actor_optimizer_updates=sum(budget.count.values()),physical=dict(budget.count),image_inference=0,queries=0,new_annotations=0,new_models=40,reused_models=120,new_heldout_rows=512))
    d.write(root/'FINAL.json',dict(status='COMPLETE',decision='TARGET_MATCHED_DIAGNOSTIC_COMPLETE',time=time.time(),physical=dict(budget.count)))


if __name__=='__main__':
    cfg=json.loads(Path(os.environ['EXEC_CONFIG']).read_text());root=Path(os.environ['EXEC_RUN'])
    lock=(root/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (root/'STARTED.json').write_text(json.dumps(dict(pid=os.getpid(),time=time.time(),commit=cfg['commit'])))
    try:main(root,cfg)
    except BaseException as exc:(root/'FAILURE.json').write_text(json.dumps(dict(error=repr(exc),time=time.time())));raise

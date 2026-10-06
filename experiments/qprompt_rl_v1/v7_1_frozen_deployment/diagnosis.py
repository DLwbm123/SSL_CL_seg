"""Read-only diagnostics of states that actually survived V7."""
import copy, itertools, math, io as bytesio
import numpy as np
import torch
from .worker import v,w,e,q,p,ROOT,C,load,policy,oldjob,Uniform,sample

def stats(counts):
    a=np.asarray(counts,dtype=float);s=float(a.sum())
    return dict(exposure_sorted=sorted(map(int,a)),total=int(s),covered=int((a>0).sum()),coverage=float((a>0).mean()),unused_fraction=float((a==0).mean()),minimum=int(a.min()),median=float(np.median(a)),maximum=int(a.max()),cv=float(a.std()/a.mean()) if s else None,N_eff=s*s/float(a@a) if s else 0.)

def exposure(path,domain):
    n=p.old.UNLABELED[domain];steps=p.old.DOMAINS[domain];spe=32 if domain=='RIM_ONE_r3' else 21
    import json
    rows=[json.loads(line) for line in (path/'SELECTIONS.private.jsonl').read_text().splitlines()]
    final=load(path/'student_latest.private.pt')['state']['exposure']
    total=[0]*n;windows=[];previous=set();run=[0]*n;longest=[0]*n
    for row in rows:
        chosen=sorted(row['indices']);current=set(chosen);counts=[0]*n
        if chosen:
            for step in range(row['block']*100,(row['block']+1)*100):
                epoch,index=divmod(step,spe);cycle,offset=divmod(2*index,len(chosen))
                order=torch.randperm(len(chosen),generator=v.generator(168,1 if domain=='RIM_ONE_r3' else 2,1,epoch,f'unlabeled/order/{cycle}')).tolist()
                for i in (order[offset],order[(offset+1)%len(chosen)]):counts[chosen[i]]+=1
        for i in range(n):
            total[i]+=counts[i];run[i]=run[i]+1 if counts[i] else 0;longest[i]=max(longest[i],run[i])
        windows.append(dict(block=row['block'],jaccard_previous=len(current&previous)/len(current|previous) if current|previous else None,**stats(counts)));previous=current
    assert total==final,'index replay does not match saved actual exposures'
    assert len(rows)==steps//100
    return dict(global_actual=stats(final),windows_reconstructed=windows,reconstructed_matches_saved_actual=True,max_consecutive_windows_sorted=sorted(longest),image_identity_mapping_published=False)

def distribution(model,x,emb,k,seed):
    chosen=[];positions=[]
    with torch.no_grad():
        for j in range(k):
            logits=model(x,emb,chosen,k);finite=logits[torch.isfinite(logits)];dist=torch.distributions.Categorical(logits=logits)
            top=finite.topk(2).values;h=float(dist.entropy())
            assert len(finite)==len(x)-j and all(torch.isneginf(logits[i]) for i in chosen)
            positions.append(dict(position=j,legal=len(finite),entropy=h,uniform_entropy=math.log(len(finite)),KL_to_uniform=math.log(len(finite))-h,logit_std=float(finite.std(unbiased=False)),max_probability=float(dist.probs.max()),greedy_top2_gap=float(top[0]-top[1])))
            chosen.append(int(logits.argmax()))
        samples=[set(q.sequence(model,x,emb,k,generator=torch.Generator().manual_seed(e.stable('V7.1/audit',C['domain'],seed,r)))[0]) for r in range(8)]
    jac=[len(a&b)/len(a|b) for a,b in itertools.combinations(samples,2)]
    return dict(greedy_conditioned_positions=positions,sampling_pairwise_jaccard=jac,sampling_jaccard_mean=float(np.mean(jac))),set(chosen)

def gradient(group,k):
    prior=e.rng_state();model=q.Policy();model.load_state_dict(group['policy']);e.rng_restore(prior)
    rewards=np.asarray(group['rewards']);adv=(rewards-rewards.mean())/max(float(rewards.std()),1e-6)
    terms=[];ent=[]
    for i,(actions,old) in enumerate(group['samples']):
        _,lp,h=q.sequence(model,group['x'],group['embeddings'],k,actions=actions)
        ratio=(lp-old).exp();terms.append(-torch.minimum(ratio*float(adv[i]),ratio.clamp(.8,1.2)*float(adv[i])).mean());ent.append(-.01*h.mean())
    params=list(model.parameters())
    a=torch.cat([t.flatten() for t in torch.autograd.grad(sum(terms)/4,params,retain_graph=True)])
    b=torch.cat([t.flatten() for t in torch.autograd.grad(sum(ent)/4,params)])
    w.COST['policy_diagnostic_backward_calls']+=2
    na,nb=float(a.norm()),float(b.norm());cos=float(torch.dot(a,b)/(a.norm()*b.norm())) if na*nb else None
    return dict(state='saved pre-update behavior policy',group_block=group['block'],policy_grad_norm=na,weighted_entropy_grad_norm=nb,entropy_coefficient=.01,angle_degrees=math.degrees(math.acos(max(-1,min(1,cos)))) if cos is not None else None,optimizer_updates=0)

def reward(t,anchor):
    saved=v.snapshot(t);quality=[];nlls=[];dice_losses=[];kl=[];valid=[]
    try:
        with v.preserve(t),torch.no_grad():
            for i in range(len(t.provider.feedback['online'])):
                b=t.provider.feedback['online'][i];w.COST['reward_label_sample_accesses']+=1
                pred=t.clean(b['image'][None].to(t.provider.device));y=b['label'][None].to(t.provider.device);mask=y[0]!=255
                supported=torch.unique(y[0][mask]);nll=torch.stack([-pred[0,int(c)][y[0]==c].clamp_min(1e-8).log().mean() for c in supported]).mean()
                dl=[]
                for c in (1,2):
                    pc=pred[0,c][mask];yc=(y[0][mask]==c).float();dl.append(1-(2*(pc*yc).sum()+1)/(pc.sum()+yc.sum()+1))
                quality.append(float(e.quality(pred,y)));nlls.append(float(nll));dice_losses.append([float(a) for a in dl])
            for i,src in anchor:
                pred=t.clean(v.image(t,i));src=src.to(pred);mask=src.max(1).values>.7
                parts=(src*(src.clamp_min(1e-8).log()-pred.clamp_min(1e-8).log()))*mask[:,None]
                kl.append(parts.mean((0,2,3)).tolist());valid.append([float((mask&(src.argmax(1)==c)).float().mean()) for c in range(3)])
                w.COST['retention_forward_images']+=1
        out=dict(quality=float(np.mean(quality)),balanced_NLL=float(np.mean(nlls)),soft_Dice_loss_rim_cup=np.mean(dice_losses,axis=0).tolist(),source_KL_channels=np.mean(kl,axis=0).tolist(),source_admitted_pixel_fraction_by_argmax=np.mean(valid,axis=0).tolist())
        out['source_KL']=sum(out['source_KL_channels']);out['objective']=out['quality']-.1*out['source_KL'];return out
    finally:
        v.restore(t,saved);assert e.same(saved,v.snapshot(t))

def zero_checks(t,model,x,emb):
    before=v.snapshot(t);weights=e.cpu(model.state_dict());n=len(x);k=p.old.k(C['domain']);zero=copy.deepcopy(model)
    for par in zero.parameters():par.data.zero_()
    a=sample(zero,x,emb,C['domain'],601,0);b=sample(Uniform(),x,emb,C['domain'],601,0)
    assert a==b and len(set(a))==k and all(0<=i<n for i in a)
    for j in range(k):
        logits=zero(x,emb,a[:j],k);assert torch.isneginf(logits[a[:j]]).all() and (logits[torch.isfinite(logits)]==0).all()
    g=torch.Generator().manual_seed(991);buffer=bytesio.BytesIO();torch.save(g.get_state(),buffer);buffer.seek(0)
    g2=torch.Generator();g2.set_state(torch.load(buffer,weights_only=False))
    assert q.sequence(model,x,emb,k,generator=g)[0]==q.sequence(model,x,emb,k,generator=g2)[0]
    assert e.same(before,v.snapshot(t)) and e.same(weights,e.cpu(model.state_dict())) and all(par.grad is None for par in model.parameters())

def diagnosis(ledger):
    from .report import self_check
    checked=self_check();e.write(ROOT/'SERIALIZATION_CHECK.json',checked);assert e.read(ROOT/'SERIALIZATION_CHECK.json')==checked
    t=v.create(ledger);entry=v.snapshot(t);d=C['domain'];diagnostics=[];rewards=[]
    with v.preserve(t):
        assert e.same(t.provider.unlabeled(0),v.NativeCurrentDomain.unlabeled(t.provider,0))
        assert 'label' not in t.provider._u[0]
    assert e.same(entry,v.snapshot(t))
    for seed in p.old.SEEDS:
        model,path=policy(seed);weights=e.cpu(model.state_dict());group=load(oldjob(f'learn_{d}_{seed}')/'group_latest.private.pt')
        latest=load(oldjob(f'learn_{d}_{seed}')/'student_latest.private.pt');assert e.same(group['anchors'],latest['anchors'])
        states={'entry':load(oldjob(f'endpoint_{d}_GRPO_{seed}')/'ENTRY.private.pt')['state'],'endpoint':load(oldjob(f'endpoint_{d}_GRPO_{seed}')/'student_latest.private.pt')['state'],'group_entry':group['entry'],'group_branch0':group['branch0']}
        greedy={};rows=[]
        for name,state in states.items():
            v.restore(t,state);before=v.snapshot(t);x,emb=v.features(t)
            assert e.same(before,v.snapshot(t)),'feature state mutation'
            zero_checks(t,model,x,emb)
            dist,greedy[name]=distribution(model,x,emb,p.old.k(d),seed);rows.append(dict(state=name,step=t.step,**dist))
        stability=[dict(a=a,b=b,jaccard=len(greedy[a]&greedy[b])/len(greedy[a]|greedy[b])) for a,b in itertools.combinations(greedy,2)]
        diagnostics.append(dict(domain=d,controller=seed,states=rows,greedy_state_stability=stability,gradient=gradient(group,p.old.k(d))))
        parts={}
        for name in ('entry','branch0'):
            v.restore(t,group[name]);parts[name]=reward(t,group['anchors'])
        delta=parts['branch0']['objective']-parts['entry']['objective']
        assert abs(delta-group['rewards'][0])<1e-6,'reward reconstruction differs'
        rewards.append(dict(domain=d,controller=seed,last_group_block=group['block'],saved_group_rewards=group['rewards'],branch0=parts,branch0_recomputed_delta=delta,full_history_decomposition='unavailable: other branch states and earlier groups not retained; no training replay',source_KL_semantics='target U source predictions; not old-domain retention guarantee'))
        assert e.same(weights,e.cpu(model.state_dict())) and all(par.grad is None for par in model.parameters())
    v.restore(t,entry);assert e.same(entry,v.snapshot(t)) and not ledger.count
    exposures=[]
    for path in sorted((Path(C['old_campaign'])/'scopes/pilot/jobs').glob('endpoint_'+d+'_*')):
        cfg=e.read(path/'CONFIG.private.json');exposures.append(dict(domain=d,method=cfg['method'],controller=cfg['controller'],**exposure(path,d)))
    e.write(ROOT/'POLICY_DISTRIBUTION_DIAGNOSTICS.json',diagnostics);e.write(ROOT/'EXPOSURE_DIAGNOSTICS.json',exposures);e.write(ROOT/'REWARD_COMPONENT_DIAGNOSTICS.json',rewards)
    e.write(ROOT/'ZERO_CHECKS.json',dict(status='PASS',both_policies_loaded=True,all_U_native_parity=True,U_labels_absent=True,zero_logits_exact_sampling=True,private_rng_isolated=True,sampling_rng_serialization_exact=True,feature_snapshot_exact=True,frozen_policy_unchanged=True,diagnostics_restored=True,student_updates=0,controller_updates=0))

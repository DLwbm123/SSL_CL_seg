"""Additive target-quality sufficient statistics, with held-out labels used only here."""
import torch


@torch.no_grad()
def measure(base,corrected,labels,mask,weight):
    assert base.shape==corrected.shape and base.shape[1]==3 and labels.shape==mask.shape==weight.shape==base[:,0].shape
    assert not ((labels[mask]<0)|(labels[mask]>=3)).any()
    b=base.argmax(1);a=corrected.argmax(1);bc=b==labels;ac=a==labels;diff=b!=a
    fixed=~bc&ac;broken=bc&~ac
    y=labels.clamp(0,2)[:,None]
    bn=-base.gather(1,y).squeeze(1).clamp_min(1e-8).log();an=-corrected.gather(1,y).squeeze(1).clamp_min(1e-8).log()
    weighted=lambda field:float((field.double()*weight.double())[mask].sum())
    result=dict(pixels=int(mask.sum()),changed=int((mask&diff).sum()),repaired=int((mask&fixed).sum()),harmed=int((mask&broken).sum()),both_wrong_changed=int((mask&diff&~bc&~ac).sum()),base_correct=int((mask&bc).sum()),corrected_correct=int((mask&ac).sum()),weight=weighted(torch.ones_like(weight)),weighted_repaired=weighted(fixed),weighted_harmed=weighted(broken),weighted_base_correct=weighted(bc),weighted_corrected_correct=weighted(ac),base_nll=float(bn[mask].double().sum()),corrected_nll=float(an[mask].double().sum()),weighted_base_nll=weighted(bn),weighted_corrected_nll=weighted(an))
    assert result['corrected_correct']-result['base_correct']==result['repaired']-result['harmed']
    assert result['changed']==result['repaired']+result['harmed']+result['both_wrong_changed']
    return result


def rates(row):
    n,w=row['pixels'],row['weight']
    return dict(net_accuracy=(row['repaired']-row['harmed'])/n if n else None,weighted_net_accuracy=(row['weighted_repaired']-row['weighted_harmed'])/w if w else None,nll_delta=(row['corrected_nll']-row['base_nll'])/n if n else None,weighted_nll_delta=(row['weighted_corrected_nll']-row['weighted_base_nll'])/w if w else None,change_fraction=row['changed']/n if n else None)




ARMS=('BASE','EMA_ALL','EMA_HALF','RANDOM_HALF','PROTO_HALF')


def top(mask,score,k):
    index=torch.nonzero(mask.flatten(),as_tuple=False).flatten()
    index=index[torch.argsort(score.flatten()[index],descending=True,stable=True)[:k]]
    chosen=torch.zeros_like(mask);chosen.flatten()[index]=True
    return chosen


@torch.no_grad()
def ema_masks(base,q,selected,seed):
    old=base.argmax(1);e=q.argmax(1);confidence=q.max(1).values-q.topk(2,dim=1).values[:,1]
    all_mask=selected&(((old==1)&(e==0))|((old==2)&(e==1)))
    ema=torch.zeros_like(selected);random=torch.zeros_like(selected);g=torch.Generator().manual_seed(seed);counts={}
    for a,b in ((1,0),(2,1)):
        group=all_mask&(old==a)&(e==b);n=int(group.sum());k=n//2
        ema|=top(group,confidence,k);index=torch.nonzero(group.flatten(),as_tuple=False).flatten()
        if n:random.flatten()[index[torch.randperm(n,generator=g)[:k]]]=True
        counts[f'conflict_{a}_{b}']=n;counts[f'quota_{a}_{b}']=k
    return all_mask,ema,random,counts


@torch.no_grad()
def controls(base,q,selected,prototype_scores,permute,seed):
    all_mask,ema,random,counts=ema_masks(base,q,selected,seed)
    old=base.argmax(1);e=q.argmax(1);proto=torch.zeros_like(selected)
    score=prototype_scores.gather(1,e[:,None]).squeeze(1)-prototype_scores.masked_fill(torch.nn.functional.one_hot(e,3).permute(0,3,1,2).bool(),-torch.inf).max(1).values
    assert torch.isfinite(score).all()
    for a,b in ((1,0),(2,1)):
        group=all_mask&(old==a)&(e==b);k=counts[f'quota_{a}_{b}'];proto|=top(group,score,k)
        assert all(int((m&group).sum())==k for m in (ema,random,proto))
    masks={'BASE':selected&False,'EMA_ALL':all_mask,'EMA_HALF':ema,'RANDOM_HALF':random,'PROTO_HALF':proto}
    weight=base.new_tensor([.5,1.,1.5])[e].double();outputs={}
    counts.update(selected=int(selected.sum()),conflict_total=int(all_mask.sum()),half_budget=int(ema.sum()),EMA_PROTO_overlap=int((ema&proto).sum()),RANDOM_PROTO_overlap=int((random&proto).sum()))
    for name,mask in masks.items():
        value=permute(base,e,mask);outputs[name]=value
        assert torch.equal(value.sort(1).values,base.sort(1).values)
        assert torch.equal(value.masked_select(~mask[:,None]),base.masked_select(~mask[:,None]))
        changed=selected&(value.argmax(1)!=old);assert torch.equal(changed,mask),'actual changes must equal selected intervention'
        counts[name+'_changed']=int(changed.sum());counts[name+'_changed_weight']=float(weight[changed].sum())
        counts[name+'_weighted_target_L1_change']=float(((value-base).abs().sum(1).double()*weight)[selected].sum())
    assert counts['EMA_HALF_changed_weight']==counts['RANDOM_HALF_changed_weight']==counts['PROTO_HALF_changed_weight']
    return outputs,counts


def analyze(rows):
    assert len(rows)==40 and {(r['context'],r['scope']) for r in rows}=={(i,k) for i in range(8) for k in ARMS}
    by={(r['context'],r['scope']):r for r in rows};comparisons=[]
    for method,control in (('EMA_ALL','BASE'),('EMA_HALF','BASE'),('EMA_HALF','RANDOM_HALF'),('PROTO_HALF','BASE'),('PROTO_HALF','EMA_HALF'),('PROTO_HALF','RANDOM_HALF')):
        values=[]
        for i in range(8):
            p=by[i,method];r=by[i,control];assert p['pixels']==r['pixels'] and p['weight']==r['weight'] and p['weight']>0
            values.append(dict(context=i,source_step=p['source_step'],labeled_images=p['labeled_images'],weighted_accuracy_delta=(p['weighted_corrected_correct']-r['weighted_corrected_correct'])/p['weight'],weighted_nll_delta=(p['weighted_corrected_nll']-r['weighted_corrected_nll'])/p['weight']))
        mean=lambda rr,k:sum(r[k] for r in rr)/len(rr)
        groups=[dict(group=f'{key}={v}',weighted_accuracy_delta=mean([r for r in values if r[key]==v],'weighted_accuracy_delta'),weighted_nll_delta=mean([r for r in values if r[key]==v],'weighted_nll_delta')) for key in ('labeled_images','source_step') for v in sorted({r[key] for r in values})]
        acc=mean(values,'weighted_accuracy_delta');nll=mean(values,'weighted_nll_delta');positive=sum(r['weighted_accuracy_delta']>0 and r['weighted_nll_delta']<0 for r in values)
        passed=acc>0 and nll<0 and positive>=6 and all(r['weighted_accuracy_delta']>0 and r['weighted_nll_delta']<0 for r in groups)
        comparisons.append(dict(method=method,control=control,equal_context_weighted_accuracy_delta=acc,equal_context_weighted_nll_delta=nll,joint_positive_contexts=positive,passed=passed,contexts=values,groups=groups))
    independent=comparisons[0]['passed'];prototype=all(r['passed'] for r in comparisons[3:])
    return dict(status='DEVELOPMENT_EMA_CONFLICT_SIGNAL' if independent else 'NO_RELIABLE_EMA_ALL_CONFLICT_SIGNAL',independent_EMA_ALL_rule_quality_signal=independent,EMA_HALF_quality_signal=comparisons[1]['passed'],EMA_ranking_increment=comparisons[2]['passed'],prototype_ranking_increment=prototype,comparisons=comparisons,independent_confirmation=False,model_level_holdout=False,training_benefit_tested=False,campaign_success=False)


def selfcheck(permute):
    old=torch.tensor([[[1,1,1,1,2,2,2,2]]]);e=old-1;selected=torch.ones_like(old,dtype=torch.bool)
    base=torch.nn.functional.one_hot(old,3).permute(0,3,1,2).float()*.7+.1
    q=torch.nn.functional.one_hot(e,3).permute(0,3,1,2).float()*.7+.1
    scores=q+torch.nn.functional.one_hot(e,3).permute(0,3,1,2)*torch.arange(8).reshape(1,1,1,8)/10;out,counts=controls(base,q,selected,scores,permute,122000)
    assert counts['conflict_total']==8 and counts['half_budget']==4
    assert not torch.equal(out['EMA_HALF'],out['PROTO_HALF'])
    assert all(counts[k+'_changed']==4 for k in ('EMA_HALF','RANDOM_HALF','PROTO_HALF'))
    changed,c2=controls(base,q,selected,-scores,permute,122000)
    assert all(torch.equal(out[k],changed[k]) for k in ('BASE','EMA_ALL','EMA_HALF','RANDOM_HALF'))
    all_mask,ema,random,_=ema_masks(base,q,selected,122000);assert torch.equal(ema,top(all_mask,q[:,0]*0,2)|top(all_mask&(old==2),q[:,0]*0,2))
    empty,_=controls(base,base,selected,scores,permute,122000);assert all(torch.equal(v,base) for v in empty.values())
    rows=[];weight=torch.ones_like(old).float()
    for i in range(8):
        for name,value in out.items():rows.append(dict(context=i,scope=name,source_step=2000 if i<4 else 8000,labeled_images=2 if i%2==0 else 8,**measure(base,value,e,selected,weight)))
    result=analyze(rows);assert result['independent_EMA_ALL_rule_quality_signal'] and not result['prototype_ranking_increment']
    return dict(status='PASS',model_forwards=0,image_reads=0,query_calls=0,optimizer_calls=0,checks=['EMA-only masks do not accept prototype inputs','prototype changes leave EMA/random invariant','exact independent half quota and weight','stable tie order','empty conflict identity','gate identity fails strict increment'])

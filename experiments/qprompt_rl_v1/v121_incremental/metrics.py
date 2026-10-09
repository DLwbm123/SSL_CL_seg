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




@torch.no_grad()
def controls(base,full,qclass,selected,permute,seed):
    old=base.argmax(1);proposal=full.argmax(1)
    candidate=selected&(((old==1)&(proposal==0))|((old==2)&(proposal==1)))
    proto=permute(base,proposal,candidate);ema=permute(base,qclass,candidate)
    destinations=proto.argmax(1);shuffled=destinations.clone();g=torch.Generator().manual_seed(seed)
    counts=dict(selected=int(selected.sum()),candidate=int(candidate.sum()),EMA_agrees_prototype=int((candidate&(qclass==proposal)).sum()),EMA_agrees_original=int((candidate&(qclass==old)).sum()),EMA_other_class=int((candidate&(qclass!=old)&(qclass!=proposal)).sum()),active_shuffle_strata=0,fully_occupied_strata=0,shuffle_pool=0)
    for a in range(3):
        for e in range(3):
            index=torch.nonzero((selected&(old==a)&(qclass==e)).flatten(),as_tuple=False).flatten()
            if not len(index):continue
            before=destinations.flatten()[index];k=int((before!=a).sum())
            if not k:continue
            counts['active_shuffle_strata']+=1;counts['fully_occupied_strata']+=int(k==len(index));counts['shuffle_pool']+=len(index)
            shuffled.flatten()[index]=before[torch.randperm(len(index),generator=g)]
            assert torch.equal(before.bincount(minlength=3),shuffled.flatten()[index].bincount(minlength=3))
    random=permute(base,shuffled,selected)
    assert torch.equal(random.argmax(1)[selected],shuffled[selected]),'shuffle actual-assignment mismatch'
    assert int((selected&(random.argmax(1)!=old)).sum())==counts['candidate']
    counts['shuffle_locations_different']=int((selected&(shuffled!=destinations)).sum())
    counts['retained_original_changed_locations']=int((candidate&(shuffled!=old)).sum())
    outputs={'BASE':base,'PROTO':proto,'EMA':ema,'SHUFFLE':random}
    for name,target in outputs.items():
        assert torch.equal(target.sort(1).values,base.sort(1).values)
        assert torch.equal(target.masked_select(~selected[:,None]),base.masked_select(~selected[:,None]))
        if name in ('PROTO','EMA'):assert torch.equal(target.masked_select(~candidate[:,None]),base.masked_select(~candidate[:,None]))
        delta=(target-base).abs().sum(1).double();weight=base.new_tensor([.5,1.,1.5])[qclass].double()
        counts[name+'_weighted_target_L1_change']=float((delta*weight)[selected].sum())
        counts[name+'_changed']=int((selected&(target.argmax(1)!=old)).sum())
    assert counts['EMA_agrees_prototype']+counts['EMA_agrees_original']+counts['EMA_other_class']==counts['candidate']
    return outputs,counts


def analyze(rows):
    assert len(rows)==32 and {(r['context'],r['scope']) for r in rows}=={(i,k) for i in range(8) for k in ('BASE','PROTO','EMA','SHUFFLE')}
    by={(r['context'],r['scope']):r for r in rows};comparisons=[]
    for control in ('BASE','EMA','SHUFFLE'):
        values=[]
        for i in range(8):
            p=by[i,'PROTO'];r=by[i,control];assert p['pixels']==r['pixels'] and p['weight']==r['weight'] and p['weight']>0
            values.append(dict(context=i,source_step=p['source_step'],labeled_images=p['labeled_images'],weighted_accuracy_delta=(p['weighted_corrected_correct']-r['weighted_corrected_correct'])/p['weight'],weighted_nll_delta=(p['weighted_corrected_nll']-r['weighted_corrected_nll'])/p['weight']))
        mean=lambda rr,k:sum(r[k] for r in rr)/len(rr)
        groups=[dict(group=f'{key}={v}',weighted_accuracy_delta=mean([r for r in values if r[key]==v],'weighted_accuracy_delta'),weighted_nll_delta=mean([r for r in values if r[key]==v],'weighted_nll_delta')) for key in ('labeled_images','source_step') for v in sorted({r[key] for r in values})]
        acc=mean(values,'weighted_accuracy_delta');nll=mean(values,'weighted_nll_delta');positive=sum(r['weighted_accuracy_delta']>0 and r['weighted_nll_delta']<0 for r in values)
        passed=acc>0 and nll<0 and positive>=6 and all(r['weighted_accuracy_delta']>0 and r['weighted_nll_delta']<0 for r in groups)
        comparisons.append(dict(control=control,equal_context_weighted_accuracy_delta=acc,equal_context_weighted_nll_delta=nll,joint_positive_contexts=positive,passed=passed,contexts=values,groups=groups))
    signal=all(r['passed'] for r in comparisons)
    return dict(status='INCREMENTAL_PROTOTYPE_QUALITY_SIGNAL' if signal else 'NO_RELIABLE_INCREMENTAL_PROTOTYPE_SIGNAL',comparisons=comparisons,train_only_signal=signal,independent_confirmation=False,training_benefit_tested=False,campaign_success=False)


def selfcheck(permute):
    old=torch.tensor([[[1,1,1,2,2,2]]]);qclass=torch.tensor([[[0,0,0,1,1,1]]]);base=torch.nn.functional.one_hot(old,3).permute(0,3,1,2).float()*.7+.1
    chosen=old.clone();chosen[0,0,0]=0;chosen[0,0,3]=1
    selected=torch.ones_like(old,dtype=torch.bool);full=permute(base,chosen,selected)
    out,c=controls(base,full,qclass,selected,permute,121000)
    assert c['candidate']==2 and c['PROTO_changed']==c['SHUFFLE_changed']==2 and c['EMA_agrees_prototype']==2
    assert torch.equal(out['PROTO'],out['EMA']) and torch.equal(out['BASE'],base)
    out2,c2=controls(base,full,qclass,selected,permute,121000);assert c2==c and torch.equal(out2['SHUFFLE'],out['SHUFFLE'])
    identity,_=controls(base,base,qclass,selected,permute,121000);assert all(torch.equal(v,base) for v in identity.values())
    labels=chosen;weight=torch.ones_like(labels).float();rows=[]
    for i in range(8):
        for k,v in out.items():rows.append(dict(context=i,scope=k,source_step=2000 if i<4 else 8000,labeled_images=2 if i%2==0 else 8,**measure(base,v,labels,selected,weight)))
    decision=analyze(rows);assert not decision['train_only_signal'] and not decision['comparisons'][1]['passed']
    for r in rows:
        if r['scope'] in ('EMA','SHUFFLE'):r.update(measure(base,base,labels,selected,weight))
    assert analyze(rows)['train_only_signal']
    return dict(status='PASS',model_forwards=0,image_reads=0,query_calls=0,optimizer_calls=0,checks=['fixed observable candidate union','same-region EMA','shuffle exact dose/destination/weight strata','fixed permutation repeatability','empty candidate identity','equal EMA cannot pass','strict joint comparison gate'])

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


def analyze(rows):
    primary=[r for r in rows if r['scope']=='selected'];assert len(primary)==8 and {r['context'] for r in primary}==set(range(8))
    values=[rates(r) for r in primary];assert all(v['weighted_net_accuracy'] is not None for v in values)
    mean=lambda rr,k:sum(rates(r)[k] for r in rr)/len(rr)
    grouped=[dict(group=f'{key}={v}',weighted_net_accuracy=mean([r for r in primary if r[key]==v],'weighted_net_accuracy'),weighted_nll_delta=mean([r for r in primary if r[key]==v],'weighted_nll_delta')) for key in ('labeled_images','source_step') for v in sorted({r[key] for r in primary})]
    acc=mean(primary,'weighted_net_accuracy');nll=mean(primary,'weighted_nll_delta');positive=sum(v['weighted_net_accuracy']>0 and v['weighted_nll_delta']<0 for v in values)
    signal=acc>0 and nll<0 and positive>=6 and all(v['weighted_net_accuracy']>0 and v['weighted_nll_delta']<0 for v in grouped)
    return dict(status='POSITIVE_CROSS_IMAGE_TARGET_QUALITY_SIGNAL' if signal else 'NO_RELIABLE_CROSS_IMAGE_TARGET_QUALITY_SIGNAL',equal_context_weighted_net_accuracy=acc,equal_context_weighted_nll_delta=nll,joint_positive_contexts=positive,groups=grouped,train_only_signal=signal,campaign_success=False,independent_confirmation=False,held_out_from_model_training=False)


def selfcheck():
    b=torch.tensor([[[[.8,.1,.1]],[[.1,.8,.1]],[[.1,.1,.8]]]])
    a=b.roll(1,1);y=torch.tensor([[[1,1,0]]]);mask=torch.ones_like(y,dtype=torch.bool);weight=torch.tensor([[[.5,1.,1.5]]])
    r=measure(b,a,y,mask,weight)
    assert (r['repaired'],r['harmed'],r['both_wrong_changed'])==(2,1,0) and r['weighted_repaired']==2 and r['weighted_harmed']==1
    assert rates(r)['weighted_net_accuracy']==1/3 and rates(r)['nll_delta']<0
    assert rates(measure(b,a,y,mask&False,weight))['nll_delta'] is None
    same=measure(b,b,y,mask,weight);assert same['changed']==0 and rates(same)['nll_delta']==0
    rows=[dict(context=i,scope='selected',labeled_images=2 if i%2==0 else 8,source_step=2000 if i<4 else 8000,**r) for i in range(8)]
    assert analyze(rows)['train_only_signal']
    for v in rows:v.update(same)
    assert not analyze(rows)['train_only_signal']
    return dict(status='PASS',model_forwards=0,image_reads=0,query_calls=0,optimizer_calls=0,checks=['repair/harm partition','weighted counts','soft target NLL','empty mask','identity target','diagnostic gate'])

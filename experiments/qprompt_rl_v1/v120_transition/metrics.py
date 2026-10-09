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



def scopes(base,corrected,labels,selected):
    original=base.argmax(1);proposed=corrected.argmax(1)
    masks={'selected':selected}
    for a in range(3):
        for b in range(3):
            mask=selected&(original==a)&(proposed==b)
            masks[f'transition_{a}_{b}']=mask
            for y in range(3):masks[f'transition_{a}_{b}_truth_{y}']=mask&(labels==y)
    assert torch.equal(sum(masks[f'transition_{a}_{b}'].int() for a in range(3) for b in range(3)),selected.int())
    return masks


def analyze(rows):
    primary={r['context']:r for r in rows if r['scope']=='selected'}
    assert set(primary)==set(range(8)) and all(r['weight']>0 for r in primary.values())
    actions=[]
    for a in range(3):
        for b in range(3):
            if a==b:continue
            rr=sorted([r for r in rows if r['scope']==f'transition_{a}_{b}'],key=lambda r:r['context'])
            assert len(rr)==8 and {r['context'] for r in rr}==set(range(8))
            values=[dict(context=r['context'],source_step=r['source_step'],labeled_images=r['labeled_images'],pixels=r['pixels'],weighted_net_accuracy=(r['weighted_repaired']-r['weighted_harmed'])/primary[r['context']]['weight'],weighted_nll_delta=(r['weighted_corrected_nll']-r['weighted_base_nll'])/primary[r['context']]['weight']) for r in rr]
            mean=lambda group,k:sum(r[k] for r in group)/len(group)
            grouped=[dict(group=f'{key}={v}',weighted_net_accuracy=mean([r for r in values if r[key]==v],'weighted_net_accuracy'),weighted_nll_delta=mean([r for r in values if r[key]==v],'weighted_nll_delta')) for key in ('labeled_images','source_step') for v in sorted({r[key] for r in values})]
            acc=mean(values,'weighted_net_accuracy');nll=mean(values,'weighted_nll_delta')
            positive=sum(r['weighted_net_accuracy']>0 and r['weighted_nll_delta']<0 for r in values)
            signal=acc>0 and nll<0 and positive>=6 and all(r['weighted_net_accuracy']>0 and r['weighted_nll_delta']<0 for r in grouped)
            actions.append(dict(transition=f'{a}->{b}',pixels=sum(r['pixels'] for r in rr),repaired=sum(r['repaired'] for r in rr),harmed=sum(r['harmed'] for r in rr),equal_context_weighted_net_accuracy=acc,equal_context_weighted_nll_delta=nll,joint_positive_contexts=positive,nonempty_contexts=sum(r['pixels']>0 for r in rr),qualified=signal,contexts=values,groups=grouped))
    qualified=[a['transition'] for a in actions if a['qualified']]
    return dict(status='OBSERVABLE_TRANSITION_QUALITY_SIGNAL' if qualified else 'NO_RELIABLE_OBSERVABLE_TRANSITION_SIGNAL',qualified_transitions=qualified,actions=actions,train_only_signal=bool(qualified),campaign_success=False,independent_confirmation=False,training_benefit_tested=False)


def selfcheck():
    b=torch.tensor([[[[.8,.1,.1,.8]],[[.1,.8,.1,.1]],[[.1,.1,.8,.1]]]])
    a=b.roll(1,1);y=torch.tensor([[[1,1,0,255]]]);mask=y!=255;weight=torch.tensor([[[.5,1.,1.5,1.]]])
    masks=scopes(b,a,y,mask);assert len(masks)==37
    r=measure(b,a,y,mask,weight)
    assert (r['repaired'],r['harmed'],r['both_wrong_changed'])==(2,1,0)
    assert r['weighted_repaired']==2 and r['weighted_harmed']==1
    rows=[]
    for i in range(8):
        for scope,m in masks.items():rows.append(dict(context=i,scope=scope,labeled_images=2 if i%2==0 else 8,source_step=2000 if i<4 else 8000,**measure(b,a,y,m,weight)))
    result=analyze(rows);assert result['qualified_transitions']==['0->1','2->0']
    assert result['actions'][0]['joint_positive_contexts']==8
    for rr in rows:
        if rr['context'] in (0,1,2):
            rr.update(repaired=0,harmed=0,weighted_repaired=0.,weighted_harmed=0.,weighted_corrected_nll=rr['weighted_base_nll'])
    assert not analyze(rows)['qualified_transitions']
    identity=scopes(b,b,y,mask)
    assert all(not m.any() for k,m in identity.items() if k.startswith('transition_') and k.split('_')[1]!=k.split('_')[2])
    assert rates(measure(b,a,y,mask&False,weight))['nll_delta'] is None
    return dict(status='PASS',model_forwards=0,image_reads=0,query_calls=0,optimizer_calls=0,checks=['all9 transitions and27 truth partitions','repair/harm weights','all-selected denominator','empty action remains zero effect','six-action gate','fewer-than6 positive fails','identity and ignore255'])

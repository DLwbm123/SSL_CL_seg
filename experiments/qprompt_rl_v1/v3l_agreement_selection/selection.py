"""Frozen teacher agreement selection and class-count matched random control."""
import torch
from experiments.qprompt_rl_v1.v3b_lrref_endpoint import engine as e
from experiments.qprompt_rl_v1.v3b_lrref_endpoint.maths import u_loss


def counts(mask,labels):
    return [[int((mask[b]&(labels[b]==c)).sum()) for c in range(3)] for b in range(len(mask))]


def random_mask(admitted,labels,target,seed_parts):
    available=counts(admitted,labels)
    if len(target)!=len(available) or any(len(row)!=3 for row in target):raise ValueError('count shape')
    result=torch.zeros_like(admitted)
    for b in range(len(admitted)):
        for c in range(3):
            n=target[b][c]
            if type(n) is not int or not 0<=n<=available[b][c]:raise ValueError('count outside population')
            indices=(admitted[b]&(labels[b]==c)).flatten().nonzero().flatten()
            g=torch.Generator().manual_seed(e.stable('V3L/random',*seed_parts,b,c))
            take=torch.randperm(len(indices),generator=g)[:n].to(indices.device)
            result[b].view(-1)[indices[take]]=True
    assert counts(result,labels)==target
    return result


def bind(t,ledger,root):
    original_losses,original_update=t.losses,t.update
    reference={}  # Paired exogenous count schedule; agreement precedes random in each block.
    def losses():
        kind=t.options.get('u_selection','none')
        if kind=='none' or t.action in (-1,0) or t.u_weight()==0:return original_losses()
        if kind not in ('agree','random'):raise ValueError('unknown U selection')
        selected=None;record=None
        def observe(u,q,valid):
            nonlocal selected,record
            with t.readonly(),torch.no_grad():
                label=q.argmax(1);admitted=valid.bool()&(q.max(1).values>.7)
                population=counts(admitted,label)
                if kind=='agree':selected=admitted&(t.clean(u).argmax(1)==label)
                else:
                    if t.step not in reference:raise RuntimeError('paired agreement count schedule missing')
                    source=reference[t.step]
                    assert source['population']==population, 'paired fixed teacher/data population differs'
                    selected=random_mask(admitted,label,source['selected'],(t.provider.seed,t.provider.domain,t.step))
                record=dict(population=population,selected=counts(selected,label),valid_pixels=int(valid.sum()))
        labeled,p,q,valid=t.components(observe_u=observe)
        ul,_=u_loss(p,q,valid,t.action,selection=selected)
        t.last.update(active_U=True,unlabeled_loss=float(ul.detach())*t.u_weight(),selection=record)
        return labeled,t.u_weight()*ul,None
    def update(*args,**kwargs):
        step=t.step;kind=t.options.get('u_selection','none')
        result=original_update(*args,**kwargs)
        if kind!='none' and t.last.get('active_U'):
            record=e.cpu(t.last['selection'])
            if kind=='agree':
                if ledger.category=='main':assert step not in reference, 'duplicate retained count schedule'
                reference[step]=record
            else:assert record==reference[step], 'random population/count/geometry mismatch'
            if ledger.category=='main':
                e.append(root/'SELECTION_LEDGER.private.jsonl',dict(seed=t.provider.seed,domain=t.provider.domain,step=step,kind=kind,**record))
            t.telemetry['selection_updates']=t.telemetry.get('selection_updates',0)+1
        return result
    t.losses,t.update=losses,update
    return t


def check():
    q=torch.tensor([[[[.8,.1,.1,.8]],[[.1,.8,.1,.1]],[[.1,.1,.8,.1]]]])
    p=torch.tensor([[[[.6,.2,.2,.6]],[[.2,.6,.2,.2]],[[.2,.2,.6,.2]]]],requires_grad=True)
    valid=torch.ones((1,1,4),dtype=torch.bool);selection=torch.tensor([[[True,False,True,False]]])
    original,_=u_loss(p,q,valid,2);full,_=u_loss(p,q,valid,2,torch.ones_like(valid));partial,_=u_loss(p,q,valid,2,selection)
    assert torch.equal(original,full) and torch.allclose(partial,original/2)
    grad=torch.autograd.grad(partial,p)[0];assert not grad[:,:,0,[1,3]].any()
    zero,_=u_loss(p,q,valid,2,torch.zeros_like(valid));assert zero==0 and zero.requires_grad
    labels=q.argmax(1);target=[[1,1,0]];before=torch.get_rng_state().clone()
    a=random_mask(valid,labels,target,(168,'D',3));b=random_mask(valid,labels,target,(168,'D',3))
    assert torch.equal(a,b) and torch.equal(before,torch.get_rng_state()) and counts(a,labels)==target
    try:random_mask(valid,labels,[[3,1,0]],(1,))
    except ValueError:pass
    else:raise AssertionError('invalid target accepted')
    print('PASS optional-mask original parity, valid-denominator dose, detached selection/zero mask, exact class counts, stateless random and invalid target')


if __name__=='__main__':check()

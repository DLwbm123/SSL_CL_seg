"""Actual-checkpoint zero-update admission; no optimizer calls."""
import torch
from lcrseg.engine.checkpoint import capture_rng_state,restore_rng_state
from sc_bayeseg.model import Model as OriginalModel
from sc_bayeseg.run import snapshot,same
from sc_bayeseg.calibration_v1.run import evaluate
from sc_bayeseg.split_v1.model import semantic


def zero_audit(model,method,trainer,payload,image,y):
    initial=snapshot(method,trainer,2400);rng=capture_rng_state()
    old=OriginalModel().cuda();old.load_state_dict(payload['model'])
    with torch.no_grad():
        oldh,_=old.proposal(image,sample=False);newh,_=model.proposal(image,sample=False)
        assert torch.equal(oldh['ell'],newh['ell'])
    baseline=evaluate(model,image,y)
    assert all(v>=.8 for row in baseline['stochastic_dice'] for v in row),'new stochastic chain failed admission'
    names=dict(model.named_parameters());source=list(old.named_parameters())
    assert list(names)[:len(source)]==[n for n,_ in source]
    assert all(torch.equal(p,names[n]) for n,p in source)
    assert all(torch.equal(v,model.physical_encoder.state_dict()[k]) for k,v in model.shape_encoder.state_dict().items())
    sem=[p for n,p in names.items() if semantic(n)];phys=list(model.physical_encoder.parameters())
    assert not ({id(p) for p in sem}&{id(p) for p in phys})
    assert not ({p.untyped_storage().data_ptr() for p in sem}&{p.untyped_storage().data_ptr() for p in phys})
    # Replay samples using the explicit Gaussian formula, with no stochastic v node.
    restore_rng_state(rng)
    with torch.no_grad():h,logq=model.proposal(image)
    restore_rng_state(rng)
    def node(raw):
        mean,rawsd=raw.chunk(2,1);sdlog=-4+3*torch.tanh(rawsd/3)
        value=mean+sdlog.exp()*torch.randn_like(mean)
        lp=(-.5*((value-mean)*(-sdlog).exp()).square()-sdlog-.5*torch.log(torch.tensor(2*torch.pi,device=image.device))).flatten(1).sum(1)
        return value,lp
    with torch.no_grad():
        v=model.shape_encoder(image).chunk(2,1)[0]
        x=node(model.physical_encoder(image));m=node(model.appearance_encoder(image));ell=node(model.logit_encoder(v));c=node(model.code_encoder(ell[0]))
        assert all(torch.equal(h[k],a[0]) for k,a in zip(('x','m','ell','c'),(x,m,ell,c)))
        assert torch.equal(logq,sum(a[1] for a in (x,m,ell,c)))
    restore_rng_state(rng);model.draw()
    try:
        _,terms,_=model.bound(image,y)
        gi=torch.autograd.grad(-terms['image'].mean()/128**2,sem,allow_unused=True,retain_graph=True)
        gs=torch.autograd.grad(-terms['shape'].mean()/128**2,sem,allow_unused=True)
        assert all(g is None or torch.count_nonzero(g)==0 for g in gi)
        shape_norm=float(sum(g.square().sum() for g in gs if g is not None).sqrt())
        assert shape_norm>0 and all(g is None or torch.isfinite(g).all() for g in gs)
    finally:model.clear_draw()
    restore_rng_state(rng)
    assert same(initial,snapshot(method,trainer,2400))
    oldstate=payload['trainer']['optimizer_state'];current=trainer.state_dict()['optimizer_state']
    assert same(oldstate['state'],current['state'])
    assert len(current['param_groups'])==len(oldstate['param_groups'])==1
    oldids=oldstate['param_groups'][0]['params'];newids=current['param_groups'][0]['params']
    assert newids[:len(oldids)]==oldids and len(newids)==len(oldids)+len(phys)
    assert not (set(newids[len(oldids):])&set(current['state']))
    result=dict(status='PASS',optimizer_updates=0,extra_reverse_calls=2,deterministic_logits_equal=True,
                stochastic_baseline=baseline,all_six_stochastic_dice_pass=True,original_model_parameters_equal=True,
                original_adam_states_equal=True,new_physical_adam_empty=True,parameter_order_preserved=True,
                parameter_and_storage_independent=True,four_node_logq_equal=True,image_semantic_gradient_zero=True,
                shape_semantic_gradient_norm=shape_norm,complete_state_unchanged=True,
                old_parameter_tensors=len(source),new_physical_parameter_tensors=len(phys),
                old_parameter_count=sum(p.numel() for _,p in source),new_physical_parameter_count=sum(p.numel() for p in phys))
    del old;torch.cuda.empty_cache()
    return result,baseline

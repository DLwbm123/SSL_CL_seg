"""Fixed engineering qualification, never validation-driven method selection."""
import copy,time
from pathlib import Path
import torch
from .model import style,weighted_kl,MainTrainer
from .protocol import nodes,options
from .runtime import setup,seal,write,save,restore,evaluate
from ..five_frameworks_v1.checkpoint import PhysicalLedger


def unit_checks():
    x=torch.arange(32,dtype=torch.float32).reshape(2,1,4,4)/32;d=x.flip(0)
    out=style(x,d,(-1,1));mu=.8*x.mean((-2,-1),keepdim=True)+.2*d.mean((-2,-1),keepdim=True)
    assert torch.allclose(out.mean((-2,-1),keepdim=True),mu,atol=2e-6)
    assert torch.isfinite(style(torch.zeros_like(x),torch.ones_like(x),(-1,1))).all()
    q=torch.tensor([[[[.05]],[[.9]],[[.05]]]])
    a=torch.tensor([[[[.9]],[[.05]],[[.05]]]])
    z=torch.zeros_like(q,requires_grad=True);valid=torch.ones(1,1,1,dtype=torch.bool)
    plain,kept,_=weighted_kl(z,q,valid);conflict,kept,_=weighted_kl(z,q,valid,a)
    assert kept.all() and torch.allclose(conflict,.5*plain)
    bgq=a;_,kept,_=weighted_kl(z,bgq,valid,q);assert not kept.any()
    near=torch.tensor([[[[.1]],[[.71]],[[.19]]]])
    v,_,_=weighted_kl(z,near,valid);assert v<plain
    conflict.backward();assert z.grad is not None and torch.isfinite(z.grad).all()
    no_pas=torch.zeros_like(valid)
    assert MainTrainer.pseudo_admission(None,q,valid,no_pas).all()
    assert not MainTrainer.pseudo_admission(None,a,valid,no_pas).any()
    from ..five_frameworks_v1.native_parent import lr
    gen=torch.Generator().manual_seed(712)
    rgb=torch.randn(16,3,3,3,generator=gen);mono=torch.randn(16,1,3,3,generator=gen)
    before,_=lr.basis(rgb,163,'golden','first');after,_=lr.basis(rgb,163,'golden','first',minimum_input_width=9)
    assert all(torch.equal(before[k],after[k]) for k in before),'RGB basis changed'
    single,meta=lr.basis(mono,163,'golden','first',minimum_input_width=9)
    assert meta['numerical_rank']==9 and single['V'].shape==(9,1) and single['G'].shape==(8,9)
    for c in (2,3,4):
        logits=torch.randn(2,c,8,8,requires_grad=True);target=logits.detach().softmax(1)
        v,_,_=weighted_kl(logits,target,torch.ones(2,8,8,dtype=torch.bool));v.backward()
        assert torch.isfinite(logits.grad).all()
    assert len(nodes())==30 and sum(n['steps'] for n in nodes())==511600
    return dict(style=True,confidence_conflict_denominator=True,background_veto=True,foreground_no_PAS=True,
                RGB_basis_unchanged=True,mono_full_rank_projection=True,classes_2_3_4=True,budget=True)


class FixedData:
    def __init__(self,provider,x,y,ids):
        self.__dict__.update(provider.__dict__);self.base=provider;self.x=x;self.y=y;self.ids=ids
    def labeled(self,step,stream='labeled'):
        self.l_reads+=1;return self.x,self.y,self.ids
    def semantic_metadata(self):return self.base.semantic_metadata()
    def unlabeled(self,step):raise PermissionError('overfit is L-only engineering')


def qualification(config):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    root=Path(config['output']);root.mkdir(parents=True,exist_ok=False)
    write(root/'STATUS.json',dict(status='RUNNING',purpose='engineering',optimizer_cap=330))
    result=dict(unit=unit_checks(),datasets={});ledger=PhysicalLedger(root/'physical_updates.jsonl');started=time.time()
    try:
        for ds in ('fundus','prostate','mnms'):
            out=root/ds;out.mkdir();node=copy.deepcopy(next(n for n in nodes() if n['dataset']==ds and n['stage']==0 and n['seed']==163));node.update(steps=5,scope='smoke')
            trainer=setup(node,config['data'],config['reference'],'cuda');identity=dict(dataset=ds,seed=163,stage=0)
            initial_ema=[p.detach().clone() for p in trainer.ema.parameters()]
            trainer.update(physical=ledger.append);trainer.update(physical=ledger.append)
            save(trainer,out/'resume.pt',identity);trainer.update(physical=ledger.append)
            expected={k:v.detach().clone() for k,v in trainer.model.state_dict().items()};expected_last=copy.deepcopy(trainer.last)
            assert trainer.provider.u_reads>0 and trainer.last['active_U']
            assert any(not torch.equal(a,b) for a,b in zip(initial_ema,trainer.ema.parameters()))
            resumed=setup(node,config['data'],config['reference'],'cuda',initialize=False)
            restore(resumed,out/'resume.pt',identity);resumed.update(physical=ledger.append)
            assert all(torch.equal(v,resumed.model.state_dict()[k]) for k,v in expected.items()),'resume student mismatch'
            assert resumed.last==expected_last,'resume losses mismatch'
            sample,_,_=trainer.provider.labeled(7,'merge_check')
            with torch.no_grad():
                trainer.model.eval().requires_grad_(False);trainer.model.role='eval'
                prediction=trainer.model(sample,mode='eval')
                deployed=seal(trainer,node,out/'source.pt');merged=deployed(sample)
                assert torch.allclose(prediction,merged,rtol=2e-4,atol=2e-5),'source merge'
            del resumed,initial_ema,expected,trainer,deployed
            target_node=copy.deepcopy(next(n for n in nodes() if n['dataset']==ds and n['stage']==1 and n['seed']==163 and n['order']==1));target_node.update(steps=5,scope='smoke')
            target=setup(target_node,config['data'],config['reference'],'cuda',out/'source.pt')
            assert target.model.sidecar.q.is_contiguous()
            anchor={k:v.detach().clone() for k,v in target.model.entry_anchor.state_dict().items()}
            target.update(physical=ledger.append);target.update(physical=ledger.append);save(target,out/'target_resume.pt',dict(dataset=ds,seed=163,stage=1))
            target.update(physical=ledger.append)
            expected={k:v.detach().clone() for k,v in target.model.state_dict().items()}
            resumed=setup(target_node,config['data'],config['reference'],'cuda',out/'source.pt',initialize=False)
            restore(resumed,out/'target_resume.pt',dict(dataset=ds,seed=163,stage=1));resumed.update(physical=ledger.append)
            assert all(torch.equal(v,resumed.model.state_dict()[k]) for k,v in expected.items()),'target resume'
            assert all(torch.equal(v,target.model.entry_anchor.state_dict()[k]) for k,v in anchor.items()),'entry modified'
            norms={name:sum(float(p.grad.double().square().sum()) for p in params if p.grad is not None)**.5 for name,params in [('A',target.model.parent.parameter_groups()['input_factors']),('B',target.model.parent.parameter_groups()['output_factors']),('R',target.model.sidecar.optimizer_parameters())]}
            assert all(v>0 for v in norms.values()),'missing A/B/R update gradient'
            sample,_,_=target.provider.labeled(7,'target_merge_check')
            with torch.no_grad():
                target.model.eval().requires_grad_(False);target.model.role='eval';prediction=target.model(sample,mode='eval')
                deployed=seal(target,target_node,out/'target.pt')
                assert torch.allclose(prediction,deployed(sample),rtol=2e-4,atol=2e-5),'target merge'
            eval_smoke=evaluate(deployed,config['data'],target_node,out)
            peak=torch.cuda.max_memory_allocated()
            del target,resumed,anchor,expected,deployed
            torch.cuda.empty_cache()
            # Two fixed labeled patients; selection may use L labels, never U GT.
            overfit_node=copy.deepcopy(node);overfit_node['steps']=96
            overfit=setup(overfit_node,config['data'],config['reference'],'cuda')
            access=overfit.provider._l;xs=[];ys=[];ids=access.patients[:2]
            for patient in ids:
                index=access.by_patient[patient][0]
                volume=access.read(index)
                z=None if ds=='fundus' else int((volume['label']>0).flatten(1).sum(1).argmax())
                item=access.read(index,z);xs.append(item['image']);ys.append(item['label'])
            x=torch.stack(xs).cuda();y=torch.stack(ys).cuda()
            overfit.provider=FixedData(overfit.provider,x,y,tuple(ids));overfit.options['warmup_fraction']=1.
            losses=[]
            for i in range(96):
                if len(ledger.path.read_text().splitlines())>=330:raise RuntimeError('engineering optimizer cap')
                overfit.update(physical=ledger.append);losses.append(overfit.last['labeled_loss'])
            before=sum(losses[:4])/4;after=sum(losses[-4:])/4
            assert after<.8*before,f'two-case fitting did not lower loss enough: {before} -> {after}'
            result['datasets'][ds]=dict(source_backward=True,target_backward=True,EMA_update=True,entry_frozen=True,exact_source_resume=True,
                exact_target_resume=True,deploy_merge=True,stage_transition=True,validation_volume_metrics=True,
                gradient_norms=norms,peak_memory_bytes=peak,two_case_overfit=dict(steps=96,initial_loss=before,final_loss=after),
                validation_smoke_domains=list(eval_smoke))
            write(root/'PROGRESS.json',result);print(ds,'PASS',before,after,flush=True)
            del overfit,x,y;torch.cuda.empty_cache()
        result.update(status='PASS',code_commit=config['code_commit'],physical_optimizer_updates=len(ledger.path.read_text().splitlines()),elapsed_seconds=time.time()-started)
        write(root/'RESULT.json',result);write(root/'STATUS.json',dict(status='PASS'))
    except BaseException as exc:
        import traceback
        result.update(status='FAIL',error=str(exc),traceback=traceback.format_exc(),elapsed_seconds=time.time()-started)
        write(root/'RESULT.json',result);write(root/'STATUS.json',dict(status='FAIL',error=str(exc)));raise

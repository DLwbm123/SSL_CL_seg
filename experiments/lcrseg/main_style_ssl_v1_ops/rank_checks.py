"""Bounded engineering checks on the existing source; no source/test training."""
import copy
import json
import os
from pathlib import Path
import torch
from experiments.lcrseg.lctx_weight_memory_v0_1 import core as lr
from experiments.lcrseg.main_style_ssl_v1.checks import unit_checks, FixedData
from experiments.lcrseg.main_style_ssl_v1.protocol import nodes
from experiments.lcrseg.main_style_ssl_v1.runtime import setup, save, restore, write
from experiments.lcrseg.five_frameworks_v1.checkpoint import PhysicalLedger
from experiments.lcrseg.tests.five_frameworks_v1.test_integration import assert_state_equal


def units():
    result=unit_checks()
    g=torch.Generator().manual_seed(301)
    for shape in ((16,3,3,3),(16,1,3,3),(64,64,3,3)):
        weight=torch.randn(shape,generator=g)
        old,_=lr.basis(weight,164,'RUNMC','healthy',minimum_input_width=9)
        new,_=lr.basis(weight,164,'RUNMC','healthy',minimum_input_width=9,require_full_rank=False)
        assert all(torch.equal(old[k],new[k]) for k in old)
    weight=torch.zeros(64,576);weight[torch.arange(61),torch.arange(61)]=torch.arange(61,0,-1).float()
    weight=weight.reshape(64,64,3,3)
    try:lr.basis(weight,164,'RUNMC','rank61')
    except ValueError as exc:assert 'rank=61' in str(exc)
    else:raise AssertionError('legacy full-rank guard was relaxed')
    tensors,meta=lr.basis(weight,164,'RUNMC','rank61',require_full_rank=False)
    assert meta['numerical_rank']==61 and tensors['V'].shape==(576,32)
    torch.testing.assert_close(tensors['V'].T@tensors['V'],torch.eye(32,dtype=torch.float64),atol=1e-12,rtol=0)
    for invalid in (torch.zeros_like(weight),weight*float('nan'),weight*float('inf')):
        try:lr.basis(invalid,164,'RUNMC','invalid',require_full_rank=False)
        except ValueError:pass
        else:raise AssertionError('invalid weights admitted')
    weight.flatten(1)[31:]=0
    try:lr.basis(weight,164,'RUNMC','rank31',require_full_rank=False)
    except ValueError as exc:assert 'rank=31' in str(exc) and 'required=32' in str(exc)
    else:raise AssertionError('insufficient protected rank admitted')
    return dict(**result,healthy_basis_bitwise_equal=True,legacy_guard_retained=True,
                rank61_protected32=True,invalid_and_rank31_rejected=True)


def diagnose(root):
    result=[]
    for seed in (163,164):
        folder=root/'jobs'/f'prostate_s{seed}_source'
        sealed=torch.load(folder/'sealed.pt',map_location='cpu',weights_only=False)
        final=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
        assert sealed['updates']==final['step']==13600 and sealed['sealed'] and sealed['seed']==seed
        assert json.loads((folder/'EXIT.json').read_text())['exit_code']==0
        assert all(torch.isfinite(v).all() for v in sealed['native'].values())
        assert all(torch.equal(v,final['student']['parent.native.'+k]) for k,v in sealed['native'].items())
        for name in lr.LAYERS:
            weight=sealed['native'][name];w=weight.double().flatten(1);m,d=w.shape
            _,s,_=torch.linalg.svd(w,full_matrices=False);tol=float(s[0]*max(m,d)*torch.finfo(torch.float64).eps)
            rank=int((s>tol).sum());k=min(m//2,d-8)
            tensors,meta=lr.basis(weight,seed,'RUNMC',name,minimum_input_width=9,require_full_rank=False)
            assert rank>=k
            if rank==min(m,d):
                old,_=lr.basis(weight,seed,'RUNMC',name,minimum_input_width=9)
                assert all(torch.equal(old[key],tensors[key]) for key in old)
            result.append(dict(seed=seed,layer=name,shape=list(w.shape),numerical_rank=rank,
                               protected_k=k,old_required=min(m,d),tolerance=tol,singular_values=s.tolist(),
                               protected_energy=meta['selected_energy'],zero_rows=int((w.norm(dim=1)==0).sum())))
    return result


def qualification(config):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    out=Path(config['output']);out.mkdir(exist_ok=False)
    ledger=PhysicalLedger(out/'physical_updates.jsonl');result={}
    try:
        result['unit']=units();result['diagnostic']=diagnose(Path(config['root']))
        predecessor=Path(config['root'])/'jobs/prostate_s164_source/sealed.pt'
        source_stat=predecessor.stat()
        node=copy.deepcopy(next(n for n in nodes() if n['id']=='prostate_s164_o1_t1'))
        node.update(steps=5,scope='smoke')
        target=setup(node,config['data'],config['reference'],'cuda',predecessor)
        anchor=copy.deepcopy(target.model.entry_anchor.state_dict());initial_ema=copy.deepcopy(target.ema.state_dict())
        identity=dict(dataset='prostate',seed=164,stage=1)
        target.update(physical=ledger.append);target.update(physical=ledger.append)
        save(target,out/'resume.pt',identity);target.update(physical=ledger.append)
        assert target.last['active_U'] and target.provider.u_reads>0
        expected=copy.deepcopy(target.model.state_dict());expected_ema=copy.deepcopy(target.ema.state_dict())
        expected_optimizer=copy.deepcopy(target.optimizer.state_dict());expected_last=copy.deepcopy(target.last)
        assert any(not torch.equal(v,target.ema.state_dict()[k]) for k,v in initial_ema.items())
        groups=target.model.parent.parameter_groups()
        norms={name:sum(float(p.grad.double().square().sum()) for p in params if p.grad is not None)**.5
               for name,params in [('A',groups['input_factors']),('B',groups['output_factors']),('R',target.model.sidecar.optimizer_parameters())]}
        assert all(v>0 for v in norms.values())
        resumed=setup(node,config['data'],config['reference'],'cuda',predecessor,initialize=False)
        restore(resumed,out/'resume.pt',identity);resumed.update(physical=ledger.append)
        for a,b in ((expected,resumed.model.state_dict()),(expected_ema,resumed.ema.state_dict()),
                    (expected_optimizer,resumed.optimizer.state_dict()),(expected_last,resumed.last),
                    (anchor,target.model.entry_anchor.state_dict())):assert_state_equal(a,b)
        assert all(p.grad is None and not p.requires_grad for p in target.model.entry_anchor.parameters())
        x,_,_=target.provider.labeled(7,'target_merge_check')
        with torch.no_grad():
            target.model.eval().requires_grad_(False);target.model.role='eval'
            before=target.model(x,mode='eval');deployed=target.model.deploy()
            torch.testing.assert_close(before,deployed(x),rtol=2e-4,atol=2e-5)
        del target,resumed,expected,expected_ema,expected_optimizer,anchor,initial_ema,deployed,x
        torch.cuda.empty_cache()
        # Two labeled BMC training cases; this is a fixed engineering fixture.
        fit_node=copy.deepcopy(node);fit_node['steps']=96
        fit=setup(fit_node,config['data'],config['reference'],'cuda',predecessor)
        access=fit.provider._l;xs=[];ys=[];ids=access.patients[:2]
        for patient in ids:
            index=access.by_patient[patient][0];volume=access.read(index)
            z=int((volume['label']>0).flatten(1).sum(1).argmax());item=access.read(index,z)
            xs.append(item['image']);ys.append(item['label'])
        fit.provider=FixedData(fit.provider,torch.stack(xs).cuda(),torch.stack(ys).cuda(),tuple(ids))
        fit.options['warmup_fraction']=1.;losses=[]
        for _ in range(96):
            fit.update(physical=ledger.append);losses.append(fit.last['labeled_loss'])
        first=sum(losses[:4])/4;last=sum(losses[-4:])/4
        assert last<.8*first,f'two-case fitting failed: {first} -> {last}'
        assert fit.provider.u_reads==0 and len(ledger.path.read_text().splitlines())==100
        now=predecessor.stat();assert (source_stat.st_ino,source_stat.st_size,source_stat.st_mtime_ns)==(now.st_ino,now.st_size,now.st_mtime_ns)
        result.update(status='PASS',source_optimizer_updates=0,engineering_optimizer_updates=100,
                      formal_optimizer_updates=0,gradient_norms=norms,exact_student_ema_optimizer_resume=True,
                      golden_batch_loss_replay=True,entry_frozen=True,deploy_merge=True,
                      two_case_overfit=dict(steps=96,initial_loss=first,final_loss=last),
                      source_checkpoint_unchanged=True,test_accessor=False)
    except BaseException as exc:
        import traceback
        result.update(status='FAIL',error=str(exc),traceback=traceback.format_exc(),
                      engineering_optimizer_updates=0 if not ledger.path.exists() else len(ledger.path.read_text().splitlines()))
        raise
    finally:write(out/'RESULT.json',result)


if __name__=='__main__':
    qualification(json.loads(Path(os.environ['RESOURCE_CONFIG']).read_text()))

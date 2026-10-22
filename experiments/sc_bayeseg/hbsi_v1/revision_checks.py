"""Independent checks for the review revision, never a real-training qualification."""
import copy
from dataclasses import replace
import itertools
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch.distributions import Normal, MultivariateNormal
from .budgets import ResourceLedger, Counters, BudgetExceeded
from .config import Config
from .data_adapter import CurrentTrainingData
from .evidence import evidence_set
from .local_inference import Refiner
from .predictive import combine_slice_banks, nondegenerate, InferenceBudget, Predictor
from .qualification import write_receipt, require_receipt, fingerprint
from .spatial_gaussian import sar, sar_solve, sar_logdet
from .stream_state import StreamState, digest
from .tests import tiny, batches, check_close, rejected
from .training import run_stage, require_resume

SEED = 20261022


@patch('experiments.sc_bayeseg.hbsi_v1.qualification.code_fingerprint', return_value='fixture-only-code-identity')
def R01(out, _mock_code):
    from .runner import train
    cfg = Config()
    evidence = fingerprint({'synthetic_fixture': 1})
    report = out / 'scientific_report.json'
    report.write_text(json.dumps({'scientific_state': 'SYNTHETIC_LEARNING_QUALIFIED', 'fixture_only': True}))
    receipt = out / 'receipt.json'
    valid = write_receipt(receipt, 'SYNTHETIC_LEARNING', cfg, report, evidence, code_version='fixture')
    require_receipt(receipt, 'SYNTHETIC_LEARNING', cfg, evidence)
    args = SimpleNamespace(output=out, command='train-source', budget_seconds=10., run_id='refused', resume_checkpoint=None, authorize_real_training=True)
    protocol = dict(steps=1, qualification_receipt=str(receipt), qualification_evidence_fingerprint=evidence)
    ledger = ResourceLedger(out / 'ledger')
    rejected_cases = []
    for field, value in (('status', 'FAIL'), ('code_fingerprint', 'stale'), ('config_fingerprint', 'wrong'),
                         ('evidence_fingerprint', 'wrong'), ('kind', 'UNIT'), ('checkpoint_fingerprint', 'wrong')):
        changed = dict(valid, **{field: value})
        receipt.write_text(json.dumps(changed))
        with patch('experiments.sc_bayeseg.hbsi_v1.runner.storage_check'), patch.object(CurrentTrainingData, 'inspect_manifest') as inspect, patch.object(CurrentTrainingData, '__init__') as opened:
            rejected(lambda: train(args, cfg, ledger, Counters(), protocol), PermissionError)
            assert inspect.call_count == opened.call_count == 0
        rejected_cases.append(field)
    receipt.write_text(json.dumps(valid))
    report.write_text(json.dumps({'scientific_state': 'SYNTHETIC_LEARNING_QUALIFIED', 'changed': True}))
    rejected(lambda: require_receipt(receipt, 'SYNTHETIC_LEARNING', cfg, evidence), PermissionError)
    rejected(lambda: require_receipt(out/'missing.json', 'SYNTHETIC_LEARNING', cfg, evidence), PermissionError)
    # Even a renamed oracle status cannot mint a learning PASS.
    oracle = out/'oracle.json'
    oracle.write_text(json.dumps({'scientific_state': 'SYNTHETIC_LEARNING_QUALIFIED', 'oracle': True}))
    assert write_receipt(out/'oracle_receipt.json', 'SYNTHETIC_LEARNING', cfg, oracle, evidence)['status']=='FAIL'
    model = out/'model_report.json'; model.write_text(json.dumps({'scientific_state': 'MODEL_LEARNING_QUALIFIED'}))
    state_file = out/'input.pt'; state_file.write_bytes(b'fixture checkpoint')
    model_receipt = out/'model_receipt.json'
    write_receipt(model_receipt, 'MODEL_LEARNING', cfg, model, evidence, state_file)
    require_receipt(model_receipt, 'MODEL_LEARNING', cfg, evidence, state_file)
    state_file.write_bytes(b'different checkpoint')
    rejected(lambda: require_receipt(model_receipt, 'MODEL_LEARNING', cfg, evidence, state_file), PermissionError)
    return dict(rejected_receipt_fields=rejected_cases, changed_report_rejected=True, missing_rejected=True,
                changed_checkpoint_rejected=True, oracle_cannot_qualify_learning=True, real_data_opens=0)


def R02(out):
    manifest = out/'manifests/training/lcrseg_v1_seed0.csv'; manifest.parent.mkdir(parents=True)
    header = 'dataset,site_or_vendor,primary_20pct_split,image_h5_relpath,label_h5_relpath\n'
    rows = ['prostate,A,train_labeled,a.h5,ay.h5\n', 'prostate,A,train_unlabeled,b.h5,\n', 'prostate,B,train_labeled,c.h5,cy.h5\n']
    manifest.write_text(header+''.join(rows))
    first = CurrentTrainingData.inspect_manifest(out, 'prostate', 'A', 'fixture-v1')
    other = CurrentTrainingData.inspect_manifest(out, 'prostate', 'B', 'fixture-v1')
    assert first['manifest_hash']==other['manifest_hash'] and first['evidence']!=other['evidence']
    s=tiny(); s.begin_task('one',first['manifest_hash'],first['evidence']); s.finalize_task()
    rejected(lambda:s.begin_task('renamed',first['manifest_hash'],first['evidence']))
    manifest.write_text(header+''.join(reversed(rows)))
    reordered=CurrentTrainingData.inspect_manifest(out, 'prostate', 'A', 'fixture-v1')
    assert first['evidence']==reordered['evidence'] and first['manifest_hash']!=reordered['manifest_hash']
    rejected(lambda:s.begin_task('another-output',reordered['manifest_hash'],reordered['evidence']))
    s.save(out/'sealed.pt'); loaded=StreamState.load(out/'sealed.pt')
    rejected(lambda:loaded.begin_task('renamed-after-load',first['manifest_hash'],first['evidence']))
    loaded.begin_task('legal-other-domain',other['manifest_hash'],other['evidence'])
    # Adding a new annotation does not license repeating the old image likelihood.
    relabelled=evidence_set('fixture-v1','A',[dict(kind='image',record='prostate:a.h5'),
                      dict(kind='label',record='prostate:a.h5',annotation_version='new-annotation')])
    rejected(lambda:s.begin_task('relabelled','f'*64,relabelled))
    rejected(lambda:evidence_set('fixture-v1','A',[dict(kind='image',record='same'),dict(kind='image',record='same')]))
    return dict(rename_rejected=True, record_order_invariant=True, output_rename_rejected=True,
                checkpoint_preserves_consumed_events=True, same_manifest_different_domain_allowed=True,
                partially_overlapping_evidence_rejected=True, duplicate_L_U_image_rejected=True,
                label_only_update='NOT_IMPLEMENTED_FAIL_CLOSED')


def R03(out):
    uniform=torch.zeros(4,dtype=torch.float64)
    collapsed=torch.tensor([0.,-100.,-100.,-100.],dtype=torch.float64)
    assert nondegenerate(uniform,.1,.5) and not nondegenerate(collapsed,.1,.5)
    assert not nondegenerate(torch.zeros(1,dtype=torch.float64),.1,.5)
    rejected(lambda:nondegenerate(torch.tensor([0.,float('nan')]),.1,.5),FloatingPointError)
    return dict(uniform_pass=True, single_dominant_fail=True, single_sample_fail=True,
                nonfinite_fail=True, threshold=dict(ess_strict_min=1., fraction_min=.1, max_weight=.5))


class FixtureData:
    def __init__(self):
        self.evidence=evidence_set('fixture-v1','A',[dict(kind='image',record='one'),dict(kind='label',record='one',annotation_version='v1')])
        self.counts={'L':1,'U':1}
    def batch(self,role,n,generator):
        x=torch.linspace(-.8,.8,16,dtype=torch.float64).reshape(1,1,4,4)
        # Consume sampler RNG too, so restore checks include sampling as well as MC noise.
        x=x+torch.randint(2,(1,),generator=generator).to(x)*.05
        result={'stage':'stage','role':'train'+role,'x':x}
        if role=='L':
            from .data_adapter import leaf_observations
            result['observations']=leaf_observations((x[:,0]>0).long(),2)
        return result


def R04(out):
    from . import budgets
    cfg=replace(tiny().config,inner_steps=1,global_mc=1,local_mc=1)
    data=FixtureData(); ledger=ResourceLedger(out/'budget',max_updates=16)
    count=budgets.ACTIVE_COUNTER or Counters()
    def initial():
        s=StreamState(cfg);s.begin_task('stage','a'*64,data.evidence);return s
    full=initial(); prior=digest(full.prior.state_dict())
    full=run_stage(full,data,ledger,count,'full',4,120.,'protocol',real_data=False)
    interrupted=initial()
    def stop(step):
        if step==2:raise RuntimeError('injected failure after durable step')
    rejected(lambda:run_stage(interrupted,data,ledger,count,'resume',4,120.,'protocol',after_checkpoint=stop,real_data=False))
    saved=ledger.data['training_sessions']['resume']; first_deadline=saved['identity']['deadline']
    path=Path(saved['checkpoint']); resumed=StreamState.load(path)
    assert resumed.steps==2 and resumed.prior_hash==prior
    rejected(lambda:require_resume(resumed,ledger,'resume',5,120.,'protocol',path),PermissionError)
    rejected(lambda:require_resume(resumed,ledger,'resume',4,121.,'protocol',path),PermissionError)
    rejected(lambda:require_resume(resumed,ledger,'resume',4,120.,'changed-protocol',path),PermissionError)
    with patch('experiments.sc_bayeseg.hbsi_v1.training.time.time',return_value=first_deadline+1):
        rejected(lambda:require_resume(resumed,ledger,'resume',4,120.,'protocol',path),BudgetExceeded)
    other_ledger=ResourceLedger(out/'different-output')
    rejected(lambda:require_resume(resumed,other_ledger,'resume',4,120.,'protocol',path),PermissionError)
    ledger=ResourceLedger(out/'budget',max_updates=16)
    resumed=run_stage(resumed,data,ledger,count,'resume',4,120.,'protocol',resume_checkpoint=path,real_data=False)
    for name in ('q','phi','model','projection'):
        assert digest(getattr(full,name).state_dict())==digest(getattr(resumed,name).state_dict())
    assert digest(full.memories['stage'].state_dict())==digest(resumed.memories['stage'].state_dict())
    assert torch.equal(full.generator.get_state(),resumed.generator.get_state())
    assert resumed.runtime['deadline']==first_deadline
    rejected(lambda:resumed.finalize_task())
    # Same path cannot overwrite a committed checkpoint.
    rejected(lambda:resumed.save(path),FileExistsError)
    return dict(interrupted_at_step=2, total_steps=4, fixture_outer_updates=8, resumed_equal_uninterrupted=True,
                rng_equal=True, original_prior_unchanged=True, original_deadline_unchanged=True,
                changed_budget_protocol_output_rejected=True, expired_deadline_rejected=True,
                failed_attempt_charged=True, partial_checkpoint_saved=True, create_only_checkpoint=True)


def R05(out):
    errors=[]
    for channels in (1,3):
        s=tiny(channels=channels,local_parameterization='sar_whitened',prior_aware_init=True)
        cfg=replace(s.config,width=6);s=StreamState(cfg)
        g=torch.Generator().manual_seed(SEED)
        theta=s.q.rsample(g);a=torch.zeros(3*channels,dtype=torch.float64)
        x=torch.randn(1,channels,4,6,generator=g,dtype=torch.float64)
        ctx=s.phi.context(x,theta,a);rho=s.phi.initial_offsets(1,x)
        z,lq=s.phi.rsample_and_log_prob(ctx,s.phi.noise(1,x,g),rho,s.model,theta)
        errors.append(check_close(lq,s.phi.log_prob(z,ctx,rho,s.model,theta)))
        raw,logdet=s.phi._transform(z,s.model,theta,inverse=True)
        back,_=s.phi._transform(raw,s.model,theta)
        errors.append(check_close(back['S'],z['S']))
        errors.append(check_close(sar(sar_solve(x,cfg.alpha),cfg.alpha),x))
        mean,sd_raw=s.phi.parameters_given('S',ctx,raw).chunk(2,1)
        from .distributions import positive, normal_log_prob
        sd=positive(sd_raw,cfg.scale_floor)
        field_mean=s.model.structure_mean(s.model.class_probabilities(z,theta),theta)+cfg.sigma_s*sar_solve(mean,cfg.alpha)
        n=cfg.height*cfg.width
        A=sar(torch.eye(n,dtype=x.dtype).reshape(n,1,cfg.height,cfg.width),cfg.alpha).reshape(n,n).T
        Ainv=torch.linalg.inv(A)
        reference=0
        for c in range(channels):
            cov=cfg.sigma_s**2*Ainv@torch.diag(sd[0,c].flatten().square())@Ainv.T
            reference=reference+MultivariateNormal(field_mean[0,c].flatten(),covariance_matrix=cov).log_prob(z['S'][0,c].flatten())
        transformed=normal_log_prob(raw['S'],mean,sd).sum()+logdet
        errors.append(check_close(transformed,reference))
        wrong=normal_log_prob(raw['S'],mean,sd).sum()
        assert abs(float(wrong-reference))>1
    return dict(max_absolute_error=max(errors), channels=[1,3], rectangular_grid=[4,6],
                independent_dense_mvn=True, inverse_transform=True, omitted_jacobian_rejected=True)


def R06(out):
    s=tiny(local_parameterization='sar_whitened',prior_aware_init=True)
    s.begin_task('one','0'*64);L,_=batches(s);x=L['x'];ref=Refiner(s.model,s.phi)
    def value(detach=False):
        g=torch.Generator().manual_seed(8347)
        theta=s.q.rsample(g);a=s.current.rsample(theta,g)
        ctx,rho=ref.refine(x,theta,a,K=3,training=True,generator=g)
        if detach:rho={k:v.detach() for k,v in rho.items()}
        e,*_=ref.draw(x,theta,a,ctx,rho,s.phi.noise(1,x,g))
        return e.sum()
    params=[s.phi.c_head.bias,s.q.mean,s.q.raw_scale,s.current.m,s.current.C,s.current.raw_scale]
    gradients=torch.autograd.grad(value(),params);rows=[]
    for name,p,grad in zip(('phi','global_mean','global_scale','domain_mean','domain_C','domain_scale'),params,gradients):
        i=int(grad.abs().argmax());old=float(p.detach().flatten()[i]);eps=1e-5
        with torch.no_grad():p.flatten()[i]=old+eps
        plus=float(value())
        with torch.no_grad():p.flatten()[i]=old-eps
        minus=float(value())
        with torch.no_grad():p.flatten()[i]=old
        fd=(plus-minus)/(2*eps);ad=float(grad.flatten()[i]);error=abs(fd-ad)/max(abs(fd),abs(ad),1e-8)
        assert error<=1e-4 or abs(fd-ad)<=1e-6
        rows.append(dict(parameter=name,relative_error=error,absolute_error=abs(fd-ad)))
    broken=torch.autograd.grad(value(True),params)
    delta=max(float((a-b).abs().max()) for a,b in zip(gradients,broken));assert delta>1e-7
    s.finalize_task();before=digest(s.payload());rng=torch.random.get_rng_state().clone()
    Predictor().predict_image_only(x,s,InferenceBudget(S=2,L=4,K=1,repeats=1))
    assert digest(s.payload())==before and torch.equal(rng,torch.random.get_rng_state())
    return dict(gradients=rows, detach_negative_difference=delta, whitened_prediction_state_unchanged=True)


def R07(out):
    g=torch.Generator().manual_seed(SEED)
    ws=[torch.randn(s,3,4,generator=g,dtype=torch.float64)*2 for s in (2,3)]
    ps=[torch.randn(*w.shape,2,1,1,generator=g,dtype=torch.float64).softmax(3) for w in ws]
    actual,diag=combine_slice_banks(ws,ps)
    weights=[];probs=[];domain=[]
    for d,(w,p) in enumerate(zip(ws,ps)):
        for s in range(len(w)):
            for indices in itertools.product(range(4),repeat=3):
                weights.append(sum(w[s,j,i] for j,i in enumerate(indices))-math.log(2*len(w)*4**3))
                probs.append(torch.stack([p[s,j,i] for j,i in enumerate(indices)]));domain.append(d)
    weights=torch.stack(weights);probs=torch.stack(probs);norm=weights.softmax(0)
    expected=torch.einsum('n,nvkhw->vkhw',norm,probs)
    errors=[check_close(actual,expected),check_close(diag['log_evidence'],torch.logsumexp(weights,0))]
    expected_domains=torch.stack([norm[torch.tensor(domain)==d].sum() for d in (0,1)])
    errors.append(check_close(diag['domain_posterior'],expected_domains))
    sw=[];sp=[]
    for w,p in zip(ws,ps):
        w=w.clone();p=p.clone()
        for s in range(len(w)):
            for j in range(3):
                perm=torch.randperm(4,generator=g);w[s,j]=w[s,j,perm];p[s,j]=p[s,j,perm]
        sw.append(w);sp.append(p)
    permuted,_=combine_slice_banks(sw,sp);errors.append(check_close(actual,permuted))
    order=[2,0,1]
    moved,_=combine_slice_banks([w[:,order] for w in ws],[p[:,order] for p in ps])
    errors.append(check_close(actual[order],moved))
    paired,_=combine_slice_banks(ws,ps,'paired')
    paired_permuted,_=combine_slice_banks(sw,sp,'paired')
    assert float((paired-paired_permuted).abs().max())>1e-4
    return dict(max_absolute_error=max(errors), cartesian_reference=True, unequal_outer_counts=[2,3],
                domain_evidence_preserved=True, independent_local_permutation_invariant=True,
                slice_permutation_equivariant=True, paired_permutation_difference=float((paired-paired_permuted).abs().max()))


def R08(out):
    sigma=.7;count=4096
    means=torch.tensor([[-.4,.2,.8],[.7,-.3,.1]],dtype=torch.float64)
    x=torch.tensor([.3,.6,-.2],dtype=torch.float64)
    logz=Normal(means,math.sqrt(1+sigma**2)).log_prob(x).sum(1)
    var=sigma**2/(1+sigma**2);mu=means+(x-means)/(1+sigma**2)
    analytic_class=Normal(torch.zeros_like(mu),torch.ones_like(mu)).cdf(mu/math.sqrt(var))
    expected=(logz.softmax(0)[:,None]*analytic_class).sum(0)
    expected_logz=torch.logsumexp(logz,0)-math.log(2)
    rows=[]
    for repeat in range(4):
        g=torch.Generator().manual_seed(SEED+repeat)
        z=means[:,:,None]+torch.randn(2,3,count,generator=g,dtype=torch.float64)
        w=Normal(z,sigma).log_prob(x[None,:,None])
        p=torch.stack([(z<=0).double(),(z>0).double()],-1)[...,None,None]
        actual,diag=combine_slice_banks([v[None] for v in w],[v[None] for v in p])
        pe=float((actual[:,1,0,0]-expected).abs().max());ze=abs(float(diag['log_evidence']-expected_logz))
        assert pe<=.03 and ze<=.15,(pe,ze)
        assert all(bool(nondegenerate(v,.1,.5).all()) for v in w)
        rows.append(dict(probability_error=pe,log_evidence_error=ze))
    assert abs(float(logz.softmax(0)[0])-.5)>.02
    return dict(repeats=rows, samples_per_slice=count, analytic_domain_posterior=logz.softmax(0).tolist(),
                nonuniform_domain_evidence=True, local_gate_positive_reference=True,
                probability_tolerance=.03, log_evidence_tolerance=.15)


def R09(out):
    from . import qualification
    package=out/'source';package.mkdir()
    (package/'module.py').write_text('VALUE = 1\n')
    with patch.object(qualification,'__file__',str(package/'qualification.py')):
        original=qualification.code_fingerprint()
        (package/'._module.py').write_bytes(b'nonimportable filesystem metadata')
        assert qualification.code_fingerprint()==original
        (package/'._module.py').write_bytes(b'changed filesystem metadata')
        assert qualification.code_fingerprint()==original
        (package/'module.py').write_text('VALUE = 2\n')
        assert qualification.code_fingerprint()!=original
    return dict(nonimportable_metadata_ignored=True, actual_source_change_detected=True, outer_updates=0)


CHECKS=[R01,R02,R03,R04,R05,R06,R07,R08,R09]


def run_checks(ledger,prefix,selected=None):
    results=[]
    for test in CHECKS:
        if selected is not None and test.__name__ not in selected:continue
        start=time.monotonic()
        try:
            with ledger.attempt(f'{prefix}_{test.__name__}',seconds=300.,updates=8 if test is R04 else 0) as out:
                details=test(out)
            status,error='PASS',None
        except Exception as exc:
            status,error,details='FAIL',f'{type(exc).__name__}: {exc}',{}
        result=dict(id=test.__name__,status=status,seed=SEED,seconds=time.monotonic()-start,details=details,error=error)
        results.append(result);print(json.dumps(result),flush=True)
        (ledger.root/f'{prefix}_REVISION_CHECKS.json').write_text(json.dumps(results,indent=2))
    return results

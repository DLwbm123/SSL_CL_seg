"""Preregistered image-only local inference at known synthetic parameters. No outer training."""
from dataclasses import replace
import json
import math
import time
import torch
from .config import Config
from .local_inference import Refiner
from .predictive import combine_slice_banks, ess, nondegenerate
from .stream_state import StreamState, digest
from .synthetic import generated_data, scores, to_json


def true_theta(state):
    theta = torch.zeros_like(state.q.mean)
    w = state.q.unpack(theta)
    w['g.c.weight'][0,0] = .1
    w['g.r.weight'][0,0,1,1] = 1.
    w['g.out.weight'][0,0,1,1] = 6.
    w['f.in.weight'][0,0,1,1] = -1.
    w['f.in.weight'][0,1,1,1] = 1.
    w['f.out.weight'][0,0,1,1] = 1.
    return theta


def local_bank(state, x, theta, a, K, seed, counter, draws=64):
    """No labels or true local latent variables can enter this function."""
    ref = Refiner(state.model, state.phi, counter)
    weights, probabilities, decomposition, gradient_norms = [], [], [], []
    for j, image in enumerate(x.split(1)):
        fit = torch.Generator().manual_seed(seed + j * 100003)
        importance = torch.Generator().manual_seed(seed + j * 100003 + 1000003)
        ctx, rho = ref.refine(image, theta, a, K=K, training=False, generator=fit)
        ws, ps, parts = [], [], []
        for _ in range(draws):
            noise = state.phi.noise(1, image, importance)
            logw, z, density, logq = ref.draw(image, theta, a, ctx, rho, noise)
            ws.append(logw[0].detach())
            ps.append(state.model.class_probabilities(z, theta)[0].detach())
            parts.append({**{k: v[0].detach() for k,v in density.items()}, 'logq':logq[0].detach(), 'ELBO':logw[0].detach()})
        weights.append(torch.stack(ws)); probabilities.append(torch.stack(ps))
        decomposition.append({k:float(torch.stack([p[k] for p in parts]).mean()) for k in parts[0]})
        diagnostic = torch.Generator().manual_seed(seed + j * 100003 + 2000003)
        value, *_ = ref.draw(image, theta, a, ctx, rho, state.phi.noise(1,image,diagnostic))
        grads = torch.autograd.grad(value.sum(), tuple(rho.values()))
        counter.add('diagnostic_local_backward_calls')
        gradient_norms.append({k:float(g.norm()) for k,g in zip(rho,grads)})
    return torch.stack(weights)[None], torch.stack(probabilities)[None], decomposition, gradient_norms


def evaluate_fixed(state, x, theta, a, K, counter):
    """Predictions are sealed before the caller accesses scoring labels."""
    before = digest(state.payload())
    repeats = []
    for r in range(4):
        seed = 17411 + r * 10000319
        w, p, decomposition, gradients = local_bank(state, x, theta, a, K, seed, counter)
        result = dict(repeat=r, seed=seed,
                      local_ess=ess(w).tolist(), local_max_weight=w.softmax(-1).max(-1).values.tolist(),
                      local_gate=bool(nondegenerate(w,.1,.5).all()),
                      fixed_noise_ELBO_parts=decomposition, local_offset_gradient_norms=gradients)
        for integrator in ('paired', 'factorized'):
            prob, diag = combine_slice_banks([w], [p], integrator)
            result[integrator] = dict(probability=prob, log_evidence=float(diag['log_evidence']),
                                      paired_joint_ess=float(ess(w.sum(1))[0]) if integrator=='paired' else None)
        repeats.append(result)
    assert digest(state.payload()) == before
    summary = {}
    for integrator in ('paired','factorized'):
        ps = torch.stack([r[integrator]['probability'] for r in repeats])
        z = [r[integrator]['log_evidence'] for r in repeats]
        spread = float((ps.max(0).values-ps.min(0).values).max())
        zspread = max(z)-min(z)
        passed = all(r['local_gate'] for r in repeats) and spread<=.1 and zspread<=1.
        summary[integrator] = dict(inference_status='ORACLE_INFERENCE_PASS' if passed else 'INFERENCE_UNQUALIFIED',
                                   probability_range=spread, log_evidence_range=zspread,
                                   local_gate=all(r['local_gate'] for r in repeats), global_ess='NOT_APPLICABLE_FIXED_PARAMETERS')
    return repeats, summary


def run(ledger,counter,prefix):
    variants=[('diagonal_original',Config()),
              ('diagonal_scale_init',replace(Config(),prior_aware_init=True)),
              ('sar_whitened_scale_init',replace(Config(),prior_aware_init=True,local_parameterization='sar_whitened'))]
    results=[]
    for name,cfg in variants:
        for K in (0,3,30):
            started=time.monotonic()
            with ledger.attempt(f'{prefix}_{name}_K{K}',seconds=300.,updates=0) as out:
                state=StreamState(cfg)
                state.q.requires_grad_(False);state.phi.requires_grad_(False)
                theta=true_theta(state)
                domain_results=[];private=[]
                for domain in (0,1):
                    x,y,a=generated_data(cfg,domain,2,cfg.seed+999)
                    repetitions,summary=evaluate_fixed(state,x,theta,a,K,counter)
                    private.append(repetitions)
                    for integrator in ('paired','factorized'):
                        # Every repeat is scored; repeat 0 is the preregistered reported prediction.
                        values=[scores(r[integrator]['probability'],y) for r in repetitions]
                        main=values[0]
                        qualified=(summary[integrator]['inference_status']=='ORACLE_INFERENCE_PASS'
                                   and main['NLL']<math.log(2) and .02<=main['predicted_foreground_fraction']<=.98)
                        summary[integrator].update(metrics=main,repeat_metrics=values,
                            scientific_state='ORACLE_LOCAL_QUALIFIED' if qualified else 'ORACLE_LOCAL_UNQUALIFIED')
                    public_repeats=[]
                    for r in repetitions:
                        public_repeats.append({k:v for k,v in r.items() if k not in ('paired','factorized')})
                    constant=torch.full((len(y),2,*y.shape[-2:]),.5,dtype=torch.float64)
                    foreground=torch.zeros_like(constant);foreground[:,1]=1
                    background=torch.zeros_like(constant);background[:,0]=1
                    domain_results.append(dict(generated_domain=domain,results=summary,repeats=public_repeats,
                        trivial_baselines={'constant_half':scores(constant,y),'all_foreground':scores(foreground,y),
                                           'all_background':scores(background,y)}))
                torch.save(private,out/'PRIVATE_PREDICTIONS.pt')
                result=dict(variant=name,K=K,L=64,repeats=4,oracle=True,outer_updates=0,real_data_access=0,
                            seconds=time.monotonic()-started,domains=domain_results,
                            formal_qualification='NOT_QUALIFIED',state_unchanged=True,
                            same_local_bank_for_integrators=True)
                (out/'FIXED_PARAMETER_RESULTS.json').write_text(json.dumps(result,indent=2))
                results.append(result)
                print(json.dumps({'variant':name,'K':K,'states':[d['results']['factorized']['scientific_state'] for d in domain_results]}),flush=True)
            (ledger.root/f'{prefix}_FIXED_PARAMETER_RESULTS.json').write_text(json.dumps(results,indent=2))
    target=next(r for r in results if r['variant']=='sar_whitened_scale_init' and r['K']==30)
    passed=all(d['results']['factorized']['scientific_state']=='ORACLE_LOCAL_QUALIFIED' for d in target['domains'])
    return dict(execution='COMPLETE',scientific_state='ORACLE_LOCAL_QUALIFIED' if passed else 'ORACLE_LOCAL_UNQUALIFIED',
                oracle=True,formal_qualification='NOT_QUALIFIED',outer_updates=0,real_data_access=0,comparisons=results)

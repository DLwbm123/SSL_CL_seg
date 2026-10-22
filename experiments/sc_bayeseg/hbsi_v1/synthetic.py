import hashlib
import json
import math
import time
import torch
from .config import Config
from .data_adapter import leaf_observations
from .diagnostics import parameter_audit
from .local_inference import Refiner
from .objective import SequentialObjective
from .predictive import Predictor, InferenceBudget
from .stream_state import StreamState, digest


def to_json(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
    if isinstance(value,dict):return {k:to_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [to_json(v) for v in value]
    return value


def generated_data(config, domain, n, seed):
    """Sample the declared joint model with fixed true G/F and changed appearance a only."""
    s=StreamState(config);g=torch.Generator().manual_seed(seed);c=config
    theta=torch.zeros_like(s.q.mean);w=s.q.unpack(theta)
    w['g.c.weight'][0,0]=.1; w['g.r.weight'][0,0,1,1]=1.
    w['g.out.weight'][0,0,1,1]=6.
    w['f.in.weight'][0,0,1,1]=-1.;w['f.in.weight'][0,1,1,1]=1.
    w['f.out.weight'][0,0,1,1]=1.
    z={'c':torch.randn(n,c.c_dim,generator=g,dtype=torch.float64),
       'R':torch.randn(n,c.r_channels,c.r_height,c.r_width,generator=g,dtype=torch.float64),
       'b':torch.randn(n,c.channels,c.b_dim,generator=g,dtype=torch.float64)*c.tau_b}
    p=s.model.class_probabilities(z,theta)
    mean=s.model.structure_mean(p,theta)
    u=torch.arange(c.height,dtype=torch.float64);v=torch.arange(c.width,dtype=torch.float64)
    eig=1-c.alpha/2*(torch.cos(2*math.pi*u/c.height)[:,None]+torch.cos(2*math.pi*v/c.width)[None,:])
    noise=torch.randn(mean.shape,generator=g,dtype=torch.float64)
    z['S']=mean+c.sigma_s*torch.fft.ifft2(torch.fft.fft2(noise)/eig).real
    # Student-t df=4 = normal / sqrt(sum of four squared normals / 4).
    e=torch.randn(mean.shape,generator=g,dtype=torch.float64)
    chi=torch.randn((4,*mean.shape),generator=g,dtype=torch.float64).square().sum(0)
    a=mean.new_tensor([0.,-.2,-2.] if domain==0 else [.15,.4,-2.])
    gamma,beta,xi=a
    x=gamma.exp()*z['S']+beta+torch.einsum('bck,khw->bchw',z['b'],s.model.basis)+xi.exp()*e/torch.sqrt(chi/4)
    labels=torch.multinomial(p.permute(0,2,3,1).reshape(-1,c.classes),1,generator=g).reshape(n,c.height,c.width)
    return x.detach(),labels,a


def scores(p,y):
    prediction=p.argmax(1); per=[]
    for i in range(len(p)):
        a=prediction[i]==1;b=y[i]==1
        den=int(a.sum()+b.sum());per.append(2*int((a&b).sum())/den if den else None)
    nll=-p.gather(1,y[:,None]).clamp_min(1e-30).log().mean()
    truth=torch.nn.functional.one_hot(y,p.shape[1]).permute(0,3,1,2)
    return {'foreground_Dice_per_image':per,'mean_Dice':sum(v for v in per if v is not None)/sum(v is not None for v in per),
            'NLL':float(nll),'Brier':float((p-truth).square().sum(1).mean()),
            'predicted_foreground_fraction':float((prediction==1).double().mean())}


def run(config, ledger, counter, run_id):
    if config!=Config():
        raise ValueError('qualification configuration is frozen to configs/synthetic.yaml')
    with ledger.attempt(run_id+'_synthetic',seconds=min(1200.,ledger.limits['seconds']-ledger.data['seconds_charged']),updates=80) as out:
        state=StreamState(config); reports=[]; predictions=[]
        start=time.monotonic()
        for domain in range(2):
            stage=f'synthetic{domain+1}'
            x,y,a=generated_data(config,domain,8,config.seed+11)
            Lx,Ly=x[:4],y[:4];Ux=x[4:]
            del x,y  # U labels are not retained or provided to a loader.
            manifest=hashlib.sha256(json.dumps({'stage':stage,'seed':config.seed+11,'N_L':4,'N_U':4}).encode()).hexdigest()
            state.begin_task(stage,manifest)
            (out/f'PARAMETER_AUDIT_stage{domain+1}.json').write_text(json.dumps(parameter_audit(state),indent=2))
            objective=SequentialObjective(state,Refiner(state.model,state.phi,counter))
            before=digest(state.prior.state_dict());history=[]
            old={k:digest(v.state_dict()) for k,v in state.memories.items()}
            for step in range(40):
                ledger.check()
                li=torch.randint(4,(1,),generator=state.generator);ui=torch.randint(4,(1,),generator=state.generator)
                L={'stage':stage,'role':'trainL','x':Lx[li],'observations':leaf_observations(Ly[li],config.classes)}
                U={'stage':stage,'role':'trainU','x':Ux[ui]}
                loss,parts=objective.estimate(L,U,{'L':4,'U':4},state.generator)
                state.optimizer.zero_grad();loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for group in state.optimizer.param_groups for p in group['params']):
                    raise FloatingPointError('outer gradient')
                state.optimizer.step();state.steps+=1
                for k in ('global_updates','domain_updates','phi_updates','outer_backward_calls'):counter.add(k)
                counter.add('current_train_images',2)
                history.append(to_json(parts))
            assert before==digest(state.prior.state_dict())
            assert all(old[k]==digest(state.memories[k].state_dict()) for k in old)
            receipt=state.finalize_task(counter.export())
            state.save(out/f'stage{domain+1}.pt')
            reports.append({'stage':stage,'steps':40,'appearance':a.tolist(),'objective_initial':history[0],
                            'objective_final':history[-1],'prior_fixed':True,'old_conditional_parameters_fixed':True,'receipt':receipt})
            (out/f'objective_stage{domain+1}.json').write_text(json.dumps(history,indent=2))
            del L,U,Lx,Ly,Ux
            # Independent generated evaluation observations; trainer cannot load them.
            for eval_domain in range(domain+1):
                ex,ey,_=generated_data(config,eval_domain,2,config.seed+999)
                for K in (0,3):
                    result=Predictor(counter).predict_image_only(ex,state,InferenceBudget(K=K,seed=17411))
                    p=result.pop('probability')
                    predictions.append({'after_stage':domain+1,'generated_domain':eval_domain+1,'K':K,
                                        'metrics':scores(p,ey),'inference':to_json(result),'image_only':True,'true_domain_used_by_predictor':False})
            print(json.dumps({'stage':domain+1,'steps':40,'last_objective':history[-1]['unscaled']}),flush=True)
        summary={'schema':1,'status':'INFERENCE_UNQUALIFIED','stages':reports,'predictions':predictions,
                 'seconds':time.monotonic()-start,'outer_updates':80,'real_data_access':0,
                 'optimization_gate':{'mean_Dice_each_domain_min':.65,'foreground_fraction_min':.02,'foreground_fraction_max':.98},
                 'analytic_reference':'T07, independent from neural VI'}
        if all(p['inference']['status']=='SYNTHETIC_INFERENCE_PASS' for p in predictions):
            summary['status']='SYNTHETIC_EXECUTION_COMPLETE_INFERENCE_PASS'
        final=[p for p in predictions if p['after_stage']==2 and p['K']==3]
        summary['optimization_status']='PASS' if all(p['metrics']['mean_Dice']>=.65 and .02<=p['metrics']['predicted_foreground_fraction']<=.98 for p in final) else 'OPTIMIZATION_FAIL'
        (out/'SYNTHETIC_RESULTS.json').write_text(json.dumps(summary,indent=2))
        (out/'DATA_BOUNDARY_AUDIT.json').write_text(json.dumps({'synthetic_only':True,'historical_training_access':0,
            'train_U_labels_retained':False,'real_validation_access':0,'sealed_test_access':0,'synthetic_evaluation_separate_seeds':True},indent=2))
        (out/'PARAMETER_AUDIT.json').write_text(json.dumps(parameter_audit(state),indent=2))
        return summary

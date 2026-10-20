"""One bounded convex bias-coordinate closure, isolated from patient training."""
import copy
import json
import os
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from lcrseg.data import H5LabeledDataset,collate_labeled
from torch.nn import functional as F
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.r_start import cpu,write,append,save,draw_g,local_sample,metrics
from sc_bayeseg.window_v1.same_family_oracle import posterior
from sc_bayeseg.window_v1.spatial_residual import SpatialResidualModel
from sc_bayeseg.window_v1.fullview import flip
from sc_bayeseg.window_v1.common import storage

SEEDS=(1720163,1730163,1740163,1750163)


def global_kl64(model):
    value=0.
    for layer in model.bayes_layers():
        for part in ('w','b'):
            mu=getattr(layer,'mean_'+part).double();ls=getattr(layer,'logsd_'+part).double()
            pm=getattr(layer,'prior_mean_'+part).double();ps=getattr(layer,'prior_logsd_'+part).double()
            value+=float((ps-ls+.5*((mu-pm).square()*(-2*ps).exp()+(2*(ls-ps)).exp()-1)).sum())
    return value


def bias_kl(delta,mu,ls,pm,ps):
    return (ps-ls+.5*((mu+delta-pm).square()*(-2*ps).exp()+(2*(ls-ps)).exp()-1)).sum()


def solve(cache,y,mu,ls,pm,ps,constant,active,out,costs,identity,deadline):
    shape=cache.shape;total=int(np.prod(shape[:-1]));target=y.flatten();delta=torch.zeros(3,device='cuda',dtype=torch.float64)
    indices=torch.tensor(active,device='cuda');history=[];status='BUDGET_EXHAUSTED'
    def evaluate(d,derivatives):
        assert time.time()<deadline
        costs['derivative_passes' if derivatives else 'objective_only_passes']+=1
        obj=torch.zeros((),device='cuda',dtype=torch.float64)
        g=torch.zeros(3,device='cuda',dtype=torch.float64);h=torch.zeros((3,3),device='cuda',dtype=torch.float64)
        padded=torch.cat([torch.zeros(1,device='cuda',dtype=torch.float64),d])
        for start in range(0,shape[0],4):
            block=torch.from_numpy(np.array(cache[start:start+4],copy=True)).to('cuda',dtype=torch.float64).reshape(-1,4)
            count=len(block)//len(target);labels=target.repeat(count);logits=block+padded
            obj+=(logits.logsumexp(1)-logits.gather(1,labels[:,None])[:,0]).sum()/total
            if derivatives:
                pf=logits.softmax(1)[:,1:];g+=pf.sum(0)/total
                h+=(torch.diag(pf.sum(0))-pf.T@pf)/total
        if derivatives:
            g-=torch.bincount(target,minlength=4)[1:].double()/len(target)
            precision=(-2*ps).exp()/(2*128**2)
            g+=(mu+d-pm)*precision;h+=torch.diag(precision)
        obj+=bias_kl(d,mu,ls,pm,ps)/(2*128**2)+constant
        return float(obj),g,h
    for iteration in range(51):
        obj,g,h=evaluate(delta,True);ga=g[indices];ha=h[indices[:,None],indices[None,:]]
        residual=float(ga.abs().max());eigen=float(torch.linalg.eigvalsh(ha).min());assert eigen>0
        row=dict(iteration=iteration,objective=obj,active_gradient_inf=residual,min_active_Hessian_eigenvalue=eigen,delta=delta.cpu().tolist())
        history.append(row);append(out/'derivatives.jsonl',dict(identity=identity,**row))
        if residual<=1e-7:status='CONVERGED';break
        if iteration==50:break
        direction=torch.zeros_like(delta);direction[indices]=-torch.linalg.solve(ha,ga)
        slope=float(g@direction);assert slope<0
        accepted=False
        for j in range(8):
            alpha=2.**(-j);candidate=delta+alpha*direction;value,_,_=evaluate(candidate,False)
            costs['backtracking_candidates']+=1
            append(out/'candidates.jsonl',dict(identity=identity,iteration=iteration,backtrack=j,objective=value,alpha=alpha))
            if value<=obj+1e-4*alpha*slope:
                delta=candidate;costs['newton_acceptances']+=1
                append(out/'physical_updates.jsonl',dict(index=costs['newton_acceptances'],identity=identity,iteration=iteration+1,delta=delta.cpu().tolist(),objective=value))
                accepted=True;break
        if not accepted:status='LINE_SEARCH_FAILED';break
    return delta,dict(status=status,accepted_updates=len(history)-1 if status!='LINE_SEARCH_FAILED' else iteration,final=history[-1],history=history)


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);started=time.time()
    costs=dict(cache_G_forward=0,cache_Theta=0,evaluation_G_forward=0,evaluation_Theta=0,engineering_G_forward=0,engineering_Theta=0,
      newton_acceptances=0,backtracking_candidates=0,derivative_passes=0,objective_only_passes=0,Adam_optimizer=0,network_reverse=0)
    try:
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        write(root/'CONTEXT.json',s);storage(s,reserve=2_000_000_000)
        sourcefile=Path(s['checkpoint']);payload=torch.load(sourcefile,map_location='cpu',weights_only=False);assert payload['cursor']==8675
        model=SpatialResidualModel().cuda().requires_grad_(False);model.load_state_dict(payload['model']);model.eval();frozen=cpu(model.state_dict())
        files=[sourcefile]+[Path(s['allocation_source'])/(a+'_T2L2')/'S0500.pt' for a in ('ZERO','R1_WARM')]
        meta={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in files}
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text())
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        b=collate_labeled([ds[i] for i in sel['train_indices']]);y=F.interpolate(b.label[:,None].float(),(128,128),mode='nearest')[:,0].long();y=torch.cat([flip(y,k) for k in range(4)]).cuda()
        layer=model.anatomy[2];mu=layer.mean_b.double().clone();ls=layer.logsd_b.double();pm=layer.prior_mean_b.double();ps=layer.prior_logsd_b.double()
        kl0=global_kl64(model);bk0=float(bias_kl(torch.zeros_like(mu),mu,ls,pm,ps));arms={};affine_errors=[]
        with torch.no_grad():
            for arm,file in zip(('ZERO','R1_WARM'),files[1:]):
                out=root/arm;out.mkdir(exist_ok=False);state=torch.load(file,map_location='cpu',weights_only=False);assert state['source_step']==1500
                q=posterior(tuple(v.cuda() for v in state['raw']))
                localKL=0.
                for mean,sd in ((q[0],q[1]),(q[2],q[3])):
                    localKL+=float(.5*(mean.double().square()+(2*sd.double()).exp()-1-2*sd.double()).flatten(1).sum(1).mean()/16384)
                constant=localKL+(kl0-bk0)/(2*16384)
                cache=np.lib.format.open_memmap(out/'TRAIN_LOGITS.npy',mode='w+',dtype=np.float32,shape=(256,8,128,128,4))
                tg=torch.Generator(device='cuda').manual_seed(1620163);lg=torch.Generator(device='cuda').manual_seed(2620163);draw=0;first=None
                for theta in range(32):
                    assert time.time()-s['original_started_unix']<85500
                    before=tg.get_state();draw_g(model,tg);costs['cache_Theta']+=1
                    for local in range(8):
                        c,r=local_sample(q,lg);logits=model.spatial_logits(c,r)
                        if first is None:first=(c.clone(),r.clone(),logits.clone(),before.clone())
                        costs['cache_G_forward']+=1;cache[draw]=logits.permute(0,2,3,1).cpu().numpy();draw+=1
                    write(root/'STATUS.json',dict(status='CACHING',arm=arm,theta=theta,costs=costs))
                cache.flush();assert draw==256
                def verify(delta):
                    original=layer.mean_b.clone();layer.mean_b.copy_((mu+delta).float())
                    gt=torch.Generator(device='cuda');gt.set_state(first[3]);draw_g(model,gt)
                    actual=model.spatial_logits(first[0],first[1]);expected=first[2].double()+torch.cat([torch.zeros(1,device='cuda'),delta]).reshape(1,4,1,1)
                    costs['engineering_G_forward']+=1;costs['engineering_Theta']+=1
                    error=float((actual.double()-expected).abs().max());affine_errors.append(error)
                    layer.mean_b.copy_(original);model.clear_draw();assert error<=2e-6
                verify(torch.tensor([0.,0.,.3],device='cuda',dtype=torch.float64));verify(torch.tensor([.1,-.2,.3],device='cuda',dtype=torch.float64))
                solutions={'G0':torch.zeros(3,device='cuda',dtype=torch.float64)};solvers={}
                for name,active in (('G1',[2]),('G3',[0,1,2])):
                    delta,receipt=solve(cache,y,mu,ls,pm,ps,constant,active,out,costs,arm+'/'+name,s['original_started_unix']+85500)
                    solutions[name]=delta;solvers[name]=receipt;verify(delta)
                    new=copy.deepcopy(payload['model']);new['anatomy.2.mean_b']=(mu+delta).float().cpu()
                    save(out/(name+'_MODEL.pt'),dict(model=new,delta=delta.cpu(),solver=receipt,source_cursor=8675))
                    write(root/'STATUS.json',dict(status='SOLVED',arm=arm,name=name,solver_status=receipt['status'],costs=costs))
                del cache
                probabilities={k:[] for k in solutions};logprobs={k:[] for k in solutions}
                for seed in SEEDS:
                    tg=torch.Generator(device='cuda').manual_seed(seed);lg=torch.Generator(device='cuda').manual_seed(seed+1000000)
                    pacc={k:torch.zeros((8,4,128,128),device='cuda',dtype=torch.float64) for k in solutions};lacc={k:v.clone() for k,v in pacc.items()}
                    for _ in range(64):
                        assert time.time()-s['original_started_unix']<85500
                        draw_g(model,tg);costs['evaluation_Theta']+=1
                        for _ in range(8):
                            c,r=local_sample(q,lg);logits=model.spatial_logits(c,r).double();costs['evaluation_G_forward']+=1
                            for name,delta in solutions.items():
                                shifted=logits+torch.cat([torch.zeros(1,device='cuda'),delta]).reshape(1,4,1,1)
                                lp=shifted.log_softmax(1);pacc[name]+=lp.exp();lacc[name]+=lp
                    for name in solutions:probabilities[name].append((pacc[name]/512).cpu());logprobs[name].append((lacc[name]/512).cpu())
                    write(root/'STATUS.json',dict(status='EVALUATING',arm=arm,seed=seed,costs=costs))
                output={}
                for name,delta in solutions.items():
                    save(out/(name+'_PRED.pt'),dict(flows=probabilities[name],mean_logprob_flows=logprobs[name],seeds=list(SEEDS)))
                    p=torch.stack(probabilities[name]);main=metrics(p.mean(0),y.cpu());per=[metrics(v,y.cpu()) for v in p]
                    expected=[float(-v.gather(1,y.cpu()[:,None]).mean()) for v in logprobs[name]]
                    bk=float(bias_kl(delta,mu,ls,pm,ps));J=sum(expected)/4+bk/(2*16384)+constant
                    output[name]=dict(metrics=main,flow_metrics=per,expected_CE_flows=expected,bias_KL=bk,constant_C=constant,coordinate_J=J,
                      delta=delta.cpu().tolist(),robust_qualified24=main['qualified_24'] and all(v['qualified_24'] for v in per))
                delta=solutions['G3'].cpu();weight=torch.cat([torch.zeros(1,dtype=torch.float64),delta]).exp().reshape(1,4,1,1)
                post=[]
                for p in probabilities['G0']:
                    v=p*weight;post.append(v/v.sum(1,keepdim=True))
                pooled=torch.stack(probabilities['G0']).mean(0)*weight;pooled/=pooled.sum(1,keepdim=True)
                output['POST_G3_SAME_DELTA']=dict(metrics=metrics(pooled,y.cpu()),flow_metrics=[metrics(v,y.cpu()) for v in post],not_separately_optimized=True)
                arms[arm]=dict(solvers=solvers,readouts=output,local_KL_per_pixel=localKL)
                assert same(frozen,model.state_dict());model.clear_draw();torch.cuda.empty_cache()
        assert costs['cache_G_forward']==512 and costs['evaluation_G_forward']==4096 and costs['engineering_G_forward']==8
        assert costs['newton_acceptances']<=200 and costs['backtracking_candidates']<=1600 and costs['derivative_passes']<=204
        assert same(frozen,model.state_dict()) and all((p.stat().st_size,p.stat().st_mtime_ns)==st for p,st in meta.items())
        write(root/'RESULTS.json',dict(status='COMPLETE',arms=arms,costs=costs,affine_max_errors=affine_errors,source_frozen=True,
          patient_mainline_unchanged=True,phase_elapsed_seconds=time.time()-started,original_elapsed_seconds=time.time()-s['original_started_unix'],space=storage(s)))
        write(root/'STATUS.json',dict(status='COMPLETE',costs=costs));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise

if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))

"""Fixed posterior sampling factorial and relaxed nonnegative output-cone audit."""
import json,math,os,traceback,itertools
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scipy.optimize import linprog
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.engine.trainer import seed_everything
from lcrseg.engine.checkpoint import capture_rng_state,restore_rng_state
from care_seg_study_v1.run import write,append
from sc_bayeseg.run import same
from sc_bayeseg.diagnostic_v1.run import dice
from sc_bayeseg.window_v1.spatial_residual import SpatialResidualModel
from sc_bayeseg.window_v1.fullview import flip
from sc_bayeseg.window_v1.common import storage


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);costs=dict(forward=0,proposal=0,theta_draw=0,residual_noise=0,LP=0,reverse=0,optimizer=0,local_updates=0)
    def forbid(*a,**kw):raise AssertionError('optimizer forbidden')
    torch.optim.Optimizer.__init__=forbid;torch.optim.Adam.step=forbid
    try:
        storage(s);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        src=Path(s['discrete_train']);controls=Path(s['controls']);vroot=Path(s['spatial_residual']);prior=json.loads((controls/'RESULTS.json').read_text())
        protected=[src/'physical_updates.jsonl',controls/'local_updates.jsonl',vroot/'local_updates.jsonl'];texts=[p.read_text() for p in protected]
        payload=torch.load(src/'S0600.pt',map_location='cpu',weights_only=False);assert payload['cursor']==8075
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text());ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in sel['train_indices']]).to('cuda');im=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False);gt=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long();image=torch.cat([flip(im,k) for k in range(4)]);y=torch.cat([flip(gt,k) for k in range(4)])
        model=SpatialResidualModel().cuda();model.load_state_dict(payload['model']);model.requires_grad_(False);rng=capture_rng_state();states={};shapley={}
        with torch.no_grad():hmean,_=model.labeled_proposal(image,y,sample=False);cmean=hmean['c'];costs['proposal']+=1
        for name in ('M','S','V'):
            path=(vroot if name=='V' else controls/name)/'S0500.pt';qcpu=torch.load(path,map_location='cpu',weights_only=False)['q'];q=tuple(t.cuda() for t in qcpu)
            reference=prior['reused_V'] if name=='V' else prior['arms'][name]['final'];cells={}
            for seed in (50163,80163):
                flow={}
                for mask in range(8):
                    seed_everything(seed);logps=[];ce=[];class_nll=[]
                    with torch.no_grad():
                        for _ in range(2):
                            model.draw();costs['theta_draw']+=1
                            if not (mask&1):
                                for layer in model.bayes_layers():layer.sampled=(layer.mean_w,layer.mean_b)
                            try:
                                for _ in range(4):
                                    h,_=model.labeled_proposal(image,y);costs['proposal']+=1
                                    residual=q[0]+q[1].exp()*torch.randn_like(q[0]);costs['residual_noise']+=1
                                    logits=model.spatial_logits(h['c'] if mask&2 else cmean,residual if mask&4 else q[0]);lp=logits.log_softmax(1);logps.append(lp)
                                    nll=F.cross_entropy(logits,y,reduction='none');ce.append(nll.sum((1,2))/128**2);class_nll.append([float(nll[y==k].mean()) for k in range(4)])
                                    costs['forward']+=1;append(root/'forward_calls.jsonl',dict(index=costs['forward'],state=name,seed=seed,mask=mask))
                            finally:model.clear_draw()
                        per_view=torch.stack(ce).mean(0);mix=torch.logsumexp(torch.stack(logps),0)-math.log(8)
                        mix_ce=-mix.gather(1,y[:,None]).squeeze(1).mean();pred=mix.argmax(1)
                        row=dict(expected_CE=float(per_view.mean()),per_view_expected_CE=per_view.tolist(),class_expected_NLL=np.asarray(class_nll).mean(0).tolist(),probability_average_CE=float(mix_ce),probability_average_Dice=dice(pred,y),probability_average_entropy=float(-(mix.exp()*mix).sum(1).mean()))
                        if mask==0:assert abs(row['expected_CE']-reference['conditional_mean_CE'])<1e-6
                        if mask==7:
                            ref=next(f for f in reference['flows'] if f['seed']==seed);assert max(abs(a+b) for a,b in zip(row['per_view_expected_CE'],ref['parts']['categories']))<1e-6
                        flow[str(mask)]=row
                cells[str(seed)]=flow;values={int(k):v['expected_CE'] for k,v in flow.items()};allocation=[]
                for i in range(3):
                    total=0.
                    for mask in range(8):
                        if mask&(1<<i):continue
                        n=mask.bit_count();weight=math.factorial(n)*math.factorial(2-n)/6
                        total+=weight*(values[mask|(1<<i)]-values[mask])
                    allocation.append(total)
                gap=values[7]-values[0];assert abs(sum(allocation)-gap)<1e-10
                shapley[name+'_'+str(seed)]=dict(order=['Theta','c','R'],CE_contribution=allocation,all_sample_minus_all_mean=gap)
            assert same(qcpu,q);states[name]=cells;write(root/'STATUS.json',dict(status='RUNNING',state=name,costs=costs,space=storage(s)))
        cones={}
        for source,sd in [('proper_L600',payload['model']),('diagnostic_recon_code600',torch.load(Path(s['discrete_recon'])/'code/S0600.pt',map_location='cpu',weights_only=False)['model'])]:
            w=sd['anatomy.2.mean_w'].double().reshape(3,-1).numpy();b=sd['anatomy.2.mean_b'].double().numpy();w=np.vstack([np.zeros((1,w.shape[1])),w]);b=np.r_[0,b];rows=[]
            for k in range(4):
                others=[j for j in range(4) if j!=k];A=np.stack([w[j]-w[k] for j in others]);rhs=np.array([b[k]-b[j]-1 for j in others])
                sol=linprog(np.ones(w.shape[1]),A_ub=A,b_ub=rhs,bounds=(0,None),method='highs');record=dict(target=k,status=int(sol.status),numerically_feasible=bool(sol.success))
                if sol.success:
                    margins=np.array([(w[k]-w[j])@sol.x+b[k]-b[j] for j in others]);assert sol.x.min()>=-1e-6 and margins.min()>=1-1e-6
                    record.update(min_margin=float(margins.min()),input_L1=float(sol.x.sum()),input_L2=float(np.linalg.norm(sol.x)),independent_certificate=True)
                costs['LP']+=1;append(root/'LP_calls.jsonl',dict(index=costs['LP'],source=source,target=k));rows.append(record)
            cones[source]=rows
        restore_rng_state(rng);assert same(rng,capture_rng_state()) and same(payload['model'],model.state_dict()) and all(v.grad is None for v in model.parameters())
        assert all(p.read_text()==t for p,t in zip(protected,texts))
        assert costs==dict(forward=384,proposal=385,theta_draw=96,residual_noise=384,LP=8,reverse=0,optimizer=0,local_updates=0)
        write(root/'RESULTS.json',dict(status='COMPLETE',costs=costs,states=states,Shapley=shapley,output_cones=cones,all_mean_and_all_sample_parity=True,model_RNG_local_states_source_ledgers_unchanged=True,no_heldout_or_U=True,space=storage(s)));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))

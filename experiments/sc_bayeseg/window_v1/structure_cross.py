"""Complete structure cross-readout and init1 with identical old numeric starts.

Uses the frozen structure-only sampling, density and acceptance definitions.
No global model/optimizer update is performed.
"""
import copy, json, math, os, statistics, traceback
from pathlib import Path
from dataclasses import replace
import torch
from torch.nn import functional as F
from lcrseg.contracts import MethodStepOutput
from lcrseg.data import H5LabeledDataset, collate_labeled
from lcrseg.engine.trainer import Trainer, TrainerState
from lcrseg.engine.checkpoint import capture_rng_state, restore_rng_state
from care_seg_study_v1.run import write, append, save
from sc_bayeseg.method import Method
from sc_bayeseg.run import snapshot, restore, same
from sc_bayeseg.model import sar, student_log_prob, normal_log_prob
from sc_bayeseg.window_v1.spatial_residual import SpatialResidualModel
from sc_bayeseg.window_v1.factorized_p1 import entropy, self_check
from sc_bayeseg.window_v1.factorized_p2 import SEEDS, contrast, task_key
from sc_bayeseg.window_v1.common import storage


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False)
    costs=dict(optimizer=0,main=0,golden=0,local_reverse=0,candidate=0,accepted=0,density=0,theta_draw=0);active={};hooks=[]
    try:
        storage(s);check=self_check();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        src=Path(s['spatial_generator']);p1=Path(s['factorized_p1']);p2=Path(s['factorized_p2'])
        history={p:p.read_text() for p in (src/'physical_updates.jsonl',p1/'RESULTS.json',p2/'RESULTS.json',p2/'reverse_calls.jsonl')}
        control=json.loads(history[p2/'RESULTS.json']);assert control['status']=='COMPLETE' and control['costs']['reverse']==1200
        payload=torch.load(src/'S0600.pt',map_location='cpu',weights_only=False);base=payload['cursor'];assert base==8675
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text());assert len(sel['train_indices'])==2
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in sel['train_indices']]).to('cuda');image=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False);gt=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long();N=128**2
        batch=replace(lb,image=image,label=gt,valid_mask=torch.ones((2,1,128,128),device='cuda',dtype=torch.bool))
        labels=dict(GT=gt,BG=torch.zeros_like(gt),CROSS=gt.flip(0),SHIFT=torch.roll(gt,(32,32),(-2,-1)),DELETE2=torch.where(gt==2,0,gt));references={}
        def loadq(path):
            state=torch.load(path,map_location='cpu',weights_only=False);references[path]=state['q'];return tuple(v.to('cuda') for v in state['q'])
        tasks=[]
        for init in (0,1):
            for name in labels:
                path=(p1/f'B_init{init}_{name}' if name in ('GT','BG') else p2/f'B_init{init}_{name}')/'S0200.pt'
                tasks.append(dict(key=task_key('B',init,name,200),init=init,name=name,q=loadq(path)))
        intervention=Path(s['structure_only']);prior_result=json.loads((intervention/'RESULTS.json').read_text());assert prior_result['status']=='COMPLETE'
        for f in ('RESULTS.json','physical_updates.jsonl','reverse_calls.jsonl','candidates.jsonl'):history[intervention/f]=(intervention/f).read_text()
        trained=torch.load(intervention/'S0200.pt',map_location='cpu',weights_only=False);assert trained['cursor']==8875
        model=SpatialResidualModel().cuda();model.load_state_dict(payload['model']);model.requires_grad_(False)
        def draw(generator=None):
            if generator is None:model.draw()
            else:
                for layer in model.bayes_layers():layer.sampled=tuple(getattr(layer,'mean_'+p)+getattr(layer,'logsd_'+p).exp()*torch.randn(getattr(layer,'mean_'+p).shape,device='cuda',dtype=torch.float32,generator=generator) for p in ('w','b'))
            costs['theta_draw']+=1
        def density(task,q,kind,fz,noise=None):
            mx,mm,lx,lm,b=q;e1=torch.randn_like(mx) if noise is None else noise[0];e2=torch.randn_like(mm) if noise is None else noise[1]
            x=mx+lx.exp()*e1;m=mm+b*e1+lm.exp()*e2
            terms=dict(image=student_log_prob(image-x-m,.1),shape=student_log_prob(sar(x-fz),.2)+model.logdet,appearance=normal_log_prob(m,torch.zeros_like(m),torch.full_like(m,math.log(.5))),joint_entropy=entropy(q))
            costs['density']+=1;assert costs['density']<=150000;append(root/'density_calls.jsonl',dict(index=costs['density'],kind=kind,task=task['key']))
            return sum(terms.values())/N,{k:v/N for k,v in terms.items()}
        def saa(task,q,kind,grad=False):
            vals=[]
            with torch.set_grad_enabled(grad):
                for _ in range(2):
                    draw()
                    try:
                        fz=model.structure_mean(F.one_hot(labels[task['name']],4).permute(0,3,1,2).to(image.dtype))
                        for _ in range(4):v,_=density(task,q,kind,fz);vals.append(v)
                    finally:model.clear_draw()
            return torch.stack(vals).mean(0)
        old_samples={}
        for line in (p2/'endpoint_scores.jsonl').read_text().splitlines():
            row=json.loads(line)
            if row['stream']==SEEDS[0] and row['theta']==0:old_samples[row['task']]=row['samples']
        def pool(items,phase,first_only=False):
            before=capture_rng_state();values={t['key']:[] for t in items};parts={t['key']:{} for t in items};error=0.
            with torch.no_grad():
                for seed in (SEEDS[:1] if first_only else SEEDS):
                    theta_gen=torch.Generator(device='cuda').manual_seed(seed)
                    for d in range(1 if first_only else 32):
                        draw(theta_gen)
                        try:
                            gens=[torch.Generator(device='cuda').manual_seed(seed*1000+d*10+i) for i in (1,2)]
                            noises=[tuple(torch.randn(image.shape,device='cuda',generator=g) for g in gens) for _ in range(16)];fcache={}
                            for task in items:
                                name=task['name']
                                if name not in fcache:fcache[name]=model.structure_mean(F.one_hot(labels[name],4).permute(0,3,1,2).to(image.dtype))
                                samples=[]
                                for noise in noises:
                                    v,t=density(task,task['q'],phase,fcache[name],noise);assert torch.isfinite(v).all()
                                    samples.append(dict(value=v.cpu().tolist(),parts={k:x.cpu().tolist() for k,x in t.items()}))
                                if first_only:
                                    for a,b in zip(samples,old_samples[task['key']]):
                                        error=max(error,max(abs(x-y) for x,y in zip(a['value'],b['value'])))
                                        for term in a['parts']:error=max(error,max(abs(x-y) for x,y in zip(a['parts'][term],b['parts'][term])))
                                else:append(root/f'{phase}_scores.jsonl',dict(task=task['key'],init=task['init'],candidate=name,stream=seed,theta=d,samples=samples))
                                values[task['key']].append([statistics.mean(x['value'][i] for x in samples) for i in range(2)])
                                for term in samples[0]['parts']:parts[task['key']].setdefault(term,[]).append([statistics.mean(x['parts'][term][i] for x in samples) for i in range(2)])
                            if d%8==7:write(root/'STATUS.json',dict(status='RUNNING',phase=phase,stream=seed,theta=d,costs=costs))
                        finally:model.clear_draw()
            restore_rng_state(before);assert same(before,capture_rng_state());return dict(values=values,parts=parts),error
        _,replay_error=pool(tasks,'baseline_check',True);assert replay_error<=1e-6
        model.load_state_dict(trained['model']);frozen={n:x.detach().cpu().clone() for n,x in model.state_dict().items()};newtasks=[];newresults={}
        init=1
        for name in labels:
            folder=root/f'B_init1_{name}';folder.mkdir();task=dict(key=task_key('B',1,name,200),init=1,name=name)
            refroot=p1 if name in ('GT','BG') else p2
            oldstart=torch.load(refroot/f'B_init1_{name}/S0000.pt',map_location='cpu',weights_only=False)
            q=tuple(x.to('cuda').detach().clone() for x in oldstart['q'])
            assert all(torch.equal(a.cpu(),b) for a,b in zip(q,oldstart['q']))
            restore_rng_state(oldstart['rng']);assert same(oldstart['rng'],payload['rng']);steps=[]
            def local_checkpoint(step):
                storage(s);save(folder/f'S{step:04}.pt',dict(step=step,q=tuple(x.detach().cpu() for x in q),rng=capture_rng_state(),costs=costs.copy(),model_cursor=8875))
                ref=torch.load(refroot/f'B_init1_{name}/S{step:04}.pt',map_location='cpu',weights_only=False)
                assert same(ref['rng'],capture_rng_state());append(root/'RNG_PAIRS.jsonl',dict(init=1,candidate=name,step=step,exact=True))
            local_checkpoint(0)
            for step in range(1,201):
                active.update(out=folder,q=q,step=step-1);q=tuple(v.detach().requires_grad_() for v in q);before_rng=capture_rng_state();before=saa(task,q,'local_gradient',True);after_rng=capture_rng_state();g=torch.autograd.grad(-before.sum(),q)
                assert all(torch.isfinite(v).all() for v in g);costs['local_reverse']+=1;assert costs['local_reverse']<=1000;append(root/'reverse_calls.jsonl',dict(index=costs['local_reverse'],branch=task['key'],step=step))
                chosen=None
                for alpha in (1.,.25,.0625,.015625,.00390625,.0009765625,.000244140625,0.):
                    candidate=tuple((v-.05*alpha*N*dv).detach() for v,dv in zip(q,g));restore_rng_state(before_rng);value=saa(task,candidate,'candidate');finite=bool(torch.isfinite(value).all());ok=finite and bool((value>=before.detach()-1e-6).all()) and float(value.mean())>float(before.detach().mean())+1e-5
                    costs['candidate']+=1;assert costs['candidate']<=8000;append(root/'candidates.jsonl',dict(index=costs['candidate'],branch=task['key'],step=step,alpha=alpha,finite=finite,accepted=ok))
                    if ok:chosen=(candidate,value,alpha);break
                restore_rng_state(after_rng)
                if chosen:q,value,alpha=chosen;costs['accepted']+=1;append(root/'local_updates.jsonl',dict(index=costs['accepted'],branch=task['key'],step=step))
                else:q=tuple(v.detach() for v in q);value=before.detach();alpha=0.
                row=dict(step=step,accepted=chosen is not None,alpha=alpha,before=before.detach().cpu().tolist(),after=value.cpu().tolist());steps.append(row);append(folder/'steps.jsonl',row);active.update(q=q,step=step)
                if step%25==0:local_checkpoint(step);write(root/'STATUS.json',dict(status='RUNNING',phase='local',branch=task['key'],step=step,costs=costs))
            task['q']=q;newtasks.append(task);newresults[name]=dict(accepted=sum(x['accepted'] for x in steps),attempts=200)
        assert same(frozen,model.state_dict()) and costs['local_reverse']==1000
        numeric_new,_=pool(newtasks,'numeric_new_F')
        dynamic=[]
        for init in (0,1):
            for name in labels:
                q=loadq(intervention/f'B_init{init}_{name}/S0200.pt')
                dynamic.append(dict(key=task_key('B',init,name,200),init=init,name=name,q=q))
        model.load_state_dict(payload['model'])
        dynamic_old,_=pool(dynamic,'dynamic_old_F');numeric_old,_=pool(newtasks,'numeric_old_F')
        assert same(payload['model'],model.state_dict()) and all(x.grad is None for x in model.parameters())
        for path,text in history.items():assert path.read_text()==text
        for path,q0 in references.items():assert all(torch.equal(a,b) for a,b in zip(q0,torch.load(path,map_location='cpu',weights_only=False)['q']))
        assert costs['optimizer']==0 and len((root/'RNG_PAIRS.jsonl').read_text().splitlines())==45
        write(root/'RESULTS.json',dict(status='COMPLETE',costs=costs,baseline_replay_max_error=replay_error,checks=check,model_unchanged=True,historical_ledgers_unchanged=True,numeric_starts_exact=True,RNG_pairs=45,branches=newresults,numeric_new_F=numeric_new,numeric_old_F=numeric_old,dynamic_old_F=dynamic_old,space=storage(s)))
        write(root/'STATUS.json',dict(status='COMPLETE',costs=costs));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        if 'out' in active:save(active['out']/'failure_partial.pt',dict(step=active['step'],q=tuple(x.detach().cpu() for x in active['q']),rng=capture_rng_state(),costs=costs))
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'STATUS.json',dict(status='FAILED',costs=costs));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))

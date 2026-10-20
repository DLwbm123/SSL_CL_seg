"""Read-only mean/dispersion endpoint crossing and class2 structure response."""
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
    costs=dict(optimizer=0,reverse=0,local_updates=0,density=0,theta_draw=0,structure_forward=0,mean_forward=0,point_maps=0);active={};hooks=[]
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
            costs['density']+=1;assert costs['density']<=41280;append(root/'density_calls.jsonl',dict(index=costs['density'],kind=kind,task=task['key']))
            return sum(terms.values())/N,{k:v/N for k,v in terms.items()}
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
                                if name not in fcache:
                                    fcache[name]=model.structure_mean(F.one_hot(labels[name],4).permute(0,3,1,2).to(image.dtype));costs['structure_forward']+=1
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
        old_rng=capture_rng_state();baseline,_error=pool(tasks,'old_parity',True);assert _error<=1e-6
        model.load_state_dict(trained['model']);old_samples={}
        for line in (intervention/'fixed_q_scores.jsonl').read_text().splitlines():
            row=json.loads(line)
            if row['stream']==SEEDS[0] and row['theta']==0:old_samples[row['task']]=row['samples']
        _,new_error=pool(tasks,'new_parity',True);assert new_error<=1e-6
        s0=payload['model'];s1=trained['model'];named=dict(model.named_parameters());assert len(named)==100
        assert all(torch.equal(s0[n],s1[n]) for n in s0 if not n.startswith('structure.') or 'prior_' in n or 'initial_' in n)
        grids={};kl={}
        for name,mm,ss in [('00',0,0),('10',1,0),('01',0,1),('11',1,1)]:
            state={n:x.clone() for n,x in s0.items()}
            for n in named:
                if n.startswith('structure.'):
                    state[n]=(s1 if ((mm if '.mean_' in n else ss)) else s0)[n].clone()
            grids[name]=state
            mean_kl=var_kl=0.
            for layer in (0,1,2):
                for part in ('w','b'):
                    key=f'structure.{layer}.';m=state[key+'mean_'+part].double();l=state[key+'logsd_'+part].double();pm=state[key+'prior_mean_'+part].double();pl=state[key+'prior_logsd_'+part].double()
                    mean_kl+=float(.5*((m-pm)*(-pl).exp()).square().sum());var_kl+=float(.5*(torch.exp(2*(l-pl))-1-2*(l-pl)).sum())
            kl[name]=dict(mean=mean_kl/(2*N),variance=var_kl/(2*N),total=(mean_kl+var_kl)/(2*N))
        assert all(torch.equal(grids['00'][n],s0[n]) and torch.equal(grids['11'][n],s1[n]) for n in s0)
        assert abs(kl['11']['total']-kl['10']['total']-kl['01']['total']+kl['00']['total'])<1e-12
        readings={}
        for name in ('10','01'):
            model.load_state_dict(grids[name]);evaluation,_=pool(tasks,'component_'+name);readings[name]=evaluation
            assert same(grids[name],model.state_dict()) and all(p.grad is None for p in model.parameters())
        # Deterministic mean-weight response diagnostics; no private fields are exported.
        profiles={};mean_outputs={}
        c2=(gt==2)[:,None];near3=F.max_pool2d(c2.float(),7,stride=1,padding=3).bool();inside3=~F.max_pool2d(F.pad((~c2).float(),(3,3,3,3),value=1),7,stride=1).bool()
        near4=F.max_pool2d(F.pad(c2.float(),(4,4,4,4),mode='circular'),9,stride=1).bool()
        regions=dict(class2=c2,interior=inside3,boundary=c2&~inside3,neighbor=near3&~c2,outside=~near3)
        def aggregate(field,mask):
            ans=[]
            for x,m in zip(field,mask):
                v=x[m];ans.append(dict(count=v.numel(),mean=float(v.mean()) if v.numel() else None,RMS=float(v.square().mean().sqrt()) if v.numel() else None,maxabs=float(v.abs().max()) if v.numel() else None))
            return ans
        qgt=tasks[0]['q'];mx=qgt[0];lx=qgt[2]
        with torch.no_grad():
            for tag in ('00','11'):
                model.load_state_dict(grids[tag])
                for layer in model.bayes_layers():layer.sampled=(layer.mean_w,layer.mean_b)
                try:
                    vals={}
                    for name in ('GT','DELETE2'):
                        vals[name]=model.structure_mean(F.one_hot(labels[name],4).permute(0,3,1,2).to(image.dtype));costs['mean_forward']+=1
                        residual=sar(mx-vals[name]);point=-(2.5)*torch.log(.2**2+residual.square()/2);costs['point_maps']+=1
                        profiles.setdefault(tag,{})[name]={k:dict(residual=aggregate(residual,m),point_shape_up_to_constant=aggregate(point,m)) for k,m in regions.items()}
                    response=vals['GT']-vals['DELETE2'];filtered=sar(response)
                    assert float(response[~near3].abs().max())<=1e-6 and float(filtered[~near4].abs().max())<=1e-6
                    profiles[tag]['deletion_response']={k:aggregate(response,m) for k,m in regions.items()};profiles[tag]['outside_support_max']=float(response[~near3].abs().max());profiles[tag]['outside_SAR_support_max']=float(filtered[~near4].abs().max())
                    profiles[tag]['qGT_x_sd']={k:aggregate(lx.exp(),m) for k,m in regions.items()};mean_outputs[tag]=vals
                finally:model.clear_draw()
        mean_delta={name:{k:aggregate(mean_outputs['11'][name]-mean_outputs['00'][name],m) for k,m in regions.items()} for name in ('GT','DELETE2')}
        model.load_state_dict(payload['model']);assert same(payload['model'],model.state_dict()) and all(p.grad is None for p in model.parameters())
        assert same(old_rng,capture_rng_state())
        for path,text in history.items():assert path.read_text()==text
        for path,q in references.items():assert all(torch.equal(a,b) for a,b in zip(q,torch.load(path,map_location='cpu',weights_only=False)['q']))
        assert costs['density']==41280 and costs['theta_draw']==258 and costs['structure_forward']==1290 and costs['mean_forward']==4 and costs['point_maps']==4
        result=dict(status='COMPLETE',costs=costs,old_parity_error=_error,new_parity_error=new_error,readings=readings,structure_KL=kl,profiles=profiles,mean_delta=mean_delta,model_rng_and_history_unchanged=True,space=storage(s))
        write(root/'RESULTS.json',result);write(root/'STATUS.json',dict(status='COMPLETE',costs=costs));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'STATUS.json',dict(status='FAILED',costs=costs));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))

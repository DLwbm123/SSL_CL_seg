"""Structure posterior intervention, fixed-q effect and matched re-inference."""
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


def active_parameter(name):return name.startswith('structure.')


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
        qtrain=tasks[0]['q'];qtrain_cpu=tuple(v.detach().cpu().clone() for v in qtrain)
        model=SpatialResidualModel().cuda();model.load_state_dict(payload['model'])
        def draw(generator=None):
            if generator is None:model.draw()
            else:
                for layer in model.bayes_layers():layer.sampled=tuple(getattr(layer,'mean_'+p)+getattr(layer,'logsd_'+p).exp()*torch.randn(getattr(layer,'mean_'+p).shape,device='cuda',dtype=torch.float32,generator=generator) for p in ('w','b'))
            costs['theta_draw']+=1
        def density(task,q,kind,fz,noise=None):
            mx,mm,lx,lm,b=q;e1=torch.randn_like(mx) if noise is None else noise[0];e2=torch.randn_like(mm) if noise is None else noise[1]
            x=mx+lx.exp()*e1;m=mm+b*e1+lm.exp()*e2
            terms=dict(image=student_log_prob(image-x-m,.1),shape=student_log_prob(sar(x-fz),.2)+model.logdet,appearance=normal_log_prob(m,torch.zeros_like(m),torch.full_like(m,math.log(.5))),joint_entropy=entropy(q))
            costs['density']+=1;assert costs['density']<=200000;append(root/'density_calls.jsonl',dict(index=costs['density'],kind=kind,task=task['key']))
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
        class StructureMethod(Method):
            def training_step(self,lb,ub,global_step,site_step):
                assert self.nl==2 and self.nu==0 and ub is None
                b=saa(tasks[0],qtrain,'training',True);kl=self.model.kl()/(2*N);loss=-b.mean()+kl
                return MethodStepOutput(loss,dict(B_coordinate=-b.mean(),global_KL=kl),dict(B=float(b.mean().detach()),global_KL=float(kl.detach())))
        method=StructureMethod(model,2,0);method.begin_site('Siemens',None,base+200)
        opt=torch.optim.Adam(model.parameters(),lr=3e-4,weight_decay=0);scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
        trainer=Trainer(method,optimizer=opt,scheduler=scheduler,device='cuda',amp=False,grad_clip_norm=1.)
        restore(payload,method,trainer);assert same({k:payload[k] for k in snapshot(method,trainer,base)},snapshot(method,trainer,base))
        named=list(model.named_parameters());assert len(named)==100 and sum(active_parameter(n) for n,v in named)==12
        for n,v in named:v.requires_grad_(active_parameter(n))
        frozen={n:v for n,v in payload['model'].items() if not active_parameter(n) or 'prior_' in n or 'initial_' in n};frozen_adam={i:copy.deepcopy(opt.state_dict()['state'][i]) for i,(n,v) in enumerate(named) if not active_parameter(n)}
        active.update(method=method,trainer=trainer,cursor=base)
        with torch.no_grad():recognition=model.recognition_logits(image).cpu().clone()
        _,replay_error=pool(tasks,'baseline_check',True);assert replay_error<=1e-6
        phase=['golden'];index=[0];goldens=[]
        def pre(opt,a,kw):
            assert costs['optimizer']<204 and opt.param_groups[0]['lr']==3e-4
            assert all(v.grad is None if not active_parameter(n) else v.grad is not None and torch.isfinite(v.grad).all() for n,v in named)
        def post(opt,a,kw):
            costs['optimizer']+=1;costs[phase[0]]+=1;active['cursor']=base+index[0]+1
            append(root/'physical_updates.jsonl',dict(physical=costs['optimizer'],phase=phase[0],main_step=index[0]+1))
        hooks=[opt.register_step_pre_hook(pre),opt.register_step_post_hook(post)]
        def frozen_check():
            assert all(torch.equal(v,model.state_dict()[n].cpu()) for n,v in frozen.items()) and same(qtrain_cpu,qtrain)
            assert all(same(v,opt.state_dict()['state'][i]) for i,v in frozen_adam.items())
        def checkpoint(cursor):
            out=snapshot(method,trainer,cursor);out['fixed_training_B_q']=qtrain_cpu;return out
        save(root/'S0000.pt',checkpoint(base))
        def update(i):
            index[0]=i;active['cursor']=base+i;return trainer.train_step(batch,None,state=TrainerState(global_step=base+i,site_step=base+i))
        for i in range(200):
            if i in (0,199):
                phase[0]='golden';before=snapshot(method,trainer,base+i);a=update(i);expected=snapshot(method,trainer,base+i+1);restore(before,method,trainer);b=update(i)
                assert torch.equal(a.total_loss,b.total_loss) and same(expected,snapshot(method,trainer,base+i+1));restore(before,method,trainer);goldens.append(dict(main_step=i+1,full_state_equal=True));write(root/'GOLDEN.json',goldens)
            phase[0]='main';out=update(i);append(root/'steps.jsonl',dict(step=i+1,loss=float(out.total_loss.detach()),scalars=out.scalars))
            if (i+1)%25==0:frozen_check();write(root/'STATUS.json',dict(status='RUNNING',phase='training',step=i+1,costs=costs,space=storage(s)))
            if (i+1)%50==0:storage(s);save(root/f'S{i+1:04}.pt',checkpoint(base+i+1))
        frozen_check();assert costs['optimizer']==204 and costs['main']==200 and costs['golden']==4
        state=opt.state_dict()['state'];oldstate=payload['trainer']['optimizer_state']['state']
        for i,(n,v) in enumerate(named):
            if active_parameter(n):assert float(state[i]['step'])==float(oldstate[i]['step'])+200
        with torch.no_grad():assert torch.equal(recognition,model.recognition_logits(image).cpu())
        for n,v in named:v.requires_grad_(False);v.grad=None
        trained_state={n:v.detach().cpu().clone() for n,v in model.state_dict().items()};fixed_effect,_=pool(tasks,'fixed_q')
        trained_rng=capture_rng_state();newtasks=[];newresults={}
        for init in (0,1):
            for name in labels:
                folder=root/f'B_init{init}_{name}';folder.mkdir();task=dict(key=task_key('B',init,name,200),init=init,name=name)
                baseq=loadq(p1/f'B_init{init}_GT/S0000.pt')
                with torch.no_grad():
                    if init==0:q=tuple(v.detach().clone() for v in baseq)
                    else:
                        for layer in model.bayes_layers():layer.sampled=(layer.mean_w,layer.mean_b)
                        try:mx=model.structure_mean(F.one_hot(labels[name],4).permute(0,3,1,2).to(image.dtype));q=(mx,image-mx,baseq[2],baseq[3],torch.zeros_like(mx))
                        finally:model.clear_draw()
                        q=tuple(v.detach().clone() for v in q)
                restore_rng_state(payload['rng']);steps=[]
                def local_checkpoint(step):
                    storage(s);save(folder/f'S{step:04}.pt',dict(step=step,q=tuple(v.detach().cpu() for v in q),rng=capture_rng_state(),costs=costs.copy(),model_cursor=base+200))
                    ref=(p1/f'B_init{init}_{name}' if name in ('GT','BG') else p2/f'B_init{init}_{name}')/f'S{step:04}.pt'
                    assert same(torch.load(ref,map_location='cpu',weights_only=False)['rng'],capture_rng_state());append(root/'RNG_PAIRS.jsonl',dict(init=init,candidate=name,step=step,exact=True))
                local_checkpoint(0)
                for step in range(1,201):
                    active.update(out=folder,q=q,step=step-1);q=tuple(v.detach().requires_grad_() for v in q);before_rng=capture_rng_state();before=saa(task,q,'local_gradient',True);after_rng=capture_rng_state();g=torch.autograd.grad(-before.sum(),q)
                    assert all(torch.isfinite(v).all() for v in g);costs['local_reverse']+=1;assert costs['local_reverse']<=2000;append(root/'reverse_calls.jsonl',dict(index=costs['local_reverse'],branch=task['key'],step=step))
                    chosen=None
                    for alpha in (1.,.25,.0625,.015625,.00390625,.0009765625,.000244140625,0.):
                        candidate=tuple((v-.05*alpha*N*dv).detach() for v,dv in zip(q,g));restore_rng_state(before_rng);value=saa(task,candidate,'candidate');finite=bool(torch.isfinite(value).all());ok=finite and bool((value>=before.detach()-1e-6).all()) and float(value.mean())>float(before.detach().mean())+1e-5
                        costs['candidate']+=1;assert costs['candidate']<=16000;append(root/'candidates.jsonl',dict(index=costs['candidate'],branch=task['key'],step=step,alpha=alpha,finite=finite,accepted=ok))
                        if ok:chosen=(candidate,value,alpha);break
                    restore_rng_state(after_rng)
                    if chosen:q,value,alpha=chosen;costs['accepted']+=1;append(root/'local_updates.jsonl',dict(index=costs['accepted'],branch=task['key'],step=step))
                    else:q=tuple(v.detach() for v in q);value=before.detach();alpha=0.
                    row=dict(step=step,accepted=chosen is not None,alpha=alpha,before=before.detach().cpu().tolist(),after=value.cpu().tolist());steps.append(row);append(folder/'steps.jsonl',row);active.update(q=q,step=step)
                    if step%25==0:local_checkpoint(step);write(root/'STATUS.json',dict(status='RUNNING',phase='local',branch=task['key'],step=step,costs=costs))
                task['q']=q;newtasks.append(task);newresults[f'init{init}_{name}']=dict(accepted=sum(x['accepted'] for x in steps),attempts=200)
        refined,_=pool(newtasks,'refined');metrics={};basevalues=control['endpoint_evaluation']['values']
        for phase,evaluation in [('control',dict(values=basevalues)),('fixed_q',fixed_effect),('refined',refined)]:
            metrics[phase]={}
            for init in (0,1):
                vv=evaluation['values'];match=vv[task_key('B',init,'GT',200)];metrics[phase][f'init{init}']={name:contrast(match,vv[task_key('B',init,name,200)]) for name in ('CROSS','BG','SHIFT','DELETE2')}
        assert costs['local_reverse']==2000 and len((root/'RNG_PAIRS.jsonl').read_text().splitlines())==90
        assert same(trained_state,model.state_dict()) and all(v.grad is None for n,v in named)
        for path,text in history.items():assert path.read_text()==text
        for path,q in references.items():assert all(torch.equal(a,b) for a,b in zip(q,torch.load(path,map_location='cpu',weights_only=False)['q']))
        restore_rng_state(trained_rng)
        result=dict(status='COMPLETE',costs=costs,checks=check,baseline_replay_max_error=replay_error,source_cursor=base,final_cursor=base+200,golden=goldens,model_88_and_adam_prior_fixed=True,recognition_exactly_unchanged=True,inference_model_frozen=True,old_sources_and_locals_unchanged=True,RNG_pairs=90,fixed_q=fixed_effect,refined=refined,metrics=metrics,branches=newresults,space=storage(s))
        write(root/'RESULTS.json',result);write(root/'STATUS.json',dict(status='COMPLETE',costs=costs));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        if 'method' in active:storage(s);save(root/'failure_model_partial.pt',dict(**snapshot(active['method'],active['trainer'],active['cursor']),fixed_training_B_q=qtrain_cpu))
        if 'out' in active:save(active['out']/'failure_partial.pt',dict(step=active['step'],q=tuple(v.detach().cpu() for v in active['q']),rng=capture_rng_state(),costs=costs))
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'STATUS.json',dict(status='FAILED',costs=costs));write(root/'EXIT.json',dict(exit_code=1));raise
    finally:
        for hook in hooks:hook.remove()


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))

"""Qualification-gated candidate comparison with independent Theta/local streams."""
import json, math, os, statistics, traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset, collate_labeled
from lcrseg.engine.trainer import seed_everything
from lcrseg.engine.checkpoint import capture_rng_state, restore_rng_state
from care_seg_study_v1.run import write, append, save
from sc_bayeseg.run import same
from sc_bayeseg.model import sar, student_log_prob, normal_log_prob
from sc_bayeseg.window_v1.spatial_residual import SpatialResidualModel
from sc_bayeseg.window_v1.factorized_p1 import entropy, self_check
from sc_bayeseg.window_v1.common import storage

SEEDS=(220163,230163,240163,250163)
TOL=1e-5


def summarize(values):
    mean=statistics.mean(values);se=statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else 0.
    flows=[statistics.mean(values[i:i+32]) for i in range(0,len(values),32)]
    low,high=mean-1.96*se,mean+1.96*se
    label='positive' if low>TOL and all(v>TOL for v in flows) else 'negative' if high<-TOL and all(v<-TOL for v in flows) else 'undecided'
    return dict(mean=mean,MC_SE=se,MC95=[low,high],flow_means=flows,classification=label,n_theta=len(values))


def contrast(left,right):
    d=[[a-b for a,b in zip(x,y)] for x,y in zip(left,right)]
    return dict(per_image=[summarize([v[i] for v in d]) for i in range(2)],mean=summarize([sum(v)/2 for v in d]))


def task_key(block,init,name,step):
    return f'{block}_i{init}_{name}_s{step}'


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False)
    costs=dict(reverse=0,candidate=0,accepted=0,density=0,theta_draw=0,optimizer=0,model_applications=0);active={}
    try:
        storage(s);checks=self_check();assert summarize([1.]*128)['classification']=='positive'
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        src=Path(s['spatial_generator']);p1=Path(s['factorized_p1'])
        ledgers={(src/'physical_updates.jsonl'):(src/'physical_updates.jsonl').read_text(),(p1/'reverse_calls.jsonl'):(p1/'reverse_calls.jsonl').read_text(),(p1/'RESULTS.json'):(p1/'RESULTS.json').read_text()}
        old=json.loads(ledgers[p1/'RESULTS.json']);assert old['status']=='COMPLETE' and old['costs']['reverse']==1600
        assert len(ledgers[src/'physical_updates.jsonl'].splitlines())==604
        payload=torch.load(src/'S0600.pt',map_location='cpu',weights_only=False);assert payload['cursor']==8675
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text());assert len(sel['train_indices'])==2
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in sel['train_indices']]).to('cuda')
        image=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False);gt=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long();N=128**2
        labels=dict(GT=gt,BG=torch.zeros_like(gt),CROSS=gt.flip(0),SHIFT=torch.roll(gt,(32,32),(-2,-1)),DELETE2=torch.where(gt==2,0,gt))
        assert all(torch.equal((gt==k).sum((-2,-1)),(labels['SHIFT']==k).sum((-2,-1))) for k in range(4))
        applicable={name:[bool(v) for v in (z!=gt).flatten(1).any(1).cpu().tolist()] for name,z in labels.items()}
        model=SpatialResidualModel().cuda();model.load_state_dict(payload['model']);model.requires_grad_(False);original_rng=capture_rng_state()
        references={}
        def load(block,init,name,step):
            file=p1/f'{block}_init{init}_{name}'/f'S{step:04}.pt'
            state=torch.load(file,map_location='cpu',weights_only=False)
            assert state['step']==step and state['model_cursor']==8675
            references[file]=state['q']
            return dict(key=task_key(block,init,name,step),block=block,init=init,name=name,step=step,q=tuple(v.to('cuda') for v in state['q']))
        def draw(generator=None):
            if generator is None:model.draw()
            else:
                for layer in model.bayes_layers():
                    layer.sampled=tuple(getattr(layer,'mean_'+part)+getattr(layer,'logsd_'+part).exp()*torch.randn(getattr(layer,'mean_'+part).shape,device='cuda',dtype=torch.float32,generator=generator) for part in ('w','b'))
            costs['theta_draw']+=1
        def density(task,q,kind,fz=None,noise=None):
            if task['block']=='A':
                cm,cl,rm,rl=q
                ec=torch.randn_like(cm) if noise is None else noise[2];er=torch.randn_like(rm) if noise is None else noise[3]
                c=cm+cl.exp()*ec;r=rm+rl.exp()*er;z=labels[task['name']]
                cat=-F.cross_entropy(model.spatial_logits(c,r),z,reduction='none').sum((-2,-1))
                kc=.5*(cm.square()+(2*cl).exp()-1-2*cl).flatten(1).sum(1)
                terms=dict(categories=cat,negative_KLc=-kc,negative_KLR=-model.residual_kl((rm,rl)))
            else:
                mx,mm,lx,lm,b=q
                e1=torch.randn_like(mx) if noise is None else noise[0];e2=torch.randn_like(mm) if noise is None else noise[1]
                x=mx+lx.exp()*e1;m=mm+b*e1+lm.exp()*e2
                terms=dict(image=student_log_prob(image-x-m,.1),shape=student_log_prob(sar(x-fz),.2)+model.logdet,appearance=normal_log_prob(m,torch.zeros_like(m),torch.full_like(m,math.log(.5))),joint_entropy=entropy(q))
            costs['density']+=1;assert costs['density']<=150000
            append(root/'density_calls.jsonl',dict(index=costs['density'],branch=task['key'],kind=kind))
            return sum(terms.values())/N,{k:v/N for k,v in terms.items()}
        def old_saa(task,q,kind,draws=2,grad=False):
            vals=[];parts=[]
            with torch.set_grad_enabled(grad):
                for _ in range(draws):
                    draw()
                    try:
                        fz=model.structure_mean(F.one_hot(labels[task['name']],4).permute(0,3,1,2).to(image.dtype)) if task['block']=='B' else None
                        for _ in range(4):
                            v,t=density(task,q,kind,fz);vals.append(v);parts.append({k:x.detach()/1 for k,x in t.items()})
                    finally:model.clear_draw()
            return torch.stack(vals).mean(0),{k:torch.stack([t[k] for t in parts]).mean(0).cpu().tolist() for k in parts[0]}
        def pool(tasks,nlocal,phase,prefix=None):
            rng=capture_rng_state();values={t['key']:[] for t in tasks};summed_parts={t['key']:{} for t in tasks};prefixes={};prefix_error=0.
            with torch.no_grad():
                for seed in SEEDS:
                    theta_gen=torch.Generator(device='cuda').manual_seed(seed)
                    for d in range(32):
                        draw(theta_gen)
                        try:
                            generators=[torch.Generator(device='cuda').manual_seed(seed*1000+d*10+i) for i in range(1,5)]
                            noises=[]
                            for _ in range(nlocal):
                                shapes=(image.shape,image.shape,(2,16),(2,16,16,16))
                                noises.append(tuple(torch.randn(shape,device='cuda',dtype=torch.float32,generator=g) for shape,g in zip(shapes,generators)))
                            fcache={}
                            for task in tasks:
                                if task['block']=='B' and task['name'] not in fcache:
                                    fcache[task['name']]=model.structure_mean(F.one_hot(labels[task['name']],4).permute(0,3,1,2).to(image.dtype))
                                samples=[]
                                for noise in noises:
                                    v,t=density(task,task['q'],phase,fcache.get(task['name']),noise)
                                    assert torch.isfinite(v).all() and all(torch.isfinite(x).all() for x in t.values())
                                    samples.append(dict(value=v.cpu().tolist(),parts={k:x.cpu().tolist() for k,x in t.items()}))
                                record=dict(task=task['key'],block=task['block'],init=task['init'],candidate=task['name'],step=task['step'],stream=seed,theta=d,samples=samples)
                                append(root/f'{phase}_scores.jsonl',record)
                                vmean=[statistics.mean(x['value'][i] for x in samples) for i in range(2)];values[task['key']].append(vmean)
                                for term in samples[0]['parts']:
                                    summed_parts[task['key']].setdefault(term,[]).append([statistics.mean(x['parts'][term][i] for x in samples) for i in range(2)])
                                index=(task['key'],seed,d)
                                if phase=='qualification' and task['block']=='B' and task['step']==200:prefixes[index]=samples
                                if prefix is not None and index in prefix:
                                    for a,b in zip(samples[:8],prefix[index]):
                                        prefix_error=max(prefix_error,max(abs(x-y) for x,y in zip(a['value'],b['value'])))
                                        for term in a['parts']:prefix_error=max(prefix_error,max(abs(x-y) for x,y in zip(a['parts'][term],b['parts'][term])))
                            if d%8==7:write(root/'STATUS.json',dict(status='RUNNING',phase=phase,stream=seed,theta=d,costs=costs))
                        finally:model.clear_draw()
            restore_rng_state(rng);assert same(rng,capture_rng_state()) and prefix_error<=1e-6
            return dict(values=values,parts=summed_parts),prefixes,prefix_error
        qt=[];replay=[]
        for init in (0,1):
            for name in ('GT','BG'):
                for step in (0,100,150,200):
                    t=load('B',init,name,step);qt.append(t)
                    if step in (0,200):replay.append(t)
                t=load('A',init,name,200);qt.append(t);replay.append(t)
        replay_errors={}
        for task in replay:
            before=capture_rng_state();seed_everything(80163);v,t=old_saa(task,task['q'],'replay',32);restore_rng_state(before)
            state='initial' if task['step']==0 else 'final';ref=old['branches'][f"{task['block']}_init{task['init']}_{task['name']}"][state]
            errs=[abs(x-y) for x,y in zip(v.cpu().tolist(),ref['value'])]
            for k in t:errs.extend(abs(x-y) for x,y in zip(t[k],ref['parts'][k]))
            replay_errors[task['key']]=max(errs)
        qualification=dict(replay_max_error=max(replay_errors.values()),replay_errors=replay_errors,replay_pass=max(replay_errors.values())<=1e-6,branches={})
        qeval=None;prefix={}
        if qualification['replay_pass']:
            qeval,prefix,_=pool(qt,8,'qualification')
            for init in (0,1):
                for name in ('GT','BG'):
                    vv=qeval['values'];get=lambda step:vv[task_key('B',init,name,step)]
                    changes={f'{a}-{b}':contrast(get(a),get(b)) for a,b in ((200,0),(150,100),(200,150))}
                    endpoint=changes['200-0'];tail=changes['200-150']
                    improve=all(x['MC95'][0]>TOL for x in endpoint['per_image']+[endpoint['mean']]) and all(x>TOL for x in endpoint['mean']['flow_means'])
                    decay=any(x['MC95'][1]<-TOL for x in tail['per_image']+[tail['mean']])
                    qualification['branches'][f'init{init}_{name}']=dict(changes=changes,endpoint_improvement=improve,clear_late_decay=decay,pass_gate=improve and not decay)
            qualification['passed']=all(b['pass_gate'] for b in qualification['branches'].values())
        else:qualification['passed']=False
        write(root/'QUALIFICATION.json',qualification)
        final_metrics={};new_results={};end_eval=None;prefix_error=None
        if qualification['passed']:
            endtasks=[t for t in qt if t['block']=='B' and t['step']==200]
            for init in (0,1):
                for name in ('CROSS','SHIFT','DELETE2'):
                    out=root/f'B_init{init}_{name}';out.mkdir();task=dict(key=task_key('B',init,name,200),block='B',init=init,name=name,step=200)
                    base=load('B',init,'GT',0)['q']
                    with torch.no_grad():
                        if init==0:q=tuple(v.detach().clone() for v in base)
                        else:
                            for layer in model.bayes_layers():layer.sampled=(layer.mean_w,layer.mean_b)
                            try:mx=model.structure_mean(F.one_hot(labels[name],4).permute(0,3,1,2).to(image.dtype));q=(mx,image-mx,base[2],base[3],torch.zeros_like(mx))
                            finally:model.clear_draw()
                            q=tuple(v.detach().clone() for v in q)
                    restore_rng_state(payload['rng']);steps=[]
                    def checkpoint(step):
                        storage(s);save(out/f'S{step:04}.pt',dict(step=step,q=tuple(v.detach().cpu() for v in q),rng=capture_rng_state(),costs=costs.copy(),model_cursor=8675))
                        ref=torch.load(p1/f'B_init{init}_GT'/f'S{step:04}.pt',map_location='cpu',weights_only=False)
                        assert same(ref['rng'],capture_rng_state());append(root/'RNG_PAIRS.jsonl',dict(init=init,candidate=name,step=step,exact=True))
                    checkpoint(0)
                    for step in range(1,201):
                        active.update(out=out,q=q,step=step-1);q=tuple(v.detach().requires_grad_() for v in q);start=capture_rng_state();before,_=old_saa(task,q,'gradient',grad=True);after=capture_rng_state()
                        g=torch.autograd.grad(-before.sum(),q);assert all(torch.isfinite(v).all() for v in g)
                        costs['reverse']+=1;assert costs['reverse']<=1200;append(root/'reverse_calls.jsonl',dict(index=costs['reverse'],branch=task['key'],step=step))
                        chosen=None
                        for alpha in (1.,.25,.0625,.015625,.00390625,.0009765625,.000244140625,0.):
                            candidate=tuple((v-.05*alpha*N*dv).detach() for v,dv in zip(q,g));restore_rng_state(start);value,_=old_saa(task,candidate,'candidate')
                            finite=bool(torch.isfinite(value).all());ok=finite and bool((value>=before.detach()-1e-6).all()) and float(value.mean())>float(before.detach().mean())+TOL
                            costs['candidate']+=1;assert costs['candidate']<=9600
                            append(root/'candidates.jsonl',dict(index=costs['candidate'],branch=task['key'],step=step,alpha=alpha,finite=finite,accepted=ok))
                            if ok:chosen=(candidate,value,alpha);break
                        restore_rng_state(after)
                        if chosen:q,value,alpha=chosen;costs['accepted']+=1;append(root/'local_updates.jsonl',dict(index=costs['accepted'],branch=task['key'],step=step))
                        else:q=tuple(v.detach() for v in q);value=before.detach();alpha=0.
                        row=dict(step=step,accepted=chosen is not None,alpha=alpha,before=before.detach().cpu().tolist(),after=value.cpu().tolist());steps.append(row);append(out/'steps.jsonl',row);active.update(q=q,step=step)
                        if step%25==0:checkpoint(step);write(root/'STATUS.json',dict(status='RUNNING',phase='candidates',branch=task['key'],step=step,costs=costs))
                    task['q']=q;endtasks.append(task);new_results[f'init{init}_{name}']=dict(attempts=200,accepted=sum(x['accepted'] for x in steps))
            assert costs['reverse']==1200 and len((root/'RNG_PAIRS.jsonl').read_text().splitlines())==54
            end_eval,_,prefix_error=pool(endtasks,16,'endpoint',prefix)
            for init in (0,1):
                vv=end_eval['values'];match=vv[task_key('B',init,'GT',200)];metrics={}
                for name in ('CROSS','BG','SHIFT','DELETE2'):
                    val=contrast(match,vv[task_key('B',init,name,200)]);val['applicable_per_image']=applicable[name]
                    for i,valid in enumerate(applicable[name]):
                        if not valid:val['per_image'][i]['classification']='not_applicable'
                    if not all(applicable[name]):val['mean']['classification']='partially_or_not_applicable'
                    metrics['pair' if name=='CROSS' else name]=val
                final_metrics[f'init{init}']=metrics
        assert same(payload['model'],model.state_dict()) and all(v.grad is None for v in model.parameters())
        for path,text in ledgers.items():assert path.read_text()==text
        for path,q in references.items():
            ref=torch.load(path,map_location='cpu',weights_only=False)['q'];assert all(torch.equal(a,b) for a,b in zip(q,ref))
        restore_rng_state(original_rng);assert same(original_rng,capture_rng_state())
        for filename,key in [('density_calls.jsonl','density'),('reverse_calls.jsonl','reverse'),('candidates.jsonl','candidate'),('local_updates.jsonl','accepted')]:
            entries=[json.loads(x) for x in (root/filename).read_text().splitlines()] if (root/filename).exists() else []
            assert [x['index'] for x in entries]==list(range(1,costs[key]+1))
        result=dict(status='COMPLETE' if qualification['passed'] else 'QUALIFICATION_STOP',qualification=qualification,costs=costs,checks=checks,streams=list(SEEDS),qualification_evaluation=qeval,endpoint_evaluation=end_eval,endpoint_prefix_error=prefix_error,metrics=final_metrics,new_branches=new_results,applicable_per_image=applicable,model_rng_grad_history_and_old_local_states_unchanged=True,source_cursor=8675,no_heldout_U=True,space=storage(s))
        write(root/'RESULTS.json',result);write(root/'STATUS.json',dict(status=result['status'],costs=costs));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        if active:storage(s);save(active['out']/'failure_partial.pt',dict(step=active['step'],q=tuple(v.detach().cpu() for v in active['q']),rng=capture_rng_state(),costs=costs))
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'STATUS.json',dict(status='FAILED',costs=costs));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))

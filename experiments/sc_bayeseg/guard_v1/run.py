"""Frozen parameter/history swaps, then G and schedule-matched D."""
import copy,json,os,time,traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.engine.trainer import Trainer,TrainerState,seed_everything
from lcrseg.engine.checkpoint import capture_rng_state,restore_rng_state
from care_seg_study_v1.run import write,append,save
from sc_bayeseg.split_v1.model import Model,semantic
from sc_bayeseg.jointwrite_v1.run import Adapter
from sc_bayeseg.run import snapshot,restore,same
from sc_bayeseg.adam_audit_v1.run import candidate,dot,norm
from sc_bayeseg.finite_replay_v1.run import cpu,logits,values,summaries
from sc_bayeseg.diagnostic_v1.run import dice


def frame(model,image,y):
    rng=capture_rng_state()
    try:
        with torch.no_grad():
            z=logits(model,image);mean=z.softmax(1)
            seed_everything(50163);ps=[]
            for _ in range(4):
                h,_=model.proposal(image);zz=torch.cat([torch.zeros_like(h['ell'][:,:1]),h['ell']],1)
                ps.append(zz.softmax(1))
            probs=dict(mean=mean,stochastic=torch.stack(ps).mean(0));vector=[];metrics={};preds={}
            for channel,p in probs.items():
                loss=-p.clamp_min(1e-30).log().gather(1,y[:,None])[:,0];pred=p.argmax(1)
                for i in range(2):
                    vector.append(float(loss[i].mean()))
                    for c in [1,2,3]:vector.append(float(loss[i][y[i]==c].mean()))
                preds[channel]=pred.detach()
                metrics[channel]=dict(ce=float(loss.mean()),class2_ce=float(loss[y==2].mean()),dice=dice(pred,y),
                                      class2_confusion=[int(((y==2)&(pred==c)).sum()) for c in range(4)])
            assert all(torch.isfinite(torch.tensor(v)) for v in vector)
            comp=z.masked_fill(F.one_hot(y,4).permute(0,3,1,2).bool(),-torch.inf).argmax(1)
            return dict(vector=vector,metrics=metrics,preds=preds,margin=(z[:,2]-z.gather(1,comp[:,None])[:,0]).detach(),background_margin=z[:,2].detach())
    finally:restore_rng_state(rng)


def accepts(vector,baseline):return all(v<=b+1e-7+1e-6*abs(b) for v,b in zip(vector,baseline))


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);start=time.time()
    costs=dict(optimizer=0,main=0,golden=0,candidates=0,extra_reverse=0);active={};hooks=[]
    def status(phase,**kw):write(root/'STATUS.json',dict(phase=phase,costs=costs.copy(),**kw))
    def count(kind,**kw):
        assert costs['candidates']<143
        costs['candidates']+=1
        append(root/'candidate_evaluations.jsonl',dict(index=costs['candidates'],kind=kind,**kw))
    def grad(*a,**kw):
        assert costs['extra_reverse']<10
        costs['extra_reverse']+=1;append(root/'reverse_calls.jsonl',dict(index=costs['extra_reverse']))
        return torch.autograd.grad(*a,**kw)
    try:
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        origin=Path(s['origin']);replay=Path(s['replay'])
        for p in [origin,replay]:assert json.loads((p/'EXIT.json').read_text())['exit_code']==0 and not (p/'FAILURE.json').exists()
        initial=torch.load(origin/'J'/'initial.pt',weights_only=False,map_location='cpu')
        assert initial['cursor']==3000 and initial['population']==dict(labeled=2,unlabeled=0)
        state=lambda k:torch.load(replay/f'S{k:02}.pt',weights_only=False,map_location='cpu')
        assert same(initial,state(0))
        selection=json.loads((origin/'PRIVATE_SELECTION.json').read_text())
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in selection['train_indices']]).to('cuda')
        image=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False)
        y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        assert int((y==2).sum())==490
        model=Model().cuda();method=Adapter(model,'joint');method.nl=2;method.begin_site('Siemens',None,3600)
        named=list(model.named_parameters());params=[p for _,p in named];assert len(params)==100
        flags=[semantic(n) for n,_ in named];assert sum(flags)==40
        def weights(p):return [p['model'][n].cuda() for n,_ in named]
        model.load_state_dict(initial['model']);restore_rng_state(initial['rng']);baseline=frame(model,image,y)
        assert same(capture_rng_state(),initial['rng'])
        oldmetrics=json.loads((replay/'HISTORICAL.json').read_text())['initial']
        assert baseline['metrics']['stochastic']['dice']==oldmetrics['stochastic_dice']
        assert baseline['metrics']['mean']['dice']==oldmetrics['direct_dice']
        if s.get('preflight_only'):
            assert same(cpu(model.state_dict()),initial['model'])
            write(root/'ZERO_CHECK.json',dict(status='PASS',costs=costs,original_baseline_equal=True,initial_frame_accepted=accepts(baseline['vector'],baseline['vector']),all_state_unchanged=True,vector=baseline['vector']))
            write(root/'EXIT.json',dict(exit_code=0));return
        # Six CE reverses are shared across all noise/history combinations.
        cache={}
        for k in [0,1,24]:
            p=state(k);model.load_state_dict(p['model']);z=logits(model,image);fs=values(z,y)
            cache[k]=dict(base=[float(v.detach()) for v in fs],grads=[cpu(grad(v,params,allow_unused=True,retain_graph=True)) for v in fs])
        oldcross={v['anchor']:v for v in json.loads((replay/'CROSS.json').read_text())};cross=[]
        for k in [2,25]:
            for noise in [1,k]:
                cells={}
                for ti,sk in enumerate([0,k-1]):
                    p=state(sk);model.load_state_dict(p['model']);oldp=weights(p)
                    if (sk==0 and noise==1) or (sk==k-1 and noise==k):
                        gg=torch.load(replay/f'G{noise:02}.pt',weights_only=False,map_location='cuda')
                    else:
                        restore_rng_state(state(noise-1)['rng']);r=method.training_step(lb,None,3000+sk,3000+sk)
                        full=grad(r.total_loss,params,allow_unused=True);assert all(g is not None and torch.isfinite(g).all() for g in full)
                        n=torch.linalg.vector_norm(torch.stack(torch._foreach_norm(full,2)),2)
                        gg=torch._foreach_mul(full,(1/(n+1e-6)).clamp(max=1.))
                    for mi,mk in enumerate([0,k-1]):
                        op=state(mk)['trainer']['optimizer_state'];group=op['param_groups'][0]
                        assert group['params']==list(range(100)) and group['lr']==3e-4
                        with torch.no_grad():delta,_,_,_,info=candidate(params,gg,[op['state'][i] for i in group['params']],group)
                        a=[dot([None if g is None else g.cuda() for g in gs],delta) for gs in cache[sk]['grads']]
                        if ti==mi:
                            oldkey=('0' if sk==0 else '1')+('0' if noise==1 else '1')
                            reused=oldcross[k]['cells'][oldkey]
                            assert max(abs(v-w) for v,w in zip(a,reused['A']))<1e-7
                            q=reused['Q'];source='previous_state_noise_result'
                            if (sk==0 and noise==1) or (sk==k-1 and noise==k):
                                target=weights(state(sk+1));err=max(float((v-(b-c)).abs().max()) for v,b,c in zip(delta,target,oldp));assert err==0
                        else:
                            count('parameter_adam_cross',anchor=k,noise=noise,theta=sk,adam=mk)
                            with torch.no_grad():
                                for par,base,d in zip(params,oldp,delta):par.copy_(base+d)
                                znew=logits(model,image);fn=[float(v) for v in values(znew,y)]
                                pred=znew.argmax(1);comp=znew.masked_fill(F.one_hot(y,4).permute(0,3,1,2).bool(),-torch.inf).argmax(1)
                                margin=znew[:,2]-znew.gather(1,comp[:,None])[:,0]
                                detail=dict(class2_confusion=[int(((y==2)&(pred==c)).sum()) for c in range(4)],class2_margin=float(margin[y==2].mean()))
                            model.load_state_dict(p['model']);q=[v-b for v,b in zip(fn,cache[sk]['base'])];source='new_candidate'
                        row=dict(Q=q,A=a,R=[v-w for v,w in zip(q,a)],displacement_norm=norm(delta),A_per_unit_displacement=[v/norm(delta) for v in a],adam_old_steps=info['old_steps'],source=source)
                        if ti!=mi:row.update(detail)
                        cells[str(ti)+str(mi)]=row
                contrasts={}
                for term in ['Q','A','R']:
                    v={key:c[term] for key,c in cells.items()}
                    contrasts[term]=dict(total=[b-a for a,b in zip(v['00'],v['11'])],parameters=[b-a for a,b in zip(v['00'],v['10'])],adam=[b-a for a,b in zip(v['00'],v['01'])],interaction=[d-b-c+a for a,b,c,d in zip(v['00'],v['10'],v['01'],v['11'])])
                cross.append(dict(anchor=k,noise=noise,cells=cells,contrasts=contrasts))
        assert costs==dict(optimizer=0,main=0,golden=0,candidates=8,extra_reverse=10)
        write(root/'CROSS.json',cross);write(root/'CROSS_QUALIFICATION.json',dict(status='PASS',costs=costs.copy(),reused_diagonal_derivatives_equal=True,actual_diagonal_displacement_error=0))
        status('CROSS_COMPLETE')
        # The optimizer is first constructed after the zero-update section.
        optimizer=torch.optim.Adam(params,lr=3e-4,weight_decay=0);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        trainer=Trainer(method,optimizer=optimizer,scheduler=scheduler,device='cuda',amp=False,grad_clip_norm=1.)
        restore(initial,method,trainer);assert same(cpu(snapshot(method,trainer,3000)),initial)
        write(root/'RESTORE.json',dict(status='PASS',full_state_equal=True,parameter_tensors=100,semantic_tensors=40))
        mask=y==2;pad=F.pad(mask[:,None].float(),(1,1,1,1),mode='replicate')[:,0]
        interior=mask & pad[:,1:-1,:-2].bool() & pad[:,1:-1,2:].bool() & pad[:,:-2,1:-1].bool() & pad[:,2:,1:-1].bool()
        subsets=dict(all_class2=mask,near_decision=mask&(baseline['margin'].abs()<=1),near_background=mask&(baseline['background_margin'].abs()<=1),spatial_inner_boundary=mask&~interior)
        assert [int(v.sum()) for v in subsets.values()]==[490,74,51,293]
        phase=['main'];arm=['G'];idx=[0];alphas=[];arms={};finals={}
        def guard(opt,args,kwargs):
            assert costs['optimizer']<54 and costs[phase[0]]<(50 if phase[0]=='main' else 4)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            assert all(g['lr']==3e-4 for g in opt.param_groups)
            append(root/'attempts.jsonl',dict(attempt=costs['optimizer']+1,arm=arm[0],phase=phase[0],step=idx[0]))
        def receipt(opt,args,kwargs):
            costs['optimizer']+=1;costs[phase[0]]+=1
            append(root/'physical_updates.jsonl',dict(physical=costs['optimizer'],arm=arm[0],phase=phase[0],step=idx[0]))
        hooks=[optimizer.register_step_pre_hook(guard),optimizer.register_step_post_hook(receipt)]
        def advance(k,previous):
            assert same(capture_rng_state(),state(k-1)['rng'])
            old=[p.detach().clone() for p in params]
            r=trainer.train_step(lb,None,state=TrainerState(global_step=3000+k-1,site_step=3000+k-1))
            active['cursor']=3000+k
            proposed=[p.detach().clone() for p in params];posttrainer=cpu(trainer.state_dict());postrng=capture_rng_state()
            def apply(alpha):
                with torch.no_grad():
                    for par,a,b,sem in zip(params,old,proposed,flags):
                        par.copy_(b if not sem or alpha==1 else a if alpha==0 else a+alpha*(b-a))
            trials=[]
            if arm[0]=='G':
                chosen=None;f=None
                for alpha in [1.,.25,.0625]:
                    apply(alpha);count('guard_trial',arm=arm[0],phase=phase[0],step=k,alpha=alpha)
                    ff=frame(model,image,y);ok=accepts(ff['vector'],baseline['vector'])
                    trials.append(dict(alpha=alpha,accepted=ok,vector=ff['vector']))
                    if ok:chosen=alpha;f=ff;break
                if chosen is None:
                    chosen=0.;apply(0.);f=previous
                    # Only semantic parameters enter these outputs; alpha0 reuses them exactly.
                    assert all(torch.equal(p,a) for p,a,sem in zip(params,old,flags) if sem)
            else:
                chosen=alphas[k-1];apply(chosen);count('schedule_endpoint',arm=arm[0],phase=phase[0],step=k,alpha=chosen)
                f=previous if chosen==0 else frame(model,image,y)
            assert same(posttrainer,cpu(trainer.state_dict())) and same(postrng,capture_rng_state())
            assert all(torch.equal(p,b) for p,b,sem in zip(params,proposed,flags) if not sem)
            delta=[p.detach()-a for p,a in zip(params,old)]
            row=dict(arm=arm[0],phase=phase[0],step=k,alpha=chosen,loss_before=float(r.total_loss),actual_semantic_delta_norm=norm([d for d,sem in zip(delta,flags) if sem]),
                     nominal_semantic_delta_norm=norm([b-a for a,b,sem in zip(old,proposed,flags) if sem]),nonsemantic_delta_norm=norm([d for d,sem in zip(delta,flags) if not sem]),
                     guard_pass=accepts(f['vector'],baseline['vector']),vector=f['vector'],metrics=f['metrics'],trials=trials)
            return row,f
        def measure(f,previous):
            output=dict(metrics=f['metrics'],vector=f['vector'],margins={key:summaries(val,subsets) for key,val in [('overall',f['margin']),('background',f['background_margin'])]},transitions={})
            if previous is not None:
                for channel,pred in f['preds'].items():
                    output['transitions'][channel]={}
                    for group,ch in subsets.items():
                        output['transitions'][channel][group]={}
                        for i in [-1,0,1]:
                            a=previous['preds'][channel] if i==-1 else previous['preds'][channel][i:i+1]
                            b=pred if i==-1 else pred[i:i+1];m=ch if i==-1 else ch[i:i+1]
                            output['transitions'][channel][group]['aggregate' if i==-1 else f'image_{i}']={f'{aa}_to_{bb}':int((m&(a==aa)&(b==bb)).sum()) for aa in range(4) for bb in range(4)}
            return output
        for name in ['G','D']:
            arm[0]=name;phase[0]='main';out=root/name;out.mkdir(exist_ok=False);restore(initial,method,trainer)
            assert same(cpu(snapshot(method,trainer,3000)),initial)
            save(out/'S00.pt',initial);previous=baseline;rows=[];append(out/'metrics.jsonl',dict(step=0,**measure(baseline,None)))
            for k in range(1,26):
                idx[0]=k;active.update(cursor=3000+k-1,arm=name)
                row,f=advance(k,previous);rows.append(row)
                if name=='G':alphas.append(row['alpha'])
                append(out/'steps.jsonl',row);append(out/'metrics.jsonl',dict(step=k,**measure(f,previous)))
                save(out/f'S{k:02}.pt',cpu(snapshot(method,trainer,3000+k)))
                previous=f;status('TRAINING',arm=name,step=k)
            finals[name]=torch.load(out/'S25.pt',weights_only=False,map_location='cpu')
            phase[0]='golden';comparisons=[]
            for k in [1,25]:
                idx[0]=k;old=torch.load(out/f'S{k-1:02}.pt',weights_only=False,map_location='cpu');restore(old,method,trainer)
                oldf=baseline if k==1 else frame(model,image,y)
                row,f=advance(k,oldf);observed=cpu(snapshot(method,trainer,3000+k));expected=torch.load(out/f'S{k:02}.pt',weights_only=False,map_location='cpu')
                eq=same(observed,expected) and row['alpha']==rows[k-1]['alpha'] and row['vector']==rows[k-1]['vector'] and row['loss_before']==rows[k-1]['loss_before']
                comparisons.append(dict(step=k,full_state_and_decision_equal=eq));assert eq
            write(out/'GOLDEN.json',comparisons)
            semdelta=[finals[name]['model'][n]-initial['model'][n] for n,_ in named if semantic(n)]
            thdelta=[finals[name]['model'][n]-initial['model'][n] for n,_ in named if n.startswith(('anatomy.','structure.'))]
            arms[name]=dict(final_metrics=rows[-1]['metrics'],alphas=[r['alpha'] for r in rows],nonzero_steps=sum(r['alpha']>0 for r in rows),
                            semantic_path_length=sum(r['actual_semantic_delta_norm'] for r in rows),semantic_final_distance=norm(semdelta),generative_posterior_final_distance=norm(thdelta),
                            guard_pass_steps=sum(r['guard_pass'] for r in rows))
        for h in hooks:h.remove()
        hooks=[]
        def forbid(*a,**kw):raise AssertionError('training budget section complete')
        optimizer.step=forbid
        # Common-random-number objective readout; no new training seeds or backward.
        density={}
        for name,p in [('initial',initial),('G',finals['G']),('D',finals['D'])]:
            model.load_state_dict(p['model']);rows=[];saved=capture_rng_state()
            try:
                for draw in range(8):
                    count('joint_readout',state=name,draw=draw)
                    restore_rng_state(state(draw)['rng']);model.draw()
                    try:
                        with torch.no_grad():
                            bound,terms,diag=model.bound(image,y);kl=float(model.kl())
                            row={key:float(-val.mean()/128**2) for key,val in terms.items()}
                            row.update(total=float(-bound.mean()/128**2)+kl/(2*128**2),kl=kl)
                            assert all(torch.isfinite(torch.tensor(v)) for v in row.values());rows.append(row)
                    finally:model.clear_draw()
            finally:restore_rng_state(saved)
            density[name]=dict(mean={key:sum(v[key] for v in rows)/8 for key in rows[0]},draws=rows)
        write(root/'DENSITY.json',density)
        # Fixed initial phi and local x/m/c; only generative posterior changes.
        model.load_state_dict(initial['model']);saved=capture_rng_state()
        with torch.no_grad():anchor,_=model.proposal(image,sample=False)
        restore_rng_state(saved);anchor={k:v.detach() for k,v in anchor.items()}
        prob=F.one_hot(y,4).permute(0,3,1,2).float()*.99+.01/4
        patterns=[prob,torch.roll(prob,(0,32),(-2,-1)),torch.roll(prob,(32,0),(-2,-1)),torch.roll(prob,(64,64),(-2,-1)),prob[:,[0,2,3,1]],prob[:,[0,3,1,2]]]
        scores={}
        for name,p in [('initial',initial),('G',finals['G']),('D',finals['D'])]:
            model.load_state_dict(initial['model']);sd=model.state_dict()
            for key in sd:
                if key.startswith(('anatomy.','structure.')):sd[key]=p['model'][key]
            model.load_state_dict(sd);assert all(torch.equal(v.cpu(),initial['model'][key]) for key,v in model.state_dict().items() if not key.startswith(('anatomy.','structure.')))
            count('posterior_state',state=name);rows=[];saved=capture_rng_state()
            try:
                for draw in range(8):
                    seed_everything(80163+draw);model.draw();parts=[]
                    try:
                        with torch.no_grad():
                            for pattern in patterns:
                                h=anchor.copy();h['ell']=(pattern[:,1:]/pattern[:,:1]).log();_,terms,_=model.log_joint(image,h)
                                parts.append(torch.stack([terms['shape'],terms['logits'],terms['shape']+terms['logits']],-1)/128**2)
                        rows.append(torch.stack(parts,-1).cpu())
                    finally:model.clear_draw()
            finally:restore_rng_state(saved)
            scores[name]=torch.stack(rows)
        save(root/'PRIVATE_POSTERIOR_SCORES.pt',scores);post={}
        for name,score in scores.items():
            post[name]={}
            for ci,component in enumerate(['shape','logits','total']):
                post[name][component]={}
                for family,ix in [('spatial',[1,2,3]),('identity',[4,5])]:
                    m=score[:,:,ci,0]-score[:,:,ci,ix].mean(-1)
                    post[name][component][family]=dict(mean=float(m.mean()),per_image_mean=[float(v) for v in m.mean(0)],positive_draw_image_fraction=float((m>0).float().mean()))
        write(root/'POSTERIOR.json',post)
        assert costs['optimizer']==54 and costs['main']==50 and costs['golden']==4 and costs['extra_reverse']==10 and costs['candidates']<=143
        for file,key,n in [('physical_updates.jsonl','physical',54),('attempts.jsonl','attempt',54),('reverse_calls.jsonl','index',10),('candidate_evaluations.jsonl','index',costs['candidates'])]:
            assert [r[key] for r in map(json.loads,(root/file).read_text().splitlines())]==list(range(1,n+1))
        assert arms['G']['alphas']==arms['D']['alphas']
        for name,a in arms.items():
            a['retention']={}
            for channel,m in a['final_metrics'].items():
                macro=sum(map(sum,m['dice']))/6;base=sum(map(sum,baseline['metrics'][channel]['dice']))/6
                a['retention'][channel]=dict(macro_dice=macro,drop_pp=(base-macro)*100,all_six_ge_08=all(v>=.8 for r in m['dice'] for v in r),pass_gate=macro>=base-.02 and all(v>=.8 for r in m['dice'] for v in r))
        result=dict(status='COMPLETE',costs=costs,started=start,finished=time.time(),baseline=baseline['metrics'],arms=arms,original_ledgers_unchanged=len((origin/'physical_updates.jsonl').read_text().splitlines())==1204 and len((replay/'physical_updates.jsonl').read_text().splitlines())==27,api_calls=model.calls)
        write(root/'RESULTS.json',result);status('COMPLETE');write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        if active and 'trainer' in locals():save(root/'failure_partial.pt',cpu(snapshot(method,trainer,active['cursor'])))
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));status('FAILED');write(root/'EXIT.json',dict(exit_code=1));raise
    finally:
        for h in hooks:h.remove()


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))

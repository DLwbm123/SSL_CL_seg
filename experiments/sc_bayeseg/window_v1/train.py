"""Short paired semantic directions with true output and objective admission."""
import copy,json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.engine.trainer import Trainer,TrainerState
from lcrseg.engine.checkpoint import capture_rng_state
from care_seg_study_v1.run import write,append,save
from sc_bayeseg.split_v1.model import Model,semantic
from sc_bayeseg.jointwrite_v1.run import Adapter
from sc_bayeseg.run import snapshot,restore,same
from sc_bayeseg.finite_replay_v1.run import cpu
from sc_bayeseg.geometry_v1.run import losses,flat
from sc_bayeseg.geometry_v1.geometry import solve,unit
from sc_bayeseg.guard_v1.run import frame,accepts
from sc_bayeseg.window_v1.common import storage,objective,joint_pass


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);start=time.time();costs=dict(optimizer=0,main=0,golden=0,extra_reverse=0,candidate_evaluations=0);active={};hooks=[];oldclip=torch.nn.utils.clip_grad_norm_
    def status(phase,**kw):write(root/'STATUS.json',dict(phase=phase,costs=costs.copy(),space=storage(s),**kw))
    def checkpoint(path,value):storage(s);save(path,cpu(value))
    try:
        assert json.loads((Path(s['finite'])/'RESULTS.json').read_text())['training_qualified']
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        initial=torch.load(Path(s['guard'])/'G'/'S00.pt',weights_only=False,map_location='cpu')
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text());lb=collate_labeled([ds[i] for i in sel['train_indices']]).to('cuda')
        image=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False);y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        model=Model().cuda();method=Adapter(model,'joint');method.nl=2;method.begin_site('Siemens',None,3600)
        named=list(model.named_parameters());params=[p for _,p in named];flags=[semantic(n) for n,_ in named];sp=[p for p,yes in zip(params,flags) if yes];assert len(sp)==40
        optimizer=torch.optim.Adam(params,lr=3e-4,weight_decay=0);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        trainer=Trainer(method,optimizer=optimizer,scheduler=scheduler,device='cuda',amp=False,grad_clip_norm=1.)
        restore(initial,method,trainer);assert same(cpu(snapshot(method,trainer,3000)),initial)
        baseline=frame(model,image,y);radius=json.loads((Path(s['geometry'])/'RESULTS.json').read_text())['states']['0']['joint'][0]['adam_semantic_norm']/16
        arm=['N'];phase=['main'];index=[0];stash={};results={};goldens={}
        def before(opt,a,kw):
            assert costs['optimizer']<54 and all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            assert all(g['lr']==3e-4 for g in opt.param_groups)
            append(root/'attempts.jsonl',dict(attempt=costs['optimizer']+1,arm=arm[0],phase=phase[0],step=index[0]))
        def after(opt,a,kw):
            costs['optimizer']+=1;costs[phase[0]]+=1
            append(root/'physical_updates.jsonl',dict(physical=costs['optimizer'],arm=arm[0],phase=phase[0],step=index[0]))
        hooks=[optimizer.register_step_pre_hook(before),optimizer.register_step_post_hook(after)]
        def clip(pp,maxnorm,*a,**kw):
            pp=list(pp);stash['semantic_gradient']=flat([p.grad for p,yes in zip(params,flags) if yes],sp)
            n=oldclip(pp,maxnorm,*a,**kw);stash['gradient_norm']=float(n);return n
        torch.nn.utils.clip_grad_norm_=clip
        def advance(k):
            old=cpu(snapshot(method,trainer,3000+k-1));assert same(old['rng'],torch.load(Path(s['replay'])/f'S{k-1:02}.pt',weights_only=False,map_location='cpu')['rng'])
            f0=frame(model,image,y);j0=objective(model,image,y,s['replay']);C=None
            if arm[0]=='C':
                vals=losses(model,image,y);grads=[]
                for i,v in enumerate(vals):
                    assert costs['extra_reverse']<432;costs['extra_reverse']+=1;append(root/'reverse_calls.jsonl',dict(index=costs['extra_reverse'],arm=arm[0],phase=phase[0],step=k,constraint=i))
                    grads.append(flat(torch.autograd.grad(v,sp,allow_unused=True,retain_graph=i<15),sp))
                C=np.stack(grads)
            r=trainer.train_step(lb,None,state=TrainerState(global_step=3000+k-1,site_step=3000+k-1));active['cursor']=3000+k
            proposed={n:p.detach().clone() for n,p in named};opstate=cpu(trainer.state_dict());postrng=capture_rng_state()
            g=stash['semantic_gradient'];valid=True;geometry=None
            if arm[0]=='N':d,_=unit(-g)
            else:
                geometry,ds=solve(C,g[None]);valid=geometry['strict']['strict_witness'];d=ds['strict']
            chosen=None;f=None;j=None;trials=[];rollback=False
            for alpha in ([1.,.25,.0625,0.] if valid else [0.]):
                with torch.no_grad():
                    pos=0
                    for n,p in named:
                        if semantic(n):
                            base=old['model'][n].to(p.device)
                            if alpha==0:p.copy_(base)
                            else:
                                dd=torch.as_tensor(d[pos:pos+p.numel()].reshape(p.shape),device=p.device,dtype=p.dtype);p.copy_(base+radius*alpha*dd)
                            pos+=p.numel()
                        else:p.copy_(proposed[n])
                costs['candidate_evaluations']+=1
                ff=f0 if alpha==0 else frame(model,image,y);jj=objective(model,image,y,s['replay']);ok=accepts(ff['vector'],f0['vector']) and joint_pass(jj,j0)
                tr=dict(index=costs['candidate_evaluations'],arm=arm[0],phase=phase[0],step=k,alpha=alpha,semantic_pass=accepts(ff['vector'],f0['vector']),joint_pass=joint_pass(jj,j0),joint=jj,vector=ff['vector'])
                append(root/'candidate_evaluations.jsonl',tr);trials.append(tr)
                if ok:chosen=alpha;f=ff;j=jj;break
            if chosen is None:
                model.load_state_dict(old['model']);chosen=0.;f=f0;j=j0;rollback=True
            assert same(opstate,cpu(trainer.state_dict())) and same(postrng,capture_rng_state())
            semnorm=sum(float((p.detach().cpu()-old['model'][n]).double().square().sum()) for n,p in named if semantic(n))**.5
            row=dict(arm=arm[0],phase=phase[0],step=k,alpha=chosen,full_model_rollback=rollback,actual_semantic_delta_norm=semnorm,loss_before=float(r.total_loss),gradient_norm=stash['gradient_norm'],joint_before=j0,joint_after=j,metrics=f['metrics'],vector=f['vector'],trials=trials,
                     geometry_summary=None if geometry is None else dict(strict_witness=valid,worst_normalized_derivative=geometry['strict']['worst_normalized_derivative'],convex_hull_gap=geometry['strict']['convex_hull_optimality_gap']))
            return row
        for name in ['N','C']:
            arm[0]=name;phase[0]='main';out=root/name;out.mkdir(exist_ok=False);restore(initial,method,trainer)
            checkpoint(out/'S00.pt',initial);rows=[]
            append(out/'metrics.jsonl',dict(step=0,metrics=baseline['metrics'],vector=baseline['vector']))
            for k in range(1,26):
                index[0]=k;active.update(cursor=3000+k-1,arm=name)
                row=advance(k);rows.append(row);append(out/'steps.jsonl',row);checkpoint(out/f'S{k:02}.pt',snapshot(method,trainer,3000+k))
                append(out/'metrics.jsonl',dict(step=k,metrics=row['metrics'],vector=row['vector'],joint=row['joint_after']))
                status('TRAINING',arm=name,step=k)
            phase[0]='golden';checks=[]
            for k in [1,25]:
                index[0]=k;restore(torch.load(out/f'S{k-1:02}.pt',weights_only=False,map_location='cpu'),method,trainer)
                row=advance(k);expected=torch.load(out/f'S{k:02}.pt',weights_only=False,map_location='cpu')
                equal=same(cpu(snapshot(method,trainer,3000+k)),expected) and row['alpha']==rows[k-1]['alpha'] and row['vector']==rows[k-1]['vector'] and row['joint_after']==rows[k-1]['joint_after']
                checks.append(dict(step=k,full_state_and_decision_equal=equal));assert equal
            write(out/'GOLDEN.json',checks);goldens[name]=checks
            final=torch.load(out/'S25.pt',weights_only=False,map_location='cpu');model.load_state_dict(final['model'])
            j8=objective(model,image,y,s['replay'],tuple(range(1,9)));jd=sum(j8)/8
            a=dict(final_metrics=rows[-1]['metrics'],nonzero_semantic_steps=sum(v['actual_semantic_delta_norm']>0 for v in rows),full_rollbacks=sum(v['full_model_rollback'] for v in rows),alphas=[v['alpha'] for v in rows],semantic_path=sum(v['actual_semantic_delta_norm'] for v in rows),common8_joint=j8,common8_joint_mean=jd)
            a['retention']={ch:dict(macro_dice=sum(map(sum,m['dice']))/6,drop_pp=(sum(map(sum,baseline['metrics'][ch]['dice']))-sum(map(sum,m['dice'])))*100/6,all_six_ge_08=all(v>=.8 for row in m['dice'] for v in row)) for ch,m in a['final_metrics'].items()}
            results[name]=a
        torch.nn.utils.clip_grad_norm_=oldclip
        for h in hooks:h.remove()
        hooks=[];model.load_state_dict(initial['model']);j8=objective(model,image,y,s['replay'],tuple(range(1,9)))
        assert costs['optimizer']==54 and costs['extra_reverse']==432 and costs['main']==50 and costs['golden']==4
        complete=dict(status='COMPLETE',costs=costs,started=start,finished=time.time(),radius=radius,baseline=baseline['metrics'],initial_common8_joint=j8,initial_common8_joint_mean=sum(j8)/8,arms=results,golden=goldens,space=storage(s),pending_readout='fixed-initial-phi generative posterior readout and next-stage protocol remain for monitoring')
        write(root/'RESULTS.json',complete);status('COMPLETE');write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        if active and 'trainer' in locals():checkpoint(root/'failure_partial.pt',snapshot(method,trainer,active['cursor']))
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));status('FAILED');write(root/'EXIT.json',dict(exit_code=1));raise
    finally:
        torch.nn.utils.clip_grad_norm_=oldclip
        for h in hooks:h.remove()


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))

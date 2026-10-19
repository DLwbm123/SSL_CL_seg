"""Budgeted CE->joint conditional test with fixed-proposal posterior swap."""
import copy
import json
import os
import time
import traceback
from pathlib import Path

import torch
from torch.nn import functional as F
from lcrseg.contracts import MethodStepOutput
from lcrseg.data import H5LabeledDataset, collate_labeled
from lcrseg.engine.trainer import Trainer, TrainerState, seed_everything
from care_seg_study_v1.run import write, append, save
from sc_bayeseg.model import Model
from sc_bayeseg.method import Method
from sc_bayeseg.run import snapshot, restore, same
from sc_bayeseg.diagnostic_v1.run import DiagnosticMethod
from sc_bayeseg.paired_lr_v1.run import evaluate


class Adapter(DiagnosticMethod):
    def __init__(self,model,mode):
        super().__init__(model,230,mode)
        self.attribute_enabled=False
        self.attribute_grads=None
        self.extra_reverse_calls=0

    def training_step(self,lb,ub,global_step,site_step):
        if self.mode!='joint':return super().training_step(lb,None,global_step,site_step)
        image=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False)
        y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        self.model.draw()
        try:
            bound,terms,diag=self.model.bound(image,y)
            kl=self.model.kl();nll=-self.nl*bound.mean()
            total=(nll+nll*0+kl)/(self.nl*128**2)
            diag.update(global_kl=float(kl.detach()),label_ce_per_pixel=float(-terms['labels'].mean().detach()/128**2),
                        labeled_slices=self.nl,unlabeled_slices=0)
            diag.update({f'nll_{k}_per_pixel':float(-v.detach().mean()/128**2) for k,v in terms.items()})
            diag['logq_per_pixel']=float((sum(terms.values())-bound).detach().mean()/128**2)
            if self.attribute_enabled:
                named=[(n,p) for n,p in self.model.named_parameters() if not n.startswith(('anatomy.','structure.'))]
                self.attribute_grads={}
                for key in ['labels','image']:
                    grads=torch.autograd.grad(-terms[key].mean()/128**2,[p for _,p in named],allow_unused=True,retain_graph=True)
                    self.extra_reverse_calls+=1
                    assert all(g is None or torch.isfinite(g).all() for g in grads)
                    self.attribute_grads[key]={n:None if g is None else g.detach().clone() for (n,_),g in zip(named,grads)}
            return MethodStepOutput(total,dict(joint_L=nll/(self.nl*128**2),posterior_kl=kl/(self.nl*128**2)),diag)
        finally:self.model.clear_draw()


def displacement(model,before,gradients):
    heads={'shape_encoder.7','logit_encoder.7','appearance_encoder.5','code_encoder.4'}
    groups={}
    for n,p in model.named_parameters():
        if n not in before:continue
        delta=p.detach()-before[n];parts=[('features',slice(None))]
        if n.rsplit('.',1)[0] in heads:parts=[('mean_head',slice(0,len(delta)//2)),('logsd_head',slice(len(delta)//2,None))]
        for kind,sl in parts:
            key=n.split('.')[0]+'.'+kind
            row=groups.setdefault(key,dict(displacement_squared_norm=0.,label_dot_displacement=0.,image_dot_displacement=0.))
            row['displacement_squared_norm']+=float(delta[sl].square().sum())
            for factor,label in [('labels','label'),('image','image')]:
                g=gradients[factor][n]
                if g is not None:row[label+'_dot_displacement']+=float((g[sl]*delta[sl]).sum())
    return groups


def stats(value):
    return dict(mean=float(value.mean()),min_sample_mean=float(value.mean(0).min()),
                min_draw_group_mean=float(value.mean(1).min()),max_draw_group_mean=float(value.mean(1).max()),
                positive_draw_sample_fraction=float((value>0).float().mean()))


def posterior_swap(root,payload,trained,lb4,ran_joint):
    image=F.interpolate(lb4.image,(128,128),mode='bilinear',align_corners=False)
    y=F.interpolate(lb4.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
    model=Model().cuda();model.load_state_dict(payload['model']);fixed=copy.deepcopy(model.state_dict())
    with torch.no_grad():anchor,_=model.proposal(image,sample=False)
    anchor={k:v.detach() for k,v in anchor.items()}
    p=F.one_hot(y,4).permute(0,3,1,2).float()*.99+.01/4
    candidates=[p,torch.roll(p,(0,32),(-2,-1)),torch.roll(p,(32,0),(-2,-1)),torch.roll(p,(64,64),(-2,-1)),p[:,[0,2,3,1]],p[:,[0,3,1,2]]]
    names=['GT','roll_0_32','roll_32_0','roll_64_64','cycle_123','cycle_132']
    for other in candidates[1:4]:
        for k in range(4):assert torch.equal((other.argmax(1)==k).sum((-2,-1)),(p.argmax(1)==k).sum((-2,-1)))
    all_scores={};counter=[]
    for state,weights in [('q0',payload['model']),('qJ' if ran_joint else 'qC',trained['model'])]:
        model.load_state_dict(fixed)
        changed=model.state_dict()
        for k in changed:
            if k.startswith(('anatomy.','structure.')):changed[k]=weights[k]
        model.load_state_dict(changed)
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in fixed.items() if not k.startswith(('anatomy.','structure.')))
        before=copy.deepcopy(model.state_dict());component={k:[] for k in ['shape','logits','total']}
        for draw in range(8):
            seed_everything(80163+draw);model.draw();scored={k:[] for k in component}
            for candidate in candidates:
                h={k:v for k,v in anchor.items()};h['ell']=(candidate[:,1:]/candidate[:,:1]).log()
                with torch.no_grad():_,terms,_=model.log_joint(image,h)
                assert all(torch.isfinite(terms[k]).all() for k in ['shape','logits'])
                for k in ['shape','logits']:scored[k].append(terms[k]/128**2)
                scored['total'].append((terms['shape']+terms['logits'])/128**2)
            for k in component:component[k].append(torch.stack(scored[k],-1).cpu())
            model.clear_draw()
        assert same(before,model.state_dict())
        all_scores[state]={k:torch.stack(v) for k,v in component.items()}
    assert model.calls['proposal_images']==4 and model.calls['density_images']==384 and model.calls['generator_draws']==16
    post='qJ' if ran_joint else 'qC'
    result=dict(status='COMPLETE',J_available=ran_joint,fixed_phi='pretraining L3',local_anchor='pretraining phi mean, fixed x/m/c',
                label_likelihood_included=False,refinement_used=False,SNIS_used=False,candidates=names,calls=model.calls.copy(),groups={})
    for group,sl in [('train',slice(0,2)),('heldout',slice(2,4))]:
        rows={}
        for component in ['shape','logits','total']:
            row={}
            for family,idx in [('spatial',[1,2,3]),('identity',[4,5])]:
                margins={}
                for state,scores in all_scores.items():
                    a=scores[component][:,sl,:];margin=a[:,:,0]-a[:,:,idx].mean(-1)
                    margins[state]=margin
                    row[family+'_'+state]=stats(margin)
                row[family+'_change']=stats(margins[post]-margins['q0'])
            rows[component]=row
        result['groups'][group]=rows
    write(root/'POSTERIOR_SWAP.json',result)
    write(root/'PRIVATE_SWAP_SCORES.json',{s:{k:v.tolist() for k,v in d.items()} for s,d in all_scores.items()})
    del model;torch.cuda.empty_cache()
    return result


def run(settings):
    root=Path(settings['output']);assert json.loads((root/'OVERLAY_REVIEW.json').read_text())['status']=='PASS_VISUAL_SCREEN'
    with (root/'RUN_STARTED.json').open('x') as f:json.dump(dict(time=time.time()),f)
    write(root/'CONTEXT.json',settings)
    costs=dict(diagnostic=0,engineering=0);active={};started=time.time()
    try:
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        origin=Path(settings['origin']);payload=torch.load(origin/'L3'/'final.pt',map_location='cpu',weights_only=False)
        assert payload['cursor']==1800 and payload['population']==dict(labeled=230,unlabeled=0)
        original=json.loads((origin/'L3'/'metrics.jsonl').read_text().splitlines()[-1])
        data=H5LabeledDataset(settings['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        assert len(data)==230
        sel=json.loads((root/'PRIVATE_SELECTION.json').read_text());assert len(set(sel['identities']))==4
        lb=collate_labeled([data[i] for i in sel['train_indices']]).to('cuda')
        lb4=collate_labeled([data[i] for i in sel['indices']]).to('cuda')
        image=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False)
        y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        results={}
        def arm(name,mode):
            out=root/name;out.mkdir(exist_ok=False)
            seed_everything(163);model=Model().cuda();method=Adapter(model,mode);method.begin_site('Siemens',None,2400)
            optimizer=torch.optim.Adam(model.parameters(),lr=3e-4,weight_decay=0)
            scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
            trainer=Trainer(method,optimizer=optimizer,scheduler=scheduler,device='cuda',amp=False,grad_clip_norm=1.)
            restore(payload,method,trainer)
            assert same(payload['model'],model.state_dict()) and same(payload['trainer'],trainer.state_dict())
            method.nl=2
            baseline=evaluate(model,image,y)
            assert all(baseline[k]==original[k] for k in baseline)
            initial=snapshot(method,trainer,1800)
            write(out/'RESTORE.json',dict(status='PASS',source_cursor=1800,model_equal=True,adam_and_scheduler_equal=True,
                  baseline_predictions_equal=True,original_population=payload['population'],current_population=initial['population'],lr=3e-4))
            if mode=='joint':
                first=method.training_step(lb,None,1800,1800);restore(initial,method,trainer)
                second=Method.training_step(method,lb,None,1800,1800)
                assert torch.equal(first.total_loss,second.total_loss)
                restore(initial,method,trainer)
                write(out/'OBJECTIVE_EQUIVALENCE.json',dict(status='PASS',original_Method_equal=True,forward_calls=2,
                                                           proposal_images=4,density_images=4,generator_draws=2,optimizer_updates=0))
            prior={k:v.clone() for k,v in model.state_dict().items() if 'prior_' in k or 'initial_' in k}
            active.update(method=method,trainer=trainer,out=out,cursor=1800)
            append(out/'metrics.jsonl',dict(added_step=0,cumulative_step=1800,**baseline))
            local=dict(diagnostic=0,engineering=0);phase=['engineering'];norms=[]
            def guard(opt,args,kwargs):
                key=phase[0]
                assert sum(costs.values())<1204 and costs[key]<(4 if key=='engineering' else 1200)
                assert local[key]<(2 if key=='engineering' else 600)
                assert all(g['lr']==3e-4 for g in opt.param_groups)
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            def receipt(opt,args,kwargs):
                key=phase[0];costs[key]+=1;local[key]+=1
                append(root/'physical_updates.jsonl',dict(physical=sum(costs.values()),kind=key,kind_physical=costs[key],arm=name,local=local[key],time=time.time()))
            pre=optimizer.register_step_pre_hook(guard);post=optimizer.register_step_post_hook(receipt)
            original_clip=torch.nn.utils.clip_grad_norm_
            def clip(parameters,max_norm,*args,**kwargs):
                v=original_clip(parameters,max_norm,*args,**kwargs);norms.append(float(v));return v
            torch.nn.utils.clip_grad_norm_=clip
            try:
                def step(index):
                    r=trainer.train_step(lb,None,state=TrainerState(global_step=index,site_step=index))
                    assert scheduler.get_last_lr()==[3e-4] and optimizer.param_groups[0]['lr']==3e-4
                    return r
                a=step(1800);expected=snapshot(method,trainer,1801);restore(initial,method,trainer)
                b=step(1800);assert torch.equal(a.total_loss,b.total_loss) and same(expected,snapshot(method,trainer,1801))
                restore(initial,method,trainer)
                write(out/'QUALIFICATION.json',dict(status='PASS',engineering_updates=2,full_state_golden_equal=True))
                phase[0]='diagnostic'
                for added in range(1,601):
                    method.attribute_enabled=mode=='joint' and added in (1,100,600)
                    before={n:p.detach().clone() for n,p in model.named_parameters() if not n.startswith(('anatomy.','structure.'))} if method.attribute_enabled else None
                    r=step(1800+added-1);active['cursor']=1800+added
                    if before is not None:
                        append(out/'UPDATE_ATTRIBUTION.jsonl',dict(added_step=added,groups=displacement(model,before,method.attribute_grads)))
                        method.attribute_grads=None
                    if added==1 or added%25==0:
                        row=dict(arm=name,added_step=added,cumulative_step=1800+added,loss=float(r.total_loss.detach()),
                                 gradient_norm_before_clip=norms[-1],**r.scalars)
                        append(out/'telemetry.jsonl',row);write(root/'STATUS.json',dict(status='RUNNING',costs=costs.copy(),**row))
                    if added in (25,100,300,600):append(out/'metrics.jsonl',dict(added_step=added,cumulative_step=1800+added,**evaluate(model,image,y)))
                    if added%100==0:save(out/'latest.pt',snapshot(method,trainer,1800+added))
                assert local==dict(diagnostic=600,engineering=2)
                assert all(torch.equal(v,model.state_dict()[k]) for k,v in prior.items())
                assert method.extra_reverse_calls==(6 if mode=='joint' else 0)
                save(out/'final.pt',snapshot(method,trainer,2400))
                metric=json.loads((out/'metrics.jsonl').read_text().splitlines()[-1])
                metric.update(status='COMPLETE',mode=mode,updates=local,prior_unchanged=True,calls=model.calls.copy(),
                              extra_reverse_calls=method.extra_reverse_calls,main_pass=all(v>=.8 for row in metric['stochastic_dice'] for v in row),
                              direct_pass=all(v>=.8 for row in metric['direct_dice'] for v in row),
                              clipping_fraction=sum(v>1 for v in norms)/len(norms),gradient_norm_max=max(norms))
                write(out/'RESULTS.json',metric);results[name]=metric
            finally:
                torch.nn.utils.clip_grad_norm_=original_clip;pre.remove();post.remove()
            active.clear();del model,method,trainer;torch.cuda.empty_cache()
            return out/'final.pt'
        final=arm('C','stochastic_ce');passed=results['C']['main_pass']
        write(root/'BRANCH_DECISION.json',dict(C_pass=passed,threshold=.8,chosen=['J'] if passed else [],basis='six terminal four-draw Dice'))
        if passed:final=arm('J','joint')
        write(root/'STATUS.json',dict(status='POSTERIOR_SWAP',costs=costs))
        trained=torch.load(final,map_location='cpu',weights_only=False)
        swap=posterior_swap(root,payload,trained,lb4,passed)
        complete=dict(status='COMPLETE',started=started,finished=time.time(),costs=costs,results=results,J_run=passed,
                      posterior_swap_completed=True)
        write(root/'RESULTS.json',complete);write(root/'STATUS.json',complete)
    except BaseException:
        failure=dict(status='FAILED',costs=costs,traceback=traceback.format_exc())
        if active:
            save(active['out']/'failure_partial.pt',snapshot(active['method'],active['trainer'],active['cursor']))
            failure['partial_saved']=True
        write(root/'FAILURE.json',failure);write(root/'STATUS.json',failure)
        raise


if __name__=='__main__':
    settings=json.loads(Path(os.environ['TASK_SETTINGS']).read_text());root=Path(settings['output'])
    if (root/'RUN_STARTED.json').exists():raise FileExistsError('create-only training already started')
    code=1
    try:run(settings);code=0
    finally:write(root/'EXIT.json',dict(exit_code=code,time=time.time()))

"""Create-only source diagnostic; all actual optimizer calls charged."""
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
from sc_bayeseg.model import Model
from sc_bayeseg.method import Method
from sc_bayeseg.run import snapshot, restore, same
from care_seg_study_v1.run import write, append, save


def dice(pred, target):
    return [[float((2*((pred[i]==k)&(target[i]==k)).sum()) /
                   ((pred[i]==k).sum()+(target[i]==k).sum()))
             if int((pred[i]==k).sum()+(target[i]==k).sum()) else 1.
             for k in (1,2,3)] for i in range(len(target))]


def logits(h):
    return torch.cat([torch.zeros_like(h['ell'][:,:1]),h['ell']],1)


class DiagnosticMethod(Method):
    def __init__(self, model, nl, mode):
        super().__init__(model,nl,0)
        self.mode=mode

    def training_step(self, labeled_batch, unlabeled_batch, global_step, site_step):
        if self.mode=='joint':
            return super().training_step(labeled_batch,None,global_step,site_step)
        image=F.interpolate(labeled_batch.image,(128,128),mode='bilinear',align_corners=False)
        y=F.interpolate(labeled_batch.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        self.model.draw()
        try:
            h,_=self.model.proposal(image,sample=self.mode!='mean_ce')
            z=logits(h)
            loss=F.cross_entropy(z,y)
            scalars={f'predicted_fraction_{k}':float((z.detach().argmax(1)==k).float().mean()) for k in range(4)}
            scalars['label_ce_per_pixel']=float(loss.detach())
            return MethodStepOutput(loss,{'label_ce':loss},scalars)
        finally:
            self.model.clear_draw()


def evaluate(model, image, y):
    state=torch.get_rng_state(); cuda=torch.cuda.get_rng_state_all()
    try:
        seed_everything(50163)
        with torch.no_grad():
            probabilities=[]
            for _ in range(4):
                h,_=model.proposal(image)
                probabilities.append(logits(h).softmax(1))
            p=torch.stack(probabilities).mean(0)
            h,_=model.proposal(image,sample=False)
            direct=logits(h).softmax(1)
            return dict(stochastic_dice=dice(p.argmax(1),y),direct_dice=dice(direct.argmax(1),y),
                        stochastic_ce=float(F.nll_loss(p.clamp_min(1e-30).log(),y)),
                        direct_ce=float(F.nll_loss(direct.clamp_min(1e-30).log(),y)),
                        stochastic_background_fraction=float((p.argmax(1)==0).float().mean()),
                        direct_background_fraction=float((direct.argmax(1)==0).float().mean()))
    finally:
        torch.set_rng_state(state);torch.cuda.set_rng_state_all(cuda)


def zero_audit(settings, root):
    dataset=H5LabeledDataset(settings['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
    assert len(dataset)==230
    assert all(r['cohort']=='canonical320' and r['evaluation_eligible']=='true' for r in dataset.records)
    totals=torch.zeros(4,dtype=torch.long); small=torch.zeros_like(totals)
    empty=0; selected=[]; patients=set(); native_dice=[]; roundtrip=[]
    order=sorted(range(len(dataset)),key=lambda i:(dataset.samples[i].row['case_id'],dataset.samples[i].slice_index))
    for i in order:
        item=dataset[i]; y=item['label']; image=item['image']
        assert torch.isfinite(image).all() and y.shape==image.shape[-2:]
        assert int(y.min())>=0 and int(y.max())<=3
        ys=F.interpolate(y[None,None].float(),(128,128),mode='nearest')[0,0].long()
        totals+=torch.bincount(y.flatten(),minlength=4); cnt=torch.bincount(ys.flatten(),minlength=4);small+=cnt
        empty+=int(not bool((ys>0).any()))
        if len(selected)<2 and item['patient_id'] not in patients and bool((cnt[1:]>=64).all()):
            selected.append(i);patients.add(item['patient_id'])
            oracle=F.one_hot(y,4).permute(2,0,1).float().clamp_min(1e-30).log().argmax(0)
            native_dice.append(dice(oracle[None],y[None])[0])
            up=F.interpolate(F.one_hot(ys,4).permute(2,0,1)[None].float(),y.shape,mode='bilinear',align_corners=False).argmax(1)
            roundtrip.append(dice(up,y[None])[0])
    assert len(selected)==2 and bool((small[1:]>0).all())
    assert all(v==1 for row in native_dice for v in row)
    lb=collate_labeled([dataset[i] for i in selected]).to('cuda')
    image=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False)
    y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
    write(root/'PRIVATE_SELECTION.json',dict(indices=selected,identities=[dataset[i]['patient_id'] for i in selected]))
    result=dict(status='PASS',labeled_slices=len(dataset),class_pixels_native=totals.tolist(),class_pixels_128=small.tolist(),
                empty_foreground_slices_128=empty,selected_class_pixels_128=[torch.bincount(a.flatten(),minlength=4).tolist() for a in y.cpu()],
                native_oracle_dice=native_dice,roundtrip_dice=roundtrip,gradient_states={},optimizer_updates=0)
    reverse=0
    for state in ('initial','source_L_final'):
        seed_everything(163);model=Model().cuda()
        if state!='initial':model.load_state_dict(torch.load(settings['source'],map_location='cpu',weights_only=False)['model'])
        before=copy.deepcopy(model.state_dict()); records=[]
        params=list(model.parameters());names=[n for n,_ in model.named_parameters()]
        for draw in range(4):
            seed_everything(60163+draw);model.draw();h,logq=model.proposal(image);joint,terms,p=model.log_joint(image,h,y)
            expected=p.gather(1,y[:,None]).clamp_min(1e-30).log().sum((1,2,3))
            torch.testing.assert_close(terms['labels'],expected,rtol=1e-4,atol=.01)
            factors={k:-v.mean()/(128**2) for k,v in terms.items()}
            factors['entropy']=logq.mean()/(128**2)
            factors['global_kl']=model.kl()/(230*128**2)
            grads={}
            for key,value in factors.items():
                grads[key]=torch.autograd.grad(value,params,allow_unused=True,retain_graph=True);reverse+=1
                assert all(g is None or torch.isfinite(g).all() for g in grads[key])
            assert sum(float(g.square().sum()) for g in grads['labels'] if g is not None)>0
            norms={};cosines={}
            for key,gg in grads.items():
                norms[key]={}
                for prefix in ('shape_encoder','logit_encoder','appearance_encoder','code_encoder','anatomy','structure'):
                    norms[key][prefix]=sum(float(g.square().sum()) for n,g in zip(names,gg) if n.startswith(prefix) and g is not None)**.5
                norm=sum(float(g.square().sum()) for g in gg if g is not None)**.5
                lab=grads['labels'];ln=sum(float(g.square().sum()) for g in lab if g is not None)**.5
                dot=sum(float((a*b).sum()) for a,b in zip(gg,lab) if a is not None and b is not None)
                cosines[key]=dot/(norm*ln) if norm*ln else None
            assert norms['shape']['logit_encoder']>0, 'shape factor lost semantic gradient'
            altered={k:v.detach() for k,v in h.items()}
            if draw%2==0:
                prob=F.one_hot(y,4).permute(0,3,1,2).float()*.99+.01/4
                altered['ell']=(prob[:,1:]/prob[:,:1]).log();kind='visible_GT_logits'
            else:altered['ell']=torch.roll(altered['ell'],64,-1);kind='spatial_roll_64'
            with torch.no_grad():_,other,_=model.log_joint(image,altered,y)
            records.append(dict(draw=draw,factors={k:float(v.detach()) for k,v in factors.items()},gradient_norms=norms,
                                cosine_to_label=cosines,intervention=kind,
                                conditional_nll_change_per_pixel={k:float((terms[k].detach()-other[k]).mean()/(128**2)) for k in terms}))
            model.clear_draw()
        assert same(before,model.state_dict())
        result['gradient_states'][state]=dict(records=records,calls=model.calls.copy(),persistent_state_unchanged=True)
        del model;torch.cuda.empty_cache()
    assert reverse==64
    result['reverse_calls']=reverse
    write(root/'ZERO_AUDIT.json',result)
    return lb,image,y


def run(settings):
    root=Path(settings['output']);root.mkdir(exist_ok=False)
    write(root/'CONTEXT.json',settings)
    counts=dict(diagnostic=0,engineering=0);active={};started=time.time()
    try:
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        write(root/'STATUS.json',dict(status='ZERO_AUDIT'))
        lb,image,y=zero_audit(settings,root)
        results={}
        def arm(name,mode,source=None):
            out=root/name;out.mkdir(exist_ok=False)
            seed_everything(163);model=Model().cuda()
            if source:model.load_state_dict(torch.load(source,map_location='cpu',weights_only=False)['model'])
            method=DiagnosticMethod(model,230,mode);method.begin_site('Siemens',None,600)
            opt=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=0)
            scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
            trainer=Trainer(method,optimizer=opt,scheduler=scheduler,device='cuda',amp=False,grad_clip_norm=1.)
            active.update(method=method,trainer=trainer,out=out,cursor=0)
            # A/B/D share both initial model and training random stream; C changes only its initial model.
            seed_everything(70163)
            prior={k:v.clone() for k,v in model.state_dict().items() if 'prior_' in k or 'initial_' in k}
            if source:write(out/'FORK.json',dict(source='A',full_model_equal=same(torch.load(source,map_location='cpu',weights_only=False)['model'],model.state_dict()),optimizer_reset=True,prior_recursion=False))
            initial=snapshot(method,trainer,0);local={'engineering':0,'diagnostic':0};phase=['engineering'];norms=[]
            def guard(optimizer,args,kwargs):
                assert counts['diagnostic']+counts['engineering']<1806
                assert counts[phase[0]] < (6 if phase[0]=='engineering' else 1800)
                assert local[phase[0]] < (2 if phase[0]=='engineering' else 600)
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            def receipt(optimizer,args,kwargs):
                key=phase[0];counts[key]+=1;local[key]+=1
                append(root/'physical_updates.jsonl',dict(physical=sum(counts.values()),kind=key,kind_physical=counts[key],arm=name,local=local[key],time=time.time()))
            pre=opt.register_step_pre_hook(guard);post=opt.register_step_post_hook(receipt)
            original_clip=torch.nn.utils.clip_grad_norm_
            def clip(parameters,max_norm,*args,**kwargs):
                value=original_clip(parameters,max_norm,*args,**kwargs);norms.append(float(value));return value
            torch.nn.utils.clip_grad_norm_=clip
            try:
                def step(index):return trainer.train_step(lb,None,state=TrainerState(global_step=index,site_step=index))
                first=step(0);expected=snapshot(method,trainer,1);restore(initial,method,trainer)
                second=step(0);assert torch.equal(first.total_loss,second.total_loss) and same(expected,snapshot(method,trainer,1))
                restore(initial,method,trainer)
                write(out/'QUALIFICATION.json',dict(status='PASS',engineering_updates=2,full_state_golden_equal=True))
                phase[0]='diagnostic'
                append(out/'metrics.jsonl',dict(step=0,**evaluate(model,image,y)))
                for index in range(600):
                    result=step(index);active['cursor']=index+1
                    if index==0 or (index+1)%25==0:
                        row=dict(arm=name,step=index+1,loss=float(result.total_loss.detach()),gradient_norm_before_clip=norms[-1],
                                 components={k:float(v.detach()) for k,v in result.losses.items()},**result.scalars)
                        append(out/'telemetry.jsonl',row);write(root/'STATUS.json',dict(status='RUNNING',costs=counts.copy(),**row))
                    if index+1 in (25,100,300,600):append(out/'metrics.jsonl',dict(step=index+1,**evaluate(model,image,y)))
                    if (index+1)%100==0:save(out/'latest.pt',snapshot(method,trainer,index+1))
                assert local==dict(engineering=2,diagnostic=600)
                assert all(torch.equal(v,model.state_dict()[k]) for k,v in prior.items())
                save(out/'final.pt',snapshot(method,trainer,600))
                metric=json.loads((out/'metrics.jsonl').read_text().splitlines()[-1])
                metric.update(status='COMPLETE',mode=mode,updates=local,prior_unchanged=True,calls=model.calls.copy(),
                              clipping_fraction=sum(v>1 for v in norms)/len(norms),gradient_norm_min=min(norms),gradient_norm_max=max(norms))
                write(out/'RESULTS.json',metric);results[name]=metric
            finally:
                torch.nn.utils.clip_grad_norm_=original_clip;pre.remove();post.remove()
            del model,method,trainer;active.clear();torch.cuda.empty_cache()
            return out/'final.pt'
        a=arm('A','stochastic_ce')
        passed=all(v>=.8 for row in results['A']['stochastic_dice'] for v in row)
        write(root/'BRANCH_DECISION.json',dict(A_pass=passed,threshold=.8,basis='all six terminal fixed-four-draw probability-mean foreground Dice',chosen=['B','C'] if passed else ['D']))
        if passed:arm('B','joint');arm('C','joint',a)
        else:arm('D','mean_ce')
        complete=dict(status='COMPLETE',started=started,finished=time.time(),costs=counts,results=results,A_pass=passed)
        write(root/'RESULTS.json',complete);write(root/'STATUS.json',complete)
    except BaseException:
        failure=dict(status='FAILED',costs=counts,traceback=traceback.format_exc())
        if active:
            save(active['out']/'failure_partial.pt',snapshot(active['method'],active['trainer'],active['cursor']))
            failure['partial_saved']=True
        write(root/'FAILURE.json',failure);write(root/'STATUS.json',failure)
        raise


if __name__=='__main__':
    settings=json.loads(Path(os.environ['TASK_SETTINGS']).read_text());root=Path(settings['output'])
    if root.exists():raise FileExistsError('create-only output exists')
    code=1
    try:run(settings);code=0
    finally:
        if root.exists():write(root/'EXIT.json',dict(exit_code=code,time=time.time()))

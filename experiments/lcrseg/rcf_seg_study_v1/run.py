"""Bounded Siemens->Philips->GE->Canon transfer, using the shared Trainer."""
import copy
import json
import os
import time
import traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.engine.trainer import Trainer, TrainerState, seed_everything
from lcrseg.engine.checkpoint import capture_rng_state, restore_rng_state
from lcrseg.methods.supervised import SupervisedMethod
from care_seg_study_v1.data import PatientBatcher, evaluate
from .data import datasets
from care_seg_study_v1.run import write, append, save
from .model import Segmenter, RoutedView, routing_prototypes
from .method import RCFMethod, canonical

SITES=('Siemens','Philips','GE','Canon')
SOURCE_STEPS=12000
TARGET_STEPS=1500


@torch.no_grad()
def router(model,labeled,unlabeled):
    """Read only training images, including visible-L images without opening their labels."""
    vectors=[];device=next(model.parameters()).device
    model.eval()
    for dataset in (labeled,unlabeled):
        for sample in dataset.samples:
            image=dataset._read_image(sample)[None].to(device)
            f=model.frozen_features(image)
            vectors.append(F.normalize(f.mean((-2,-1)),dim=1).cpu())
    return routing_prototypes(torch.cat(vectors))


def trainer_for(method,steps,source=False):
    opt=torch.optim.AdamW([p for p in method.model.parameters() if p.requires_grad],
                         lr=1e-4 if source else 5e-4,weight_decay=1e-4)
    schedule=torch.optim.lr_scheduler.LambdaLR(opt,lambda step: max(0.,1-step/steps)**.9 if source else 1.)
    return Trainer(method,optimizer=opt,scheduler=schedule,device='cuda',amp=False)


def snapshot(method,trainer,step):
    return dict(model=copy.deepcopy(method.model.state_dict()),method=method.method_state_dict(),
                trainer=copy.deepcopy(trainer.state_dict()),rng=capture_rng_state(),step=step)


def restore(payload,method,trainer):
    method.model.load_state_dict(payload['model']);method.load_method_state_dict(payload['method'])
    trainer.load_state_dict(payload['trainer']);restore_rng_state(payload['rng'])


def qualify(method,trainer,lb,ub,out,source=False):
    actual=0
    def record(opt,args,kwargs):
        nonlocal actual
        actual+=1
        append(out/'engineering_updates.jsonl',dict(physical=actual,time=time.time()))
    handle=trainer.optimizer.register_step_post_hook(record)
    if hasattr(method,'prepare'):method.prepare(0)
    l=lb.batch_at(0);u=None if source else ub.batch_at(0)
    initial=snapshot(method,trainer,0)
    first=trainer.train_step(l,u,state=TrainerState())
    expected={n:p.detach().clone() for n,p in method.model.named_parameters() if p.requires_grad}
    restore(initial,method,trainer)
    second=trainer.train_step(l,u,state=TrainerState())
    if not torch.equal(first.total_loss,second.total_loss) or any(not torch.equal(p,expected[n]) for n,p in method.model.named_parameters() if p.requires_grad):
        raise AssertionError('exact checkpoint-resume qualification failed')
    restore(initial,method,trainer)
    count=2;overfit=None
    if source:
        start=float(first.total_loss.detach())
        for k in range(32):
            result=trainer.train_step(l,None,state=TrainerState(global_step=k,site_step=k))
        end=float(result.total_loss.detach());count+=32
        if not end<start:raise AssertionError('two-case loss did not decrease')
        overfit=dict(updates=32,initial_loss=start,final_loss=end)
        restore(initial,method,trainer)
    write(out/'QUALIFICATION.json',dict(status='PASS',engineering_updates=count,exact_resume=True,
          finite_backward=True,two_case_overfit=overfit))
    if actual!=count:raise AssertionError('engineering receipt count mismatch')
    handle.remove()
    del initial,expected
    trainer.optimizer.zero_grad(set_to_none=True)


def train(method,labeled,unlabeled,out,steps,root,source=False):
    out.mkdir(exist_ok=False)
    lb=PatientBatcher(labeled,{s.row['patient_id'] for s in labeled.samples},True,163)
    ub=PatientBatcher(unlabeled,{s.row['patient_id'] for s in unlabeled.samples},False,163)
    trainer=trainer_for(method,steps,source)
    qualify(method,trainer,lb,ub,out,source)
    frozen={n:v.detach().cpu().clone() for n,v in method.model.state_dict().items()
            if not any(n==pname for pname,p in method.model.named_parameters() if p.requires_grad)} if not source else {}
    physical=0
    def guard(opt,args,kwargs):
        if physical>=steps:raise RuntimeError('fixed update budget exhausted')
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in method.model.parameters()):
            raise FloatingPointError('nonfinite optimizer gradient')
    def receipt(opt,args,kwargs):
        nonlocal physical
        physical+=1;append(out/'physical_updates.jsonl',dict(step=physical,time=time.time()))
    trainer.optimizer.register_step_pre_hook(guard);trainer.optimizer.register_step_post_hook(receipt)
    for step in range(steps):
        if hasattr(method,'prepare'):
            if step%150==0:
                write(root/'STATUS.json',dict(status='CALIBRATING',stage=out.name,step=step,time=time.time()))
            method.prepare(step)
        result=trainer.train_step(lb.batch_at(step),None if source else ub.batch_at(step),
                                 state=TrainerState(global_step=step,site_step=step,epoch=step//150))
        if step==0 or (step+1)%25==0:
            row=dict(status='RUNNING',stage=out.name,step=step+1,budget=steps,
                     loss=float(result.total_loss.detach()),time=time.time(),**result.scalars)
            write(root/'STATUS.json',row);append(out/'telemetry.jsonl',row);print(json.dumps(row),flush=True)
        if (step+1)%(500 if source else 150)==0:
            save(out/'latest.pt',snapshot(method,trainer,step+1))
    if frozen:
        for n,v in method.model.state_dict().items():
            if n in frozen and not torch.equal(v.cpu(),frozen[n]):
                raise AssertionError('historical/shared state changed: '+n)
    save(out/'final.pt',dict(model=method.model.state_dict(),step=steps,
                           active=method.model.active,routers=method.model.routers))
    write(out/'TRAINING_COMPLETE.json',dict(status='COMPLETE',physical_updates=physical,
           frozen_parameters_and_buffers_unchanged=not source))
    del trainer,frozen


def evaluate_seen(model,data,sites,out):
    rows={}
    for site in sites:
        view=RoutedView(model)
        metrics,private=evaluate(view,data,site)
        metrics['routing_slice_counts']=dict(zip(model.routers,view.counts))
        rows[site]=metrics
        write(out/f'private_{site}.json',private)
    write(out/'RESULTS.json',rows)
    return rows


def main(settings):
    root=Path(settings['output']);data=Path(settings['data'])
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    seed_everything(163)
    try:
        write(root/'STATUS.json',dict(status='PREPARING_SOURCE',time=time.time()))
        model=Segmenter().cuda();model.pretrained(settings['pretrained'])
        labeled,unlabeled=datasets(data,'Siemens')
        source=SupervisedMethod(model,config=dict(lambda_dice=1.),continual=True)
        source.begin_site('Siemens',None,SOURCE_STEPS)
        train(source,labeled,unlabeled,root/'source',SOURCE_STEPS,root,True)
        model.eval();model.routers['source']=router(model,labeled,unlabeled)
        source_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
        source_router=copy.deepcopy(model.routers)
        save(root/'source'/'sealed.pt',dict(model=source_state,routers=source_router,steps=SOURCE_STEPS))
        source_results=evaluate_seen(model,data,['Siemens'],root/'source')
        del source,model,labeled,unlabeled
        results={}
        for arm in ('confidence','rcf'):
            seed_everything(163)
            model=Segmenter().cuda();model.load_state_dict(source_state);model.isolate()
            model.routers=copy.deepcopy(source_router);trajectory=[]
            for index,site in enumerate(SITES[1:],1):
                seed_everything(163+index)
                model.add_task(site)
                labeled,unlabeled=datasets(data,site)
                method=RCFMethod(model,labeled,unlabeled,arm)
                method.begin_site(site,None,TARGET_STEPS)
                out=root/(arm+'_'+site)
                train(method,labeled,unlabeled,out,TARGET_STEPS,root)
                model.eval();model.routers[site]=router(model,labeled,unlabeled)
                save(out/'deployment.pt',dict(model=model.state_dict(),routers=model.routers,
                                            tasks=list(SITES[1:index+1]),steps=TARGET_STEPS))
                trajectory.append(evaluate_seen(model,data,SITES[:index+1],out))
                del method,labeled,unlabeled
            results[arm]=trajectory;del model
        final=dict(status='COMPLETE',source=source_results,trajectories=results,
                   formal_updates=SOURCE_STEPS+2*3*TARGET_STEPS,engineering_updates=46,
                   scope='single seed validation transfer; no test-label access')
        write(root/'RESULTS.json',final);write(root/'STATUS.json',final)
    except BaseException as exc:
        failure=dict(status='FAILED',error=str(exc),traceback=traceback.format_exc(),time=time.time())
        write(root/'FAILURE.json',failure);write(root/'STATUS.json',failure);raise


if __name__=='__main__':
    main(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))

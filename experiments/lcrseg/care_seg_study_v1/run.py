"""Study orchestration; all backward/optimizer/EMA updates use the shared Trainer."""
import copy
import json
import os
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from lcrseg.engine.trainer import Trainer, TrainerState, seed_everything
from lcrseg.engine.checkpoint import capture_rng_state, restore_rng_state
from lcrseg.methods.care_seg_v1 import CARESegMethod
from lcrseg.methods.supervised import SupervisedMethod
from .native import NativeSegmenter
from .baselines import EMAControl
from .data import datasets, volumes, PatientBatcher, evaluate

ARMS = ('supervised', 'ema', 'care_no_repair', 'care')
STEPS = 4000
TREE = [dict(name='foreground',children=[[0],[1,2,3]]),
        dict(name='ventricle_side',children=[[1,2],[3]]),
        dict(name='lv_cavity_wall',children=[[1],[2]])]


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())


def save(path, value):
    temporary = Path(str(path)+'.tmp')
    with temporary.open('wb') as f:
        torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def make_method(arm, source, settings, labeled, unlabeled, config):
    model = NativeSegmenter(settings['reference'],source).cuda()
    if arm.startswith('care'):
        cfg = dict(config,repair_steps=0 if arm=='care_no_repair' else config['repair_steps'])
        method = CARESegMethod(model,config=cfg)
        method.prepare_datasets(labeled,unlabeled,'Philips')
    elif arm=='ema':
        method = EMAControl(model,config)
    elif arm=='supervised':
        method = SupervisedMethod(model,config=config,continual=True)
    else:
        raise ValueError('unknown arm')
    method.begin_site('Philips',None,STEPS)
    if arm.startswith('care'):
        method._make_old_model()
    return method


def make_trainer(method):
    parameters = [p for p in method.model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD([dict(params=[p for p in parameters if p.ndim>1],weight_decay=1e-4),
                                 dict(params=[p for p in parameters if p.ndim<=1],weight_decay=0)],
                                lr=.005,momentum=.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:max(0.,1-step/STEPS)**.9)
    return Trainer(method,optimizer=optimizer,scheduler=scheduler,device='cuda',amp=False)


def checkpoint(method, trainer, step, arm):
    return dict(arm=arm,step=step,model=method.model.state_dict(),method=method.method_state_dict(),
                trainer=trainer.state_dict(),rng=capture_rng_state())


def restore(payload,method,trainer,arm):
    if payload['arm']!=arm:
        raise ValueError('checkpoint arm mismatch')
    method.model.load_state_dict(payload['model'])
    method.load_method_state_dict(payload['method'])
    trainer.load_state_dict(payload['trainer'])
    restore_rng_state(payload['rng'])


def qualify(method, trainer, l, u, out, arm):
    if hasattr(method,'before_training_step'):
        method.before_training_step(0)
    initial = copy.deepcopy(checkpoint(method,trainer,0,arm))
    ledger = out/'engineering_updates.jsonl'
    count = sum(1 for _ in ledger.open()) if ledger.exists() else 0
    def guard(opt,args,kwargs):
        if count >= 32: raise RuntimeError('engineering cap exhausted')
    def record(opt,args,kwargs):
        nonlocal count
        count += 1
        append(ledger,dict(physical=count,time=time.time()))
    handles = [trainer.optimizer.register_step_pre_hook(guard),trainer.optimizer.register_step_post_hook(record)]
    first = trainer.train_step(l,u,state=TrainerState())
    after = copy.deepcopy(method.model.state_dict())
    restore(initial,method,trainer,arm)
    second = trainer.train_step(l,u,state=TrainerState())
    if not torch.equal(first.total_loss,second.total_loss) or any(not torch.equal(v,after[k]) for k,v in method.model.state_dict().items()):
        raise AssertionError('GPU exact resume check failed')
    restore(initial,method,trainer,arm)
    for handle in handles: handle.remove()
    write(out/'QUALIFICATION.json',dict(status='PASS',engineering_optimizer_updates=count,
        loss=float(first.total_loss.detach()),exact_resume=True,controller=getattr(method,'accounting',{})))


def run_arm(arm, source, settings, config, labeled, unlabeled, root):
    out = root/arm
    if (out/'RESULTS.json').exists():
        return json.loads((out/'RESULTS.json').read_text())
    out.mkdir(exist_ok=True)
    seed_everything(163)
    method = make_method(arm,source,settings,labeled,unlabeled,config)
    trainer = make_trainer(method)
    patients = config['roles']['Philips']['A'] if arm.startswith('care') else sorted({s.row['patient_id'] for s in labeled.samples})
    lb = PatientBatcher(labeled,patients,True,163)
    ub = PatientBatcher(unlabeled,{s.row['patient_id'] for s in unlabeled.samples},False,163)
    start = 0
    latest = out/'latest.pt'
    ledger = out/'physical_updates.jsonl'
    physical = sum(1 for _ in ledger.open()) if ledger.exists() else 0
    if latest.exists():
        payload = torch.load(latest,map_location='cpu',weights_only=False)
        restore(payload,method,trainer,arm);start=payload['step']
        append(out/'recoveries.jsonl',dict(checkpoint_step=start,prior_physical=physical,replayed=physical-start,time=time.time()))
    else:
        if physical:
            raise RuntimeError('physical updates exist without a durable checkpoint')
        l,u=lb.batch_at(0),None if arm=='supervised' else ub.batch_at(0)
        qualify(method,trainer,l,u,out,arm)
        save(latest,checkpoint(method,trainer,0,arm))
    def pre_step(opt,args,kwargs):
        if physical>=STEPS+128:
            raise RuntimeError('bounded recovery update allowance exhausted')
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in method.model.parameters()):
            raise FloatingPointError('nonfinite gradient')
    def post_step(opt,args,kwargs):
        nonlocal physical
        physical+=1
        append(ledger,dict(physical=physical,time=time.time()))
    trainer.optimizer.register_step_pre_hook(pre_step)
    trainer.optimizer.register_step_post_hook(post_step)
    started=time.time()
    for step in range(start,STEPS):
        if hasattr(method,'before_training_step'):
            if step//config['block_steps']!=method.last_refresh:
                write(root/'STATUS.json',dict(status='REFRESHING',arm=arm,step=step,physical=physical,time=time.time()))
            method.before_training_step(step)
        result = trainer.train_step(lb.batch_at(step),None if arm=='supervised' else ub.batch_at(step),
                                    state=TrainerState(global_step=step,site_step=step))
        if step==start or (step+1)%20==0:
            row=dict(status='RUNNING',arm=arm,step=step+1,physical=physical,seconds=time.time()-started,
                     loss=float(result.total_loss.detach()),**result.scalars)
            append(out/'telemetry.jsonl',row);write(root/'STATUS.json',row);print(json.dumps(row),flush=True)
        if (step+1)%100==0 or step+1==STEPS:
            save(latest,checkpoint(method,trainer,step+1,arm))
    save(out/'final.pt',checkpoint(method,trainer,STEPS,arm))
    write(root/'STATUS.json',dict(status='EVALUATING',arm=arm,step=STEPS,physical=physical,time=time.time()))
    metrics = {}
    # Fixed final student is primary for every arm; no best-checkpoint selection.
    for site in ('Siemens','Philips'):
        metrics[site],private = evaluate(method.model,settings['data'],site)
        write(out/f'private_{site}.json',private)
    result=dict(arm=arm,status='COMPLETE',steps=STEPS,physical_updates=physical,
                engineering_optimizer_updates=sum(1 for _ in (out/'engineering_updates.jsonl').open()),metrics=metrics,
                controller=getattr(method,'accounting',{}))
    write(out/'RESULTS.json',result)
    return result


def main(settings):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='3':
        raise PermissionError('this protocol is bound to physical GPU 3')
    root=Path(settings['output'])
    if not root.is_dir():
        raise FileNotFoundError('prepare a create-only run directory first')
    torch.set_num_threads(4)
    seed_everything(163)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    try:
        write(root/'STATUS.json',dict(status='PREPARING_DATA',time=time.time()))
        labeled,unlabeled=datasets(Path(settings['data']),'Philips')
        patients=sorted({r['patient_id'] for r in labeled.records})
        if len(patients)!=13 or len({r['patient_id'] for r in unlabeled.records})!=50:
            raise ValueError('unexpected frozen Philips population')
        shuffled=np.random.default_rng(163).permutation(patients).tolist()
        config=dict(tree=TREE,roles={'Philips':dict(A=sorted(shuffled[:7]),B=sorted(shuffled[7:]))},
                    geometry='physical3d',volumes=volumes(labeled,unlabeled),grid=[32,32],
                    block_steps=250,repair_radius=20.,repair_steps=50,repair_bound=.5,repair_lr=.1,
                    minimum_patients=2,em_iterations=25,fit_pixels_per_patient=2048,
                    ema=.99,lambda_dice=0.,lambda_care=1.,lambda_assim=1.,assim_ramp_steps=1,tau_cls=.95)
        config_path=root/'private_config.json'
        if config_path.exists() and json.loads(config_path.read_text())!=config:
            raise ValueError('frozen private configuration changed')
        if not config_path.exists():write(config_path,config)
        source=torch.load(settings['source'],map_location='cpu',weights_only=False)
        if not (root/'SOURCE_BASELINE.json').exists():
            write(root/'STATUS.json',dict(status='SOURCE_EVALUATING',time=time.time()))
            model=NativeSegmenter(settings['reference'],source).cuda()
            metrics={}
            for site in ('Siemens','Philips'):
                metrics[site],private=evaluate(model,settings['data'],site)
                write(root/f'private_source_{site}.json',private)
            write(root/'SOURCE_BASELINE.json',metrics);del model
        results={}
        for arm in ARMS:
            write(root/'STATUS.json',dict(status='PREPARING',arm=arm,time=time.time()))
            results[arm]=run_arm(arm,source,settings,config,labeled,unlabeled,root)
        base=json.loads((root/'SOURCE_BASELINE.json').read_text())
        table={}
        for arm,result in results.items():
            m=result['metrics']
            table[arm]=dict(Philips_dice=m['Philips']['macro_dice'],Siemens_dice=m['Siemens']['macro_dice'],
                           old_domain_change_pp=100*(m['Siemens']['macro_dice']-base['Siemens']['macro_dice']),
                           two_domain_mean=(m['Philips']['macro_dice']+m['Siemens']['macro_dice'])/2)
        full=table['care']
        passes=all(full['Philips_dice']>=table[a]['Philips_dice']+.005 and
                   full['Siemens_dice']>=table[a]['Siemens_dice']-.01 for a in ARMS if a!='care')
        summary=dict(status='COMPLETE',table=table,pilot_gate=passes,scope='single seed, two-domain validation pilot; not independent test evidence')
        write(root/'RESULTS.json',summary);write(root/'STATUS.json',summary)
    except BaseException as exc:
        failure=dict(status='FAILED',error=str(exc),traceback=traceback.format_exc(),time=time.time())
        write(root/'FAILURE.json',failure);write(root/'STATUS.json',failure);raise


if __name__=='__main__':
    main(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))

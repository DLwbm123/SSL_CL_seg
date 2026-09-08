"""One loop for the six single-domain recipes; fixed snapshots never select weights."""
import argparse, gc, hashlib, json, math, os, time
from collections import Counter
from pathlib import Path
import torch
from .core import *
from .diagnostics import snapshot

def save(path,student,teacher,opt,proto,support,**meta):
    temp=Path(str(path)+'.tmp')
    torch.save(dict(student=student.state_dict(),ema=teacher.state_dict(),optimizer=opt.state_dict(),proto=proto,support=support,**meta),temp);os.replace(temp,path)

def run(data,reference,output,domain,seed,arm,device,*,expected=None,shape=(384,384),epochs=100,qualification=False,resume=False,stop_epoch=None):
    from .freeze import verify
    source='qualification' if qualification else verify()
    if not qualification and (seed not in (11,12,13) or epochs!=100 or stop_epoch is not None):raise PermissionError('fixed matrix')
    spec=config(arm);root=Path(output);root.mkdir(parents=True,exist_ok=resume)
    if (root/'receipt.json').exists():raise FileExistsError('terminal run protected')
    kw={} if expected is None else dict(expected=expected)
    l=DomainData(data,domain,'train_labeled',shape=shape,**kw)
    # Count metadata only; no U image accessor is retained for SUP or warm-up.
    u_count=len(DomainData(data,domain,'train_unlabeled',shape=shape,**kw));u=None
    steps=max(math.ceil(len(l)/2),math.ceil(u_count/2));total=steps*epochs
    counters=Counter();proto=support=None;start_epoch=1;started=time.time();label_order=hashlib.sha256()
    if resume:
        ck=torch.load(root/'latest.pt',map_location=device,weights_only=False)
        if any(ck[k]!=v for k,v in dict(domain=domain,seed=seed,arm=arm,source=source).items()):raise PermissionError('resume identity')
        lines=(root/'steps.jsonl').read_text().splitlines()
        if len(lines)!=ck['counters']['optimizer_steps']:raise RuntimeError('checkpoint/log divergence; preserve attempt')
        # Assign the two loaded states directly to meta modules; no extra initialized shadows.
        from experiments.lcrseg.di_dmpa_jascl.modeling import _official_probabilistic_classifier
        with rng(device,'resume_import'):_official_probabilistic_classifier(reference,upstream_path=UPSTREAM_PATH)
        with rng(device,'resume_meta'),torch.device('meta'):
            student=build_lcrseg_unet_jascl_model(reference,upstream_path=UPSTREAM_PATH,input_channels=3,num_classes=3)
            teacher=build_lcrseg_unet_jascl_model(reference,upstream_path=UPSTREAM_PATH,input_channels=3,num_classes=3)
        student.load_state_dict(ck.pop('student'),assign=True);teacher.load_state_dict(ck.pop('ema'),assign=True)
        student.decoder.conv_logit.grad_update.requires_grad_(False)
        for p in teacher.parameters():p.requires_grad_(False)
        opt=optimizer_for(student);opt.load_state_dict(ck.pop('optimizer'));proto,support=ck['proto'],ck['support'];counters=Counter(ck['counters']);start_epoch=ck['epoch']+1;init=ck['init'];del ck
        precision()
    else:
        student=build(reference,device,seed,domain);teacher=ema_from(student);opt=optimizer_for(student)
        init=dict(student_hash=state_hash(student),ema_hash=state_hash(teacher),backbone_hash=hashlib.sha256(student.enc1.block[0].weight.detach().cpu().numpy().tobytes()).hexdigest(),head_hash=hashlib.sha256(student.decoder.conv_logit.mu.weight.detach().cpu().numpy().tobytes()).hexdigest())
        assert init['student_hash']==init['ema_hash']
        write_json(root/'initialization.json',dict(source=source,domain=domain,seed=seed,arm=arm,**init))
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    student.train();teacher.eval()
    for epoch in range(start_epoch,epochs+1):
        begin=time.time()
        if spec['ssl'] and epoch>20 and u is None:u=DomainData(data,domain,'train_unlabeled',shape=shape,**kw)
        if spec['pas'] and epoch>=21 and (epoch-21)%5==0:
            proto,support=prototypes(student,l,device,counters,seed,epoch,'training_prototype')
        li=orders(len(l),seed,domain,epoch,'labeled_order',steps)
        ui=orders(u_count,seed,domain,epoch,'unlabeled_order',steps)
        for step in range(steps):
            k=(epoch-1)*steps+step
            for group in opt.param_groups:group['lr']=.001*(1-k/total)**.9
            labeled=batch(l,li[step],device,seed,epoch,step,'labeled')
            unlabeled=batch(u,ui[step],device,seed,epoch,step,'unlabeled') if spec['ssl'] and epoch>20 else None
            label_order.update(json.dumps([epoch,step,li[step]]).encode())
            row=train_step(student,teacher,opt,labeled,unlabeled,proto,support,arm,seed,domain,epoch,step,counters,diagnostic=qualification)
            append(root/'steps.jsonl',dict(epoch=epoch,step=step,update=counters['optimizer_steps'],lr=opt.param_groups[0]['lr'],**row))
        if epoch==20:write_json(root/'warmup.json',dict(student_hash=state_hash(student),ema_hash=state_hash(teacher),label_order_hash=label_order.hexdigest(),u_opens=0 if u is None else u.opens))
        if epoch%20==0 or epoch==epochs:
            try:
                snapshot(student,teacher,l,data=data,reference=reference,device=device,domain=domain,seed=seed,arm=arm,epoch=epoch,output=root/f'eval{epoch}',proto=proto,support=support,counters=counters,expected=expected,shape=shape,export_deployment=epoch==epochs)
            except BaseException as error:
                write_json(root/'diagnostic_failure.json',dict(status='INCOMPLETE_DIAGNOSTICS',epoch=epoch,error=repr(error)))
                raise
        save(root/'latest.pt',student,teacher,opt,proto,support,domain=domain,seed=seed,arm=arm,epoch=epoch,counters=dict(counters),source=source,init=init,complete=epoch==epochs)
        append(root/'epochs.jsonl',dict(epoch=epoch,updates=counters['optimizer_steps'],seconds=time.time()-begin,student_hash=state_hash(student) if epoch==epochs else None))
        print(json.dumps(dict(domain=domain,arm=arm,seed=seed,epoch=epoch,updates=counters['optimizer_steps'])),flush=True)
        if stop_epoch==epoch:return dict(status='QUALIFICATION_INTERRUPTED',updates=counters['optimizer_steps'])
    assert counters['optimizer_steps']==total
    # Export references existing student tensors; do not create a third model or shadow.
    torch.save(dict(student=student.state_dict(),domain=domain,seed=seed,arm=arm,source=source,epoch=epochs,complete=True,student_hash=state_hash(student)),root/'deploy_student.pt')
    from experiments.lcrseg.single_teacher_scd_v0_1.engine import audit_live_models
    live=audit_live_models(2)
    result=dict(status='TRAINING_COMPLETE',source=source,domain=domain,arm=arm,seed=seed,data_split_seed=0,optimizer_updates=total,counters=dict(counters),student_hash=state_hash(student),ema_hash=state_hash(teacher),initialization=init,labeled_opens=l.opens,unlabeled_opens=0 if u is None else u.opens,models=live,student_parameter_bytes=tensor_bytes(dict(student.named_parameters())),ema_parameter_bytes=tensor_bytes(dict(teacher.named_parameters())),optimizer_bytes=tensor_bytes(opt.state_dict()),training_prototype_bytes=tensor_bytes((proto,support)),sigma_bytes=tensor_bytes(student.decoder.conv_logit.sigma.weight),sigma_gradient=False,label_order_hash=label_order.hexdigest(),peak_allocated=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,peak_reserved=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None,seconds=time.time()-started,finished_unix=time.time())
    write_json(root/'receipt.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('data','reference','output','domain','arm'):p.add_argument('--'+k,required=True)
    p.add_argument('--seed',type=int,required=True)
    a=vars(p.parse_args())
    try:run(**a,device=torch.device('cuda:0'))
    except BaseException as e:
        if Path(a['output']).is_dir():write_json(Path(a['output'])/'failure.json',dict(status='INCOMPLETE_TRAINING',error=repr(e)))
        raise

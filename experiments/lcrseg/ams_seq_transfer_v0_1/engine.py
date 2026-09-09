"""One current-domain trainer. Evaluation runs in a different process after exit."""
import argparse,gc,json,math,os,hashlib,time
from pathlib import Path
from collections import Counter
import torch
from . import core as c
from .contract import verify,admit,read
from experiments.lcrseg.ssl_anchored_mix_v0_1.telemetry import Operations,flush
from experiments.lcrseg.single_teacher_scd_v0_1.engine import audit_live_models

def save(path,payload):
    tmp=Path(str(path)+'.tmp');torch.save(payload,tmp);os.replace(tmp,path)

def start_models(reference,device,task,parent=None,source=None):
    if parent is None:
        if task['source_task_id'] is not None:raise PermissionError('missing new source student')
        student=c.build(reference,device,task['seed'],task['domain'],'SUP_CE');parent_hash=None
    else:
        payload=torch.load(parent,map_location=device,weights_only=False)
        if set(payload)!={'student','student_hash','task','source','head_mode','complete'}:raise PermissionError('student-only artifact required')
        expected=payload['task']
        if not payload['complete'] or payload['head_mode']!=c.LINEAR or payload['source']!=source or expected['task_id']!=task['source_task_id'] or expected['seed']!=task['seed'] or expected['domain']!=task['source_domain'] or expected['arm']!='SRC_CE':
            raise PermissionError('source identity mismatch')
        student=c.from_state(reference,device,payload.pop('student'),c.LINEAR)
        parent_hash=payload['student_hash'];assert c.state_hash(student)==parent_hash
        del payload
    teacher=c.ema_from(student);opt=c.optimizer_for(student)
    boundary=dict(student_hash=c.state_hash(student),EMA_hash=c.state_hash(teacher),parent_student_hash=parent_hash,
                  optimizer_state_empty=len(opt.state)==0,source_EMA_loaded=False,source_Adam_loaded=False,source_RNG_loaded=False)
    assert boundary['student_hash']==boundary['EMA_hash'] and boundary['optimizer_state_empty']
    return student,teacher,opt,boundary

def train(base,task_id,data,reference,device,*,resume=False,fixture=None,stop_after=None):
    source=verify() if fixture is None else 'SYNTHETIC'
    if fixture is None:
        task=admit(base,task_id,source);epochs=100;shape=(384,384);expected=(c.MANIFEST_SHA,c.SPLIT_SHA)
        if stop_after is not None:raise PermissionError('formal budget cannot be shortened')
    else:
        if not (Path(data)/'SYNTHETIC_FIXTURE.json').is_file() or fixture['shape']!=(24,24):raise PermissionError('fixture boundary')
        task=fixture['task'];epochs=fixture['epochs'];shape=fixture['shape'];expected=fixture['expected']
    root=Path(base)/'tasks'/task_id
    if (root/'receipt.json').exists():raise FileExistsError('complete task protected')
    root.mkdir(parents=True,exist_ok=resume)
    parent=None if task['source_task_id'] is None else Path(base)/'tasks'/task['source_task_id']/'deploy_student.pt'
    if parent is not None:
        receipt=read(parent.parent/'receipt.json')
        if receipt['status']!='TRAINING_COMPLETE' or receipt['source']!=source:raise PermissionError('parent incomplete')
    with c.training_access(task['domain']):
        l=c.DomainData(data,task['domain'],'train_labeled',expected=expected,shape=shape)
        u_count=len(c.DomainData(data,task['domain'],'train_unlabeled',expected=expected,shape=shape))
        patients=c.patients_for(data,task['domain'],expected);steps=max(math.ceil(len(l)/2),math.ceil(u_count/2));total=steps*epochs
        if fixture is None and total!=task['updates']:raise RuntimeError('budget/data mismatch')
        counts=Counter();position=0;order_hash=hashlib.sha256();u=None;prior_L=prior_U=0
        if resume:
            payload=torch.load(root/'latest.pt',map_location=device,weights_only=False)
            if payload['task']!=task or payload['source']!=source or payload['total']!=total:raise PermissionError('resume identity')
            position=payload['position'];lines=(root/'steps.jsonl').read_text().splitlines()
            if len(lines)!=position:raise RuntimeError('uncommitted update tail; preserve attempt, do not replay')
            student=c.from_state(reference,device,payload.pop('student'),c.LINEAR);teacher=c.from_state(reference,device,payload.pop('EMA'),c.LINEAR)
            for p in teacher.parameters():p.requires_grad_(False)
            opt=c.optimizer_for(student);opt.load_state_dict(payload.pop('optimizer'));counts=Counter(payload['counts']);boundary=payload['boundary']
            prior_L,prior_U=payload['L_opens'],payload['U_opens'];rng=payload['rng'];c.random.setstate(rng['python']);c.np.random.set_state(rng['numpy']);torch.set_rng_state(rng['cpu'].cpu())
            if device.type=='cuda':torch.cuda.set_rng_state(rng['cuda'].cpu(),device)
            del payload,rng
        else:
            student,teacher,opt,boundary=start_models(reference,device,task,parent,source)
            if parent is not None and boundary['parent_student_hash']!=receipt['student_hash']:raise PermissionError('source receipt/content mismatch')
            c.write_json(root/'boundary.json',dict(status='PASS',source=source,**boundary))
        for k in range(position):
            epoch,index=divmod(k,steps);ii=c.orders(len(l),task['seed'],task['domain'],epoch+1,'labeled_order',steps)[index]
            order_hash.update(json.dumps([epoch+1,index,ii]).encode())
        student.train();teacher.eval();started=time.time()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(position//steps+1,epochs+1):
            lis=c.orders(len(l),task['seed'],task['domain'],epoch,'labeled_order',steps)
            uis=c.orders(u_count,task['seed'],task['domain'],epoch,'unlabeled_order',steps)
            if epoch>20 and task['arm'] in ('T_UCTX','T_AMS') and u is None:u=c.DomainData(data,task['domain'],'train_unlabeled',expected=expected,shape=shape)
            for index in range(steps):
                k=(epoch-1)*steps+index
                if k<position:continue
                for group in opt.param_groups:group['lr']=.001*(1-k/total)**.9
                lb=c.batch(l,lis[index],device,task['seed'],epoch,index,'labeled')
                ub=c.batch(u,uis[index],device,task['seed'],epoch,index,'unlabeled') if u is not None else None
                row=c.step(student,teacher,opt,lb,ub,task['arm'],task['seed'],task['domain'],epoch,index,counts,[patients[i] for i in lis[index]])
                order_hash.update(json.dumps([epoch,index,lis[index]]).encode());position=k+1
                c.append(root/'steps.jsonl',dict(epoch=epoch,step=index,update=position,lr=opt.param_groups[0]['lr'],**row));flush('step_committed')
                if index==steps-1 or position==stop_after:
                    rng=dict(python=c.random.getstate(),numpy=c.np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)
                    save(root/'latest.pt',dict(task=task,source=source,total=total,position=position,student=student.state_dict(),EMA=teacher.state_dict(),optimizer=opt.state_dict(),counts=dict(counts),boundary=boundary,L_opens=prior_L+l.opens,U_opens=prior_U+(u.opens if u else 0),rng=rng))
                if position==stop_after:return dict(status='SYNTHETIC_INTERRUPTED',position=position)
            if epoch==20:c.write_json(root/'warmup.json',dict(student_hash=c.state_hash(student),EMA_hash=c.state_hash(teacher),optimizer_hash=c.optimizer_hash(opt),label_order_hash=order_hash.hexdigest(),U_opens=prior_U+(u.opens if u else 0)))
            print(json.dumps(dict(task_id=task_id,epoch=epoch,updates=position)),flush=True)
        assert position==total and counts['optimizer_steps']==total
        student_hash=c.state_hash(student);save(root/'deploy_student.pt',dict(student=student.state_dict(),student_hash=student_hash,task=task,source=source,head_mode=c.LINEAR,complete=True))
        result=dict(status='TRAINING_COMPLETE',source=source,task=task,updates=position,student_hash=student_hash,EMA_hash=c.state_hash(teacher),optimizer_hash=c.optimizer_hash(opt),label_order_hash=order_hash.hexdigest(),boundary=boundary,counts=dict(counts),L_opens=prior_L+l.opens,U_opens=prior_U+(u.opens if u else 0),models=audit_live_models(2),student_parameter_bytes=c.tensor_bytes(dict(student.named_parameters())),optimizer_bytes=c.tensor_bytes(opt.state_dict()),sigma_bytes=c.tensor_bytes(student.decoder.conv_logit.sigma.weight),grad_update_bytes=c.tensor_bytes(student.decoder.conv_logit.grad_update),sigma_requires_grad=student.decoder.conv_logit.sigma.weight.requires_grad,sigma_has_grad=student.decoder.conv_logit.sigma.weight.grad is not None,peak_allocated=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,peak_reserved=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None,seconds=time.time()-started,completed_unix=time.time())
        c.write_json(root/'receipt.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('base','task_id','data','reference'):p.add_argument('--'+k.replace('_','-'),required=True)
    p.add_argument('--resume',action='store_true');a=vars(p.parse_args());root=Path(a['base'])/'tasks'/a['task_id']
    try:
        op=Operations(root.parent/(root.name+'_operations'))
        if a['resume']:
            previous=read(op.root/'operation_counts.json');op.counts=Counter(previous['counts'])
            with (op.root/('resume_prior_'+str(time.time_ns())+'.json')).open('x') as f:json.dump(previous,f)
        with op:train(**a,device=torch.device('cuda:0'))
    except BaseException as e:
        c.write_json(root.parent/(root.name+'_failure.json'),dict(status='INCOMPLETE_TRAINING',error=repr(e)));raise

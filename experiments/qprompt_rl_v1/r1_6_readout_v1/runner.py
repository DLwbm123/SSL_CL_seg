"""Frozen R1.6 worker; historical model, optimizer and augmentation are imported."""
import gc,json,os,random,time,uuid,traceback
from pathlib import Path
import numpy as np
import torch
from r1_12h import core as c
from r1_12h import runner as r
from .runtime import PLAN,TASKS,CONFIG_SHA,seed_value,rpc,process_identity,save_checkpoint,write_diagnostic
from .method import objective,gradient_blocks,readout_loss,matched_assignments
RUN=r.RUN; CODE=r.CODE


def setup(seed):
    r.setup();random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    c.seed=lambda *parts:seed_value(seed,*parts)
    r.seed=c.seed


def provenance():
    r.provenance()
    return dict(code_commit=CODE,training_code_commit=CODE,config_digest=CONFIG_SHA)


def schedule(seed,domain):
    path=RUN/'schedules'/f'{seed}__{domain}.json'
    doc=json.loads(path.read_text());sha=c.file_sha(path)
    assert sha==json.loads((RUN/'SEED_AND_SCHEDULE_AUDIT.json').read_text())['schedules'][f'{seed}__{domain}']['sha256']
    return doc,sha


def prefix_path(spec):
    dep=spec['depends_on'][0]
    if spec['seed']==261:
        return Path(json.loads((RUN/'PREFIX_AUDIT.private.json').read_text())[dep]['path'])
    return RUN/'tasks'/dep/'prefix.pt'


def finite_tree(value):
    if isinstance(value,torch.Tensor):return bool(torch.isfinite(value).all())
    if isinstance(value,dict):return all(finite_tree(x) for x in value.values())
    if isinstance(value,(list,tuple)):return all(finite_tree(x) for x in value)
    return True


def prepare():
    setup(261);provenance()
    old=Path(os.environ['EXEC_OLD_RUN'])
    # Full L/val binding in a separate preparation process; workers open current L only.
    binding={};schedules={};models={}
    for domain,counts in zip(PLAN['domains'],[(16,40),(10,25)]):
        for role,count in zip(('train_labeled','val'),counts):
            ds=r.dataset(domain,role);assert len(ds)==count
            for i in range(len(ds)):
                item=ds[i];assert torch.isfinite(item['image']).all() and (item['label']!=255).any()
            binding[domain+'/'+role]=count
        ds=r.dataset(domain,'train_labeled')
        for seed in PLAN['optimization_seeds']:
            rows=[]
            for step in range(3000):
                g=torch.Generator().manual_seed(seed_value(seed,domain,step,'indices'))
                rows.append(dict(global_step=step,indices=torch.randperm(len(ds),generator=g)[:2].tolist(),geometry_seed=seed_value(seed,domain,step,'geometry'),photometric_seed=seed_value(seed,domain,step,'photometric')))
            doc=dict(domain=domain,rows=ds.rows,steps=rows)
            if seed==261:assert doc==json.loads((old/'schedules'/f'{domain}.json').read_text())
            path=RUN/'schedules'/f'{seed}__{domain}.json'
            if path.exists():assert json.loads(path.read_text())==doc
            else:c.atomic(path,doc)
            schedules[f'{seed}__{domain}']=dict(sha256=c.file_sha(path),old_schedule_equal=seed==261,split_seed=0)
    for seed in PLAN['optimization_seeds']:
        setup(seed)
        for backbone in PLAN['backbones']:
            model,digest=c.build(backbone,False,r.ASSETS,r.SOURCE);del model
            models[f'{seed}__{backbone}']=dict(body_digest=digest,query_head_seed=seed_value(seed,backbone,'query-head'),body_seed=seed_value(seed,backbone,'body'))
    audit={}
    # Trusted historical receipt format is verified by the launcher before preparation.
    trusted=json.loads((RUN/'TRUSTED_PREFIXES.private.json').read_text())
    for backbone in PLAN['backbones']:
        for domain in PLAN['domains']:
            task=f's261__{backbone}__{domain}__QUERY_PREFIX';path=old/'tasks'/f'{backbone}__{domain}__QUERY_PREFIX'/'prefix.pt'
            try:
                expected=trusted[f'{backbone}__{domain}'];sha=c.file_sha(path);assert sha==expected
                state=torch.load(path,map_location='cpu',weights_only=False)
                assert state['code_commit']=='41dfa243f15b184d728c19422a9f23ddcb346de9'
                assert state['global_step']==state['data_cursor']==state['scheduler']['last_epoch']==2000
                assert state['backbone']==backbone and state['domain']==domain and state['arm']=='QUERY_PREFIX'
                assert state['data_schedule_digest']==schedules[f'261__{domain}']['sha256']
                assert state['model_initialization_digest']==models[f'261__{backbone}']['body_digest']
                assert finite_tree(state) and state['reference'] is None and state['bank'] is None
                audit[task]=dict(valid=True,path=str(path),sha256=sha,source_commit=state['code_commit'],step=2000)
                del state
            except Exception as e:audit[task]=dict(valid=False,error=repr(e))
    c.atomic(RUN/'PREFIX_AUDIT.private.json',audit)
    c.atomic(RUN/'PREFIX_AUDIT.json',{k:{a:b for a,b in v.items() if a not in ('path','error')} for k,v in audit.items()})
    c.atomic(RUN/'SEED_AND_SCHEDULE_AUDIT.json',dict(schedules=schedules,models=models,split_seed=0,independent_optimization_seeds=[261,262,263]))
    c.atomic(RUN/'DATA_BINDING.json',dict(records=binding,manifest_sha=r.MANIFEST_SHA,split_sha=r.SPLIT_SHA,migration_sha=c.file_sha(r.MIGRATION),**provenance()))
    c.atomic(RUN/'ENVIRONMENT.json',dict(python=__import__('sys').version,torch=torch.__version__,numpy=np.__version__,determinism='warn_only',tf32=False))


def bank_init(model,cache,backbone,folder):
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
    bank=r.initialize_bank(model,cache,backbone,torch.device('cuda:0'))
    random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])
    c.append(folder/'TASK_LEDGER.jsonl',dict(event='bank_initialization',forwards=len(cache),optimizer_calls=0,rng_preserved=True))
    return bank


def claim(task):
    token=uuid.uuid4().hex
    rpc(RUN,action='claim',task=task,token=token,pid=os.getpid(),identity=process_identity(os.getpid()))
    return token


def update(model,opt,sch,bank,x,y,arm,task,token,step,folder,kind='formal',diagnose=False):
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):total,terms,read,current,support,out=objective(model,x,y,arm,bank)
    grads=gradient_blocks(model,terms) if diagnose else {}
    total.backward()
    norm=torch.stack([p.grad.float().square().sum() for p in model.parameters() if p.grad is not None]).sum().sqrt()
    if not torch.isfinite(norm):raise FloatingPointError('nonfinite gradient')
    grant=rpc(RUN,action='attempt',kind=kind,task=task,token=token,step=step,request_id=uuid.uuid4().hex,device='cuda')
    c.append(folder/'TASK_LEDGER.jsonl',dict(event='optimizer_attempt',step=step,**grant,code_commit=CODE,config_digest=CONFIG_SHA))
    try:opt.step()
    except BaseException:
        c.append(folder/'TASK_LEDGER.jsonl',dict(event='optimizer_failed',step=step,**grant));raise
    sch.step()
    if bank is not None:bank.update(current.detach(),support)
    c.append(folder/'TASK_LEDGER.jsonl',dict(event='committed',step=step+1,**grant,code_commit=CODE,config_digest=CONFIG_SHA))
    return dict(losses={**{k:float(v.detach()) for k,v in terms.items()},'total':float(total.detach())},readout=read,gradients={**grads,'total_norm':float(norm)})


def state_for(model,opt,sch,bank,step,metadata,elapsed):
    return c.snapshot(model,opt,sch,None,bank,step,training_seconds=elapsed,trainable_parameters=[n for n,p in model.named_parameters() if p.requires_grad],generator_state='stateless per-step CPU geometry and CUDA photometric generators; seeds bound to schedule',**metadata)


def train():
    task=os.environ['EXEC_TASK'];spec=TASKS[task];setup(spec['seed']);provenance();folder=RUN/'tasks'/task;folder.mkdir(parents=True,exist_ok=True)
    token=claim(task);doc,sha=schedule(spec['seed'],spec['domain']);ds=r.dataset(spec['domain'],'train_labeled');assert ds.rows==doc['rows'];cache=[ds[i] for i in range(len(ds))]
    model,init=c.build(spec['backbone'],False,r.ASSETS,r.SOURCE);model.cuda().train();opt,sch=c.optimizer_for(model,spec['backbone']);bank=None
    metadata=dict(**provenance(),task=task,backbone=spec['backbone'],domain=spec['domain'],arm=spec['arm'],optimization_seed=spec['seed'],data_schedule_digest=sha,model_initialization_digest=init,prefix_source_commit=None,prefix_sha256=None)
    step=0;elapsed=0.;latest=folder/'latest.pt'
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=False)
        assert state['code_commit']==CODE and state['config_digest']==CONFIG_SHA and state['data_schedule_digest']==sha
        if state['bank'] is not None:bank=c.PrototypeBank(128 if spec['backbone']=='UNET_QUERY_128' else 384).cuda()
        step=c.restore(state,model,opt,sch,None,bank);elapsed=state['training_seconds'];metadata.update(prefix_source_commit=state['prefix_source_commit'],prefix_sha256=state['prefix_sha256']);del state
    elif spec['global_start']==2000:
        prefix=prefix_path(spec);state=torch.load(prefix,map_location='cpu',weights_only=False)
        assert state['global_step']==state['data_cursor']==state['scheduler']['last_epoch']==2000 and state['data_schedule_digest']==sha
        if spec['seed']==261:
            entry=json.loads((RUN/'PREFIX_AUDIT.private.json').read_text())[spec['depends_on'][0]];assert entry['valid'] and c.file_sha(prefix)==entry['sha256'] and state['code_commit']==entry['source_commit']
        else:assert state['code_commit']==CODE and state['optimization_seed']==spec['seed']
        step=c.restore(state,model,opt,sch);metadata.update(prefix_source_commit=state['code_commit'],prefix_sha256=c.file_sha(prefix));del state
        if spec['arm'] in ('B1','B3'):bank=bank_init(model,cache,spec['backbone'],folder)
        c.atomic(folder/'PREFIX_BINDING.json',dict(**metadata,restored_step=step,scheduler_step=sch.last_epoch,student_digest=c.tensor_sha(model.state_dict())))
    assert (bank is not None)==(spec['arm'] in ('B1','B3'))
    if not latest.exists():save_checkpoint(folder,state_for(model,opt,sch,bank,step,metadata,elapsed),RUN)
    start=time.monotonic();deadline=json.loads((RUN/'SESSION.json').read_text())['deadline'];torch.cuda.reset_peak_memory_stats()
    try:
        while step<spec['global_end'] and time.time()<deadline:
            if (RUN/'BLOCKED_MODES.json').exists() and 'train' in json.loads((RUN/'BLOCKED_MODES.json').read_text()):break
            x,y=r.load_batch(cache,doc['steps'][step],torch.device('cuda:0'))
            info=update(model,opt,sch,bank,x,y,spec['arm'],task,token,step,folder,diagnose=(step+1)%100==0);step+=1
            if step%100==0 or step==spec['global_end']:
                save_checkpoint(folder,state_for(model,opt,sch,bank,step,metadata,elapsed+time.monotonic()-start),RUN)
                c.atomic(folder/'PROGRESS.json',dict(global_step=step,**metadata))
                write_diagnostic(folder/'DIAGNOSTICS.jsonl',dict(step=step,**metadata),**info,runtime=dict(bank_support=None if bank is None else bank.supported.tolist(),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),training_seconds=elapsed+time.monotonic()-start,lr=[p['lr'] for p in opt.param_groups]))
        done=step==spec['global_end'];duration=elapsed+time.monotonic()-start
        save_checkpoint(folder,state_for(model,opt,sch,bank,step,metadata,duration),RUN,('final.pt' if spec['is_final_endpoint'] else 'prefix.pt') if done else None)
        c.atomic(folder/('TRAIN_DONE.json' if done else ('ENGINEERING_STOP.json' if time.time()<deadline else 'TIME_LIMIT.json')),dict(global_step=step,committed_valid_updates=step-spec['global_start'],training_seconds=duration,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),**metadata))
    except BaseException as e:
        c.atomic(folder/f'FAILURE_{time.time_ns()}.private.json',dict(error=repr(e),traceback=traceback.format_exc(),step=step,**metadata));raise
    finally:rpc(RUN,action='release',task=task,token=token)


def evaluate():
    spec=TASKS[os.environ['EXEC_TASK']];setup(spec['seed']);r.CONFIG_SHA=CONFIG_SHA;r.provenance=provenance_for_evaluation
    r.evaluate()
    path=RUN/'tasks'/spec['id']/'EVALUATION.json';value=json.loads(path.read_text());state=torch.load(path.parent/'final.pt',map_location='cpu',weights_only=False)
    value.update({k:state[k] for k in ('optimization_seed','prefix_sha256','prefix_source_commit','data_schedule_digest','config_digest')});c.atomic(path,value)


def provenance_for_evaluation():return dict(code_commit=CODE,config_digest=CONFIG_SHA)


def main():
    mode=os.environ['EXEC_MODE']
    if mode=='prepare':prepare()
    elif mode=='train':train()
    elif mode=='evaluate':evaluate()
    elif mode in ('qualify','smoke','forensic'):
        from . import qualification
        getattr(qualification,mode)()
    elif mode=='supervise':
        from .supervisor import supervise
        supervise()
    else:raise ValueError(mode)

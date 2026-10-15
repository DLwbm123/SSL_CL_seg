"""Scheduler-only GPU4--7 pooling; workers retain the admitted code snapshot."""
import ctypes,json,os,select,signal,struct,subprocess,time,shutil
from pathlib import Path


def write(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def process(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        return dict(state=fields[0],birth=int(fields[19]))
    except FileNotFoundError:return None


def alive(entry):
    info=process(entry['pid'])
    return info is not None and info['birth']==entry['birth'] and info['state']!='Z'


def slot(running,gpus,capacity):
    counts={g:sum(e['gpu']==g for e in running.values()) for g in gpus}
    available=[g for g in gpus if counts[g]<capacity]
    return min(available,key=lambda g:(counts[g],g)) if available else None


def selfcheck():
    running={'a':dict(gpu=4),'b':dict(gpu=5),'legacy':dict(gpu=2)}
    assert slot(running,[4,5,6,7],2)==6
    filled={str(i):dict(gpu=g) for i,g in enumerate([4,4,5,5,6,6,7,7])}
    assert slot(filled,[4,5,6,7],2) is None
    del filled['0'];assert slot(filled,[4,5,6,7],2)==4


def watch(config):
    """SIGSTOP on receipt rename, before the next optimizer invocation."""
    libc=ctypes.CDLL(None,use_errno=True);fd=libc.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC)
    if fd<0:raise OSError(ctypes.get_errno(),'inotify init')
    root=Path(config['root']);change=Path(config['change']);pending={}
    for key,entry in config['inherited'].items():
        if entry['gpu'] in config['gpus']:continue
        folder=root/'jobs'/key;receipt=folder/'latest.pt.receipt.json'
        initial=json.loads(receipt.read_text())['step'] if receipt.exists() else 0
        wd=libc.inotify_add_watch(fd,os.fsencode(folder),0x80) # IN_MOVED_TO
        if wd<0:raise OSError(ctypes.get_errno(),'inotify watch')
        pending[wd]=(key,entry,initial)
    write(change/'WATCH_STATUS.json',dict(status='WAITING_CHECKPOINT',nodes=[v[0] for v in pending.values()]))
    while pending:
        if not select.select([fd],[],[],2)[0]:
            for wd,(key,entry,initial) in list(pending.items()):
                if not alive(entry):pending.pop(wd)
            continue
        data=os.read(fd,65536);offset=0
        while offset<len(data):
            wd,mask,cookie,length=struct.unpack_from('iIII',data,offset);offset+=16
            name=data[offset:offset+length].split(b'\0',1)[0];offset+=length
            if wd not in pending or name!=b'latest.pt.receipt.json':continue
            key,entry,initial=pending[wd]
            if not alive(entry):pending.pop(wd);continue
            stopped=time.time();os.kill(entry['pid'],signal.SIGSTOP)
            for _ in range(100):
                info=process(entry['pid'])
                if info is None or info['state']=='T':break
                time.sleep(.001)
            folder=root/'jobs'/key
            receipt=json.loads((folder/'latest.pt.receipt.json').read_text())
            calls=len((folder/'physical_updates.jsonl').read_text().splitlines())
            if info is None or info['state']!='T' or receipt['step']<=initial or calls!=receipt['step']:
                if alive(entry):os.kill(entry['pid'],signal.SIGCONT)
                write(change/(key+'.boundary_missed.json'),dict(receipt=receipt,calls=calls,retired=False))
                pending[wd]=(key,entry,receipt['step']);continue
            archive=change/key;archive.mkdir(exist_ok=False)
            for name in ('STATUS.json','latest.pt','latest.pt.receipt.json','physical_updates.jsonl'):
                shutil.copy2(folder/name,archive/name)
            value=dict(id=key,old=entry,step=receipt['step'],physical_updates=calls,checkpoint=str(archive/'latest.pt'),
                       stopped=stopped,prefix_elapsed=json.loads((archive/'STATUS.json').read_text())['elapsed_seconds'],
                       archived_ledger_is_copy=True)
            write(change/(key+'.ready.json'),value);pending.pop(wd)
    os.close(fd);write(change/'WATCH_STATUS.json',dict(status='COMPLETE'))


def resume(config):
    # Reuse the original worker loop; only entry construction is restored.
    from experiments.lcrseg.main_style_ssl_v1 import runtime
    cfg=config['worker'];handoff=config['handoff'];out=Path(cfg['output'])
    assert handoff['physical_updates']==handoff['step']
    assert len((out/'physical_updates.jsonl').read_text().splitlines())==handoff['step']
    original_setup=runtime.setup;original_write=runtime.write
    class ResumePath(type(Path())):
        def mkdir(self,mode=0o777,parents=False,exist_ok=False):
            if self==out and self.is_dir() and not exist_ok:return None
            return super().mkdir(mode,parents,exist_ok)
    runtime.Path=ResumePath
    def restored_setup(node,data,reference,device,predecessor=None,initialize=True):
        trainer=original_setup(node,data,reference,device,predecessor,initialize=False)
        identity={k:node[k] for k in ('id','dataset','seed','order','stage','domain')}
        runtime.restore(trainer,handoff['checkpoint'],identity)
        assert trainer.step==handoff['step']
        trainer.physical_optimizer_updates=handoff['physical_updates']
        return trainer
    def resumed_write(path,value):
        if Path(path)==out/'STATUS.json' and value.get('status')=='STARTING':
            value.update(status='RESUMING',step=handoff['step'],physical_updates=handoff['physical_updates'])
        if Path(path)==out/'COSTS.json':
            value['active_segment_elapsed_seconds']=value['elapsed_seconds']
            value['prefix_elapsed_seconds']=handoff['prefix_elapsed']
            value['elapsed_seconds']+=handoff['prefix_elapsed']
            value['resource_migration']=dict(checkpoint_step=handoff['step'],repeated_optimizer_calls=0,old_gpu=handoff['old']['gpu'])
        original_write(path,value)
    runtime.setup=restored_setup;runtime.write=resumed_write
    runtime.worker(cfg)


def run(config):
    from experiments.lcrseg.main_style_ssl_v1.protocol import nodes
    from experiments.lcrseg.main_style_ssl_v1.controller import summarize
    selfcheck();root=Path(config['root']);change=Path(config['change']);plan=nodes()
    running=config['inherited'].copy();states=config['states'].copy();gpus=config['gpus'];capacity=config['capacity']
    assert gpus==[4,5,6,7] and capacity==2
    started=time.time()-config['previous_elapsed'];logs={}
    def launch(key,gpu,worker_cfg,resume_value=None):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
        if key in config.get('worker_code_roots',{}):
            env['PYTHONPATH']=config['worker_code_roots'][key]
        name=key+('.resume' if resume_value else '.new')
        if resume_value:
            cfg=change/(key+'.resume.config.json');write(cfg,dict(worker=worker_cfg,handoff=resume_value))
            env.update(TASK_SCRIPT=config['script'],RESOURCE_ACTION='resume',RESOURCE_CONFIG=str(cfg))
            command=['bash','./s',config['python'],'./c.py']
        else:
            cfg=root/'configs'/(key+'.json');write(cfg,worker_cfg)
            env.update(TASK_MODULE='experiments.lcrseg.main_style_ssl_v1.entry',TASK_ACTION='worker',TASK_CONFIG=str(cfg))
            command=['bash','./s',config['python'],'./n.py']
        log=(change/(name+'.log')).open('x');logs[key]=log
        p=subprocess.Popen(command,cwd=config['ops'],env=env,stdout=log,stderr=subprocess.STDOUT)
        running[key]=dict(pid=p.pid,gpu=gpu,birth=process(p.pid)['birth'])
        if resume_value:
            running[key]['retire_old']=resume_value['old']
            running[key]['resumed_from']=resume_value['step']
    # The suspended old coordinator has no GPU state; retire only that PID.
    old=config['old_controller']
    if alive(old):os.kill(old['pid'],signal.SIGTERM);os.kill(old['pid'],signal.SIGCONT)
    while len(states)<len(plan):
        for key,entry in list(running.items()):
            retire=entry.get('retire_old')
            status_file=root/'jobs'/key/'STATUS.json'
            status=json.loads(status_file.read_text()) if status_file.exists() else {}
            healthy=(status.get('status')=='RUNNING' and status.get('step',0)>entry.get('resumed_from',0)) or status.get('status')=='COMPLETE'
            if retire and healthy:
                if alive(retire):os.kill(retire['pid'],signal.SIGTERM);os.kill(retire['pid'],signal.SIGCONT)
                write(change/(key+'.migrated.json'),dict(old=retire,new={k:v for k,v in entry.items() if k!='retire_old'},
                      resumed_step=status['step'],physical_updates=status['physical_updates'],repeated_optimizer_calls=0))
                del entry['retire_old']
            if alive(entry):continue
            exit_path=root/'jobs'/key/'EXIT.json';exit_value=json.loads(exit_path.read_text()) if exit_path.exists() else {}
            states[key]='COMPLETE' if status.get('status')=='COMPLETE' and exit_value.get('exit_code')==0 else 'FAILED'
            if key in logs:logs.pop(key).close()
            del running[key]
        for key,entry in list(running.items()):
            ready=change/(key+'.ready.json')
            if entry['gpu'] in gpus or not ready.exists():continue
            gpu=slot(running,gpus,capacity)
            if gpu is None:continue
            handoff=json.loads(ready.read_text());assert handoff['old']['pid']==entry['pid']
            assert alive(entry) and process(entry['pid'])['state']=='T'
            cfg=json.loads((root/'configs'/(key+'.json')).read_text())
            del running[key];launch(key,gpu,cfg,handoff)
        for node in plan:
            key=node['id'];pred=node['predecessor']
            if key in states or key in running:continue
            if pred and states.get(pred) in ('FAILED','BLOCKED'):
                states[key]='BLOCKED';out=root/'jobs'/key;out.mkdir()
                write(out/'STATUS.json',dict(status='BLOCKED',node=node,reason='predecessor failed; no automatic retry'));continue
            gpu=slot(running,gpus,capacity)
            if gpu is None or pred and states.get(pred)!='COMPLETE':continue
            cfg=dict(node=node,output=str(root/'jobs'/key),data=config['data'],reference=config['reference'],
                     predecessor=None if pred is None else str(root/'jobs'/pred/'sealed.pt'))
            launch(key,gpu,cfg)
        write(root/'STATUS.json',dict(status='RUNNING',completed=states,running=running,elapsed_seconds=time.time()-started,total_nodes=len(plan),
              scheduling=dict(gpus=gpus,capacity_per_gpu=capacity,legacy_draining=[k for k,e in running.items() if e['gpu'] not in gpus],
                              pending_retirements=[k for k,e in running.items() if 'retire_old' in e])))
        if len(states)<len(plan):time.sleep(2)
    summary=summarize(root,plan);failed={k:v for k,v in states.items() if v!='COMPLETE'}
    write(root/'STATUS.json',dict(status='COMPLETE' if not failed else 'COMPLETE_WITH_FAILURES',nodes=states,elapsed_seconds=time.time()-started))
    write(root/'FINAL.json',dict(completed=len(states),failures=failed,summary=summary,public_delivery='PENDING_COMPLETION_DELIVERY',scientific_success='NOT_ASSUMED',
          attempt_history=config.get('attempt_history'),worker_code_commits=config.get('worker_code_commits',{})))


if __name__=='__main__':
    selfcheck()
    cfg=json.loads(Path(os.environ['RESOURCE_CONFIG']).read_text())
    {'watch':watch,'resume':resume,'run':run}[os.environ['RESOURCE_ACTION']](cfg)

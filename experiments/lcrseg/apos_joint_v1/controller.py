"""Independent original-clock V3 controller: source gate then all qualifications."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
from .runtime import guard,resources
from ..main_style_ssl_v1.runtime import write

def birth(pid):return int(Path('/proc',str(pid),'stat').read_text().split()[21])
def selfcheck(cfg):
    import torch
    from .math import check
    from .runtime import SyntheticCalls,synthetic_record
    calls=SyntheticCalls(cfg);old_backward=torch.autograd.backward
    def counted_backward(*a,**kw):synthetic_record(cfg,'reverse',1);return old_backward(*a,**kw)
    torch.autograd.backward=counted_backward
    from . import model as alg
    from .run import terms
    from ..space_lora_v1.core import source_model
    prior=torch.load(Path(cfg['reuse_root'])/'prior.private.pt',map_location='cpu',weights_only=False);result={}
    for idx,dev in enumerate(('cpu','cuda')):
        write(Path(cfg['root'])/'SELF_CHECK_PROGRESS.private.json',dict(stage=dev,synthetic_reverse_reserved=8*(idx+1),medical_samples=0,optimizer=0));result[dev.upper()]=check(dev,prior,calls)
    x=torch.linspace(-1,1,384,device='cuda').reshape(1,1,1,384).expand(2,1,384,384).contiguous();y=torch.zeros(2,384,384,device='cuda',dtype=torch.long);y[:,130:220,140:230]=1;y[:,120:130,140:230]=2;y[:,150:210,250:290]=3
    benchmarks={};native=0
    for idx,arm in enumerate(('N','Q','R')):
        write(Path(cfg['root'])/'SELF_CHECK_PROGRESS.private.json',dict(stage=arm,synthetic_reverse_reserved=17+idx,synthetic_native_samples_reserved=10+idx*8,medical_samples=0,optimizer=0))
        c=dict(cfg,stage='target',arm=arm,bucket='synthetic',phase='synthetic',group='synthetic');m=alg.make(c);m.calls=calls
        with torch.no_grad():
            out=m(x);assert float((out.exp()-m.last['logq0'].exp()).abs().max())<=5e-7 and torch.equal(out.argmax(1),m.last['logq0'].argmax(1))
        native+=4
        if arm=='N':
            ref=source_model(cfg)
            with torch.no_grad():calls.sample(len(x));expected=torch.nn.functional.log_softmax(ref(x),1)
            assert torch.equal(expected,m.last['logq']);native+=2;del ref
        with torch.no_grad():m.adapter_B.weight.fill_(1e-3)
        torch.cuda.synchronize();start=time.time();l,u,_=terms(m,x,y);(l+u).backward();torch.cuda.synchronize();benchmarks[arm]=time.time()-start;native+=4
        assert m.adapter_B.weight.grad is not None and float(m.adapter_B.weight.grad.norm())>0 and all(p.grad is None for p in m.parent.parameters())
        assert float((m.last['logq']-m.last['logq0']).abs().max())>0
        del m,l,u,out;torch.cuda.empty_cache()
    projected=1.7*2000*sum(benchmarks.values())+9400*max(benchmarks.values())/4+1200
    result.update(status='PASS' if projected<20000 else 'STOP_RUNTIME',synthetic_reverse=19,synthetic_native_samples=native,native_identity=True,nonzero_real_prior_and_drift=True,benchmarks_seconds=benchmarks,projected_seconds_including_margin=projected,peak_bytes=torch.cuda.max_memory_allocated(),medical_samples=0,optimizer=0)
    result['counted_synthetic_calls']=json.loads((Path(cfg['root'])/'SYNTHETIC_COUNTERS.json').read_text())
    write(Path(cfg['root'])/'SELF_CHECK.json',result)

def worker(cfg):
    import torch
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if cfg['task']=='selfcheck':return selfcheck(cfg)
    from . import run
    return {'qualification':run.qualify,'formal':run.train,'evaluate':run.evaluate}[cfg['task']](cfg)

def main(cfg):
    if cfg.get('task'):return worker(cfg)
    root=Path(cfg['root']);write(root/'WORKER.private.json',dict(pid=os.getpid(),birth=birth(os.getpid()),process_group=os.getpgrp()))
    def launch(task,stage=None,arm=None):
        guard(cfg,new=True);name=task if stage is None else stage+'_'+arm+'_'+task;c=dict(cfg,task=task)
        if stage:c.update(stage=stage,arm=arm,phase='qualification' if task=='qualification' else 'formal',group=('source' if stage=='source_eval' else stage)+'_'+arm,bucket='source_evaluation' if stage=='source_eval' else 'target_evaluation' if task=='evaluate' else 'target_'+('qualification' if task=='qualification' else 'formal'),output=str(root/'qualification'/arm if task=='qualification' else root/('source' if stage=='source_eval' else stage)/arm))
        settings=root/('SETTINGS_'+name+'.private.json');write(settings,c)
        with (root/(name+'.log')).open('x') as f:
            p=subprocess.Popen([sys.executable,cfg['entry']],env=dict(os.environ,TASK_SETTINGS=str(settings)),stdout=f,stderr=subprocess.STDOUT);active=dict(pid=p.pid,birth=birth(p.pid),name=name);write(root/('ACTIVE_'+name+'.private.json'),active);write(root/'ACTIVE.private.json',active);write(root/'STATUS.json',dict(status='RUNNING',stage=name,resources=guard(cfg)));code=p.wait()
        write(root/('EXIT_'+name+'.json'),dict(exit_code=code));assert code==0,('worker failed',name,code)
    def stop(status):write(root/'STATUS.json',dict(status=status,target_formal_started=False,resources=resources(cfg)));write(root/'EXIT.json',dict(exit_code=0))
    try:
        if not (root/'SELF_CHECK.json').exists():launch('selfcheck')
        if json.loads((root/'SELF_CHECK.json').read_text())['status']!='PASS':return stop('NUMERIC_STOP')
        for arm in ('Q','R'):launch('evaluate','source_eval',arm)
        from .analysis import source_gate
        if source_gate(cfg)['status']!='PASS':return stop('SOURCE_STOP')
        launch('qualification','target','N')
        if json.loads((root/'P0_RESULT.json').read_text())['status']!='PASS':return stop('P0_STOP')
        for arm in ('Q','R'):launch('qualification','target',arm)
        for arm in ('N','Q','R'):launch('formal','target',arm)
        for arm in ('N','Q','R'):launch('evaluate','target',arm)
        write(root/'STATUS.json',dict(status='COMPLETE',resources=resources(cfg)));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException as e:
        write(root/'FAILURE.private.json',dict(error=str(e),traceback=traceback.format_exc(),resources=resources(cfg)));write(root/'STATUS.json',dict(status='ENGINEERING_STOP',resources=resources(cfg)));write(root/'EXIT.json',dict(exit_code=1));raise

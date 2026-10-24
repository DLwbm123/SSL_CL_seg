"""One original-clock controller, source gate before any target worker."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
from .runtime import guard,resources
from ..main_style_ssl_v1.runtime import write

def birth(pid):return int(Path('/proc',str(pid),'stat').read_text().split()[21])
def selfcheck(cfg):
    import torch
    from .appearance import check
    from . import model as alg
    from .run import terms
    from .runtime import SyntheticCalls
    from ..space_lora_v1.core import source_model
    calls=SyntheticCalls(cfg);original=torch.autograd.backward
    def backward(*a,**k):calls.extra('reverse',1);return original(*a,**k)
    torch.autograd.backward=backward
    result={'CPU':check('cpu'),'CUDA':check('cuda')}
    # Analytic tests above: two canonicalizer forwards (2 samples each) per device.
    calls.extra('appearance_samples',8);calls.extra('spline_samples',24);calls.extra('bias_samples',12)
    x=torch.linspace(-1,1,384,device='cuda').reshape(1,1,1,384).expand(2,1,384,384).contiguous()
    y=torch.zeros(2,384,384,device='cuda',dtype=torch.long);y[:,130:220,140:230]=1;y[:,120:130,140:230]=2;y[:,150:210,250:290]=3
    benchmarks={}
    for stage,arm in [('source','B'),('source','F'),('target','B'),('target','S'),('target','C')]:
        cc=dict(cfg,stage=stage,arm=arm,bucket='synthetic',phase='synthetic',group='synthetic',synthetic=True,task='selfcheck')
        m=alg.make(cc);m.calls=calls;m.frozen_reference=alg.frozen(m)
        if stage=='source' and arm=='B':
            reference=source_model(cfg)
            with torch.no_grad():
                calls.sample(len(x));expected=torch.nn.functional.log_softmax(reference(x),1);got=m(x)
            assert torch.equal(expected,got),'native function identity';del reference
        if stage=='target' and arm=='C':
            with torch.no_grad():
                for h in m.anatomy:h.weight.fill_(.001)
            m.frozen_reference=alg.frozen(m)
        torch.cuda.synchronize();start=time.time();l,u,_=terms(m,x,y);(l+u).backward();torch.cuda.synchronize();elapsed=time.time()-start
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        assert sum(float(p.grad.square().sum()) for p in m.canonicalizer.parameters() if p.grad is not None)>1e-12
        alg.check_frozen(m);benchmarks[stage+'_'+arm]=elapsed;del m,l,u;torch.cuda.empty_cache()
    projected=1.6*(2000*sum(benchmarks.values()))+5000*max(benchmarks.values())/4+1200
    result.update(status='PASS' if projected<20000 else 'STOP_RUNTIME',native_identity=True,benchmarks_seconds_per_batch_forward_backward=benchmarks,projected_seconds_including_margin=projected,peak_bytes=torch.cuda.max_memory_allocated(),medical_samples=0,optimizer=0,synthetic_calls=calls.count)
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
        if stage:
            c.update(stage=stage,arm=arm,phase='qualification' if task=='qualification' else 'formal',group=('source' if stage=='source_eval' else stage)+'_'+arm,bucket='source_evaluation' if stage=='source_eval' else 'target_evaluation' if task=='evaluate' else stage+('_qualification' if task=='qualification' else '_formal'),output=str(root/'qualification'/(stage+'_'+arm) if task=='qualification' else root/('source' if stage=='source_eval' else stage)/arm))
        settings=root/('SETTINGS_'+name+'.private.json');write(settings,c)
        with (root/(name+'.log')).open('x') as f:
            p=subprocess.Popen([sys.executable,cfg['entry']],env=dict(os.environ,TASK_SETTINGS=str(settings)),stdout=f,stderr=subprocess.STDOUT)
            active=dict(pid=p.pid,birth=birth(p.pid),name=name);write(root/('ACTIVE_'+name+'.private.json'),active);write(root/'ACTIVE.private.json',active);write(root/'STATUS.json',dict(status='RUNNING',stage=name,resources=guard(cfg)));code=p.wait()
        write(root/('EXIT_'+name+'.json'),dict(exit_code=code))
        if code:raise RuntimeError('worker failed '+name)
    try:
        if not (root/'SELF_CHECK.json').exists():launch('selfcheck')
        if json.loads((root/'SELF_CHECK.json').read_text())['status']!='PASS':
            write(root/'STATUS.json',dict(status='NUMERIC_STOP',resources=resources(cfg)));write(root/'EXIT.json',dict(exit_code=0));return
        for arm in ('B','F'):launch('qualification','source',arm)
        for arm in ('B','F'):launch('formal','source',arm)
        for arm in ('B','F'):launch('evaluate','source_eval',arm)
        from .analysis import source_gate
        gate=source_gate(cfg)
        if gate['status']!='PASS':
            write(root/'STATUS.json',dict(status='SOURCE_STOP',source_gate=gate['checks'],target_started=False,resources=resources(cfg)));write(root/'EXIT.json',dict(exit_code=0));return
        for arm in ('B','S','C'):launch('qualification','target',arm)
        for arm in ('B','S','C'):launch('formal','target',arm)
        for arm in ('B','S','C'):launch('evaluate','target',arm)
        write(root/'STATUS.json',dict(status='COMPLETE',source_gate='PASS',resources=resources(cfg)));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException as e:
        write(root/'FAILURE.private.json',dict(error=str(e),traceback=traceback.format_exc(),resources=resources(cfg)));write(root/'STATUS.json',dict(status='ENGINEERING_STOP',resources=resources(cfg)));write(root/'EXIT.json',dict(exit_code=1));raise

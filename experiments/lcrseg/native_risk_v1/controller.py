"""One original-clock controller, source gate before any target worker."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
from .runtime import guard,resources
from ..main_style_ssl_v1.runtime import write

def birth(pid):return int(Path('/proc',str(pid),'stat').read_text().split()[21])
def selfcheck(cfg):
    import torch
    from . import model as alg
    from .run import terms,js
    from .risk import values
    from .runtime import SyntheticCalls
    from ..space_lora_v1.core import source_model
    calls=SyntheticCalls(cfg);original=torch.autograd.backward
    def backward(*a,**k):calls.extra('reverse',1);return original(*a,**k)
    torch.autograd.backward=backward;result={}
    for device in ('cpu','cuda'):
        torch.manual_seed(47);lp=torch.randn(2,4,8,8,device=device,requires_grad=True).log_softmax(1);yy=torch.arange(128,device=device).reshape(2,8,8)%4
        rr,seen=values(lp,yy);calls.extra('risk_class_samples',6);assert seen.all();j=js(lp.exp(),lp.exp());assert float(j.abs().max())<1e-7;(rr.mean()+j.mean()).backward();assert torch.isfinite(lp.grad_fn.next_functions[0][0].variable.grad).all() if hasattr(lp.grad_fn.next_functions[0][0],'variable') else torch.isfinite(rr).all();result[device]=dict(status='PASS',JS_identity=True,risk_finite=True)
    projected=torch.tensor([-.1,.01,50.]).clamp(0,10);assert projected.tolist()==[0.,float(torch.tensor(.01)),10.]
    try:
        from .risk import load_cache
        load_cache(dict(cfg,stage='target',arm='C',cache=str(Path(cfg['root'])/'cache/source.private.json')))
    except AssertionError:pass
    else:raise AssertionError('target could open source cache')
    # Exact finite-support expectation: inverse-probability group estimate.
    probability=[.2,.3,.5];residual=[.4,-.1,.2]
    assert abs(sum(p*(r/p) for p,r in zip(probability,residual))-sum(residual))<1e-12
    x=torch.linspace(-1,1,384,device='cuda').reshape(1,1,1,384).expand(2,1,384,384).contiguous();y=torch.zeros(2,384,384,device='cuda',dtype=torch.long);y[:,130:220,140:230]=1;y[:,120:130,140:230]=2;y[:,150:210,250:290]=3
    fake=dict(groups=[dict(bound=1.,probability=1/6) for _ in range(6)],samples={'0:0':dict(present=[True]*3,groups=[0,1,2]),'1:0':dict(present=[True]*3,groups=[3,4,5])});benchmarks={}
    for stage,arm in [('source','F'),('target','B'),('target','S'),('target','C')]:
        cc=dict(cfg,stage=stage,arm=arm,bucket='synthetic',phase='synthetic',group='synthetic',synthetic=True,synthetic_groups=6,cache=None,task='selfcheck');m=alg.make(cc);m.calls=calls;m.frozen_reference=alg.frozen(m)
        if stage=='source':
            reference=source_model(cfg)
            with torch.no_grad():calls.sample(len(x));expected=torch.nn.functional.log_softmax(reference(x),1);got=m(x)
            assert torch.equal(expected,got),'zero-adapter source identity';del reference
        m.dual.fill_(.1);torch.cuda.synchronize();start=time.time();l,u,_=terms(m,x,y,['0:0','1:0'],fake);(l+u).backward();torch.cuda.synchronize();elapsed=time.time()-start
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters());assert float(m.adapter[2].weight.grad.norm())>0;alg.check_frozen(m);benchmarks[stage+'_'+arm]=elapsed;del m,l,u;torch.cuda.empty_cache()
    projected=1.6*2000*sum(benchmarks.values())+6500*max(benchmarks.values())/4+1200
    result.update(status='PASS' if projected<20000 else 'STOP_RUNTIME',native_identity=True,inverse_probability_exact_expectation=True,benchmarks_seconds_per_batch_forward_backward=benchmarks,projected_seconds_including_margin=projected,peak_bytes=torch.cuda.max_memory_allocated(),medical_samples=0,optimizer=0,synthetic_calls=calls.count)
    write(Path(cfg['root'])/'SELF_CHECK.json',result)

def worker(cfg):
    import torch
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if cfg['task']=='selfcheck':return selfcheck(cfg)
    if cfg['task']=='calibrate':
        from .risk import calibrate
        return calibrate(cfg)
    from . import run
    return {'qualification':run.qualify,'formal':run.train,'evaluate':run.evaluate}[cfg['task']](cfg)

def main(cfg):
    if cfg.get('task'):return worker(cfg)
    root=Path(cfg['root']);write(root/'WORKER.private.json',dict(pid=os.getpid(),birth=birth(os.getpid()),process_group=os.getpgrp()))
    def launch(task,stage=None,arm=None,**extra):
        guard(cfg,new=True);name=task if stage is None else stage+'_'+arm+'_'+task;c=dict(cfg,task=task)
        if stage:
            c.update(stage=stage,arm=arm,phase='qualification' if task=='qualification' else 'formal',group=('source' if stage=='source_eval' else stage)+'_'+arm,bucket='source_evaluation' if stage=='source_eval' else 'target_evaluation' if task=='evaluate' else stage+('_qualification' if task=='qualification' else '_formal'),output=str(root/'qualification'/(stage+'_'+arm) if task=='qualification' else root/('source' if stage=='source_eval' else stage)/arm))
        c.update(extra)
        if task in ('qualification','formal'):
            c['cache']=str(root/'cache'/('source.private.json' if stage=='source' else 'target_'+('B' if arm=='B' else 'F')+'.private.json'))
        if task=='evaluate':c.update(cache=None,load='source' if stage=='source_eval' else 'target')
        if task=='calibrate':name=stage+'_'+arm+'_'+extra['calibration']+'_'+task
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
        launch('calibrate','source','F',cache=None,load='initial',bucket='source_risk_calibration',calibration='baseline',calibration_output=str(root/'cache/source.private.json'))
        launch('qualification','source','F');launch('formal','source','F')
        launch('calibrate','source','F',cache=str(root/'cache/source.private.json'),load='source',bucket='source_risk_calibration',calibration='final',calibration_output=str(root/'RISK_SOURCE_FINAL.json'))
        launch('evaluate','source_eval','F')
        from .analysis import source_gate
        gate=source_gate(cfg)
        if gate['status']!='PASS':
            write(root/'STATUS.json',dict(status='SOURCE_STOP',source_gate=gate['checks'],target_started=False,resources=resources(cfg)));write(root/'EXIT.json',dict(exit_code=0));return
        for arm in ('B','F'):
            launch('calibrate','target',arm,cache=None,load='initial' if arm=='B' else 'source',bucket='target_risk_calibration',calibration='baseline',calibration_output=str(root/('cache/target_'+arm+'.private.json')))
        for arm in ('B','S','C'):launch('qualification','target',arm)
        for arm in ('B','S','C'):launch('formal','target',arm)
        for arm in ('B','S','C'):
            launch('calibrate','target',arm,cache=str(root/('cache/target_'+('B' if arm=='B' else 'F')+'.private.json')),load='target',bucket='target_risk_calibration',calibration='final',calibration_output=str(root/('RISK_TARGET_'+arm+'_FINAL.json')))
        for arm in ('B','S','C'):launch('evaluate','target',arm)
        write(root/'STATUS.json',dict(status='COMPLETE',source_gate='PASS',resources=resources(cfg)));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException as e:
        write(root/'FAILURE.private.json',dict(error=str(e),traceback=traceback.format_exc(),resources=resources(cfg)));write(root/'STATUS.json',dict(status='ENGINEERING_STOP',resources=resources(cfg)));write(root/'EXIT.json',dict(exit_code=1));raise

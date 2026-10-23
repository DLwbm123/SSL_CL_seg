"""Create-only bounded six-stage controller, neutral child argv."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
from .runtime import guard
from ..main_style_ssl_v1.runtime import write


def birth(pid):return int(Path('/proc',str(pid),'stat').read_text().split()[21])

def selfcheck(cfg):
    import torch
    from .math import check
    from .model import Model
    from .data import fields
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    results=[check('cpu'),check('cuda')]
    class Dummy(torch.nn.Module):
        def __init__(self):super().__init__();self.conv=torch.nn.Conv2d(1,16,1)
        def features(self,x):return self.conv(x)
    prior=dict(mu=torch.zeros(8,3072),factor=torch.zeros(8,3072,8),tau2=torch.full((8,3072),.03),active=torch.ones(8,dtype=torch.bool))
    g=torch.Generator().manual_seed(9);prior['factor'].normal_(std=.02,generator=g)
    for arm in ('D','G','T'):
        m=Model(Dummy(),prior,arm).cuda().eval()
        x=torch.ones(2,1,32,32,device='cuda')*.2;x[1]*=2
        q=m(x)
        assert torch.allclose(q.exp().sum(1),torch.ones_like(q[:,0]),atol=1e-5)
        assert torch.allclose(q[:1],m(x[:1]),atol=1e-5,rtol=1e-5)
        (q.square().mean()).backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    absent=torch.zeros(2,32,32,dtype=torch.long);d,p=fields(absent)
    assert torch.equal(d,torch.ones_like(d)) and (p==0).all()
    write(Path(cfg['root'])/'SELF_CHECK.json',dict(status='PASS',gaussian_dense_checks=results,synthetic_reverse=7,normalized_predictions=True,batch_independent_quadrature=True,absent_classes=True,medical_forward=0,optimizer=0))


def worker(cfg):
    import torch
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    task=cfg['task']
    if task=='selfcheck':return selfcheck(cfg)
    if task=='prior':
        from .data import fit_prior
        return fit_prior(cfg)
    from . import run
    return {'qualification':run.qualify,'formal':run.train,'evaluate':run.evaluate}[task](cfg)


def main(cfg):
    if cfg.get('task'):return worker(cfg)
    root=Path(cfg['root']);write(root/'WORKER.private.json',dict(pid=os.getpid(),birth=birth(os.getpid()),process_group=os.getpgrp()))
    def launch(task,stage=None,arm=None):
        guard(cfg,new=True)
        name=task if stage is None else stage+'_'+arm+'_'+task
        cc=dict(cfg,task=task)
        if stage:
            phase='qualification' if task=='qualification' else 'formal'
            cc.update(stage=stage,arm=arm,phase=phase,group=stage+'_'+arm,bucket='evaluation' if task=='evaluate' else stage+'_'+phase,
                      output=str(root/('qualification' if task=='qualification' else stage)/(stage+'_'+arm if task=='qualification' else arm)))
        settings=root/('SETTINGS_'+name+'.private.json');write(settings,cc)
        env=dict(os.environ,TASK_SETTINGS=str(settings));log=root/(name+'.log')
        with log.open('x') as stream:
            process=subprocess.Popen([sys.executable,cfg['entry']],env=env,stdout=stream,stderr=subprocess.STDOUT)
            active=dict(pid=process.pid,birth=birth(process.pid),name=name)
            write(root/('ACTIVE_'+name+'.private.json'),active);write(root/'ACTIVE.private.json',active)
            write(root/'STATUS.json',dict(status='RUNNING',task=name,resources=guard(cfg)))
            code=process.wait()
        write(root/('EXIT_'+name+'.json'),dict(exit_code=code))
        if code:raise RuntimeError('worker failed: '+name)
    try:
        if (root/'SELF_CHECK.json').exists():
            assert json.loads((root/'SELF_CHECK.json').read_text())['status']=='PASS'
            write(root/'SYNTHETIC_REUSE.json',dict(status='PASS',new_calls=0))
        else:launch('selfcheck')
        if cfg.get('resume_after_deterministic_nll_repair'):
            assert json.loads((root/'NLL_IMPLEMENTATION_REPAIR.json').read_text())['status']=='PASS'
            assert json.loads((root/'PRIOR_FIT.json').read_text())['status']=='PASS'
            write(root/'PRIOR_REUSE.json',dict(status='PASS',new_label_reads=0))
        else:launch('prior')
        for arm in ('D','G','T'):launch('qualification','source',arm)
        for arm in ('D','G','T'):launch('formal','source',arm)
        for arm in ('D','G','T'):launch('evaluate','source',arm)
        for arm in ('D','G','T'):launch('qualification','target',arm)
        for arm in ('D','G','T'):launch('formal','target',arm)
        for arm in ('D','G','T'):launch('evaluate','target',arm)
        write(root/'STATUS.json',dict(status='COMPLETE',source_and_target_arms=['D','G','T'],resources=guard(cfg)))
        write(root/'EXIT.json',dict(exit_code=0))
    except BaseException as exc:
        write(root/'FAILURE.private.json',dict(error=str(exc),traceback=traceback.format_exc(),resources=guard(cfg)))
        write(root/'STATUS.json',dict(status='STOPPED',reason=str(exc),resources=guard(cfg)))
        write(root/'EXIT.json',dict(exit_code=1));raise

import json,os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='3';os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
cfg=json.loads(Path(os.environ['TASK_SETTINGS']).read_text());sys.path.insert(0,cfg['base'])
import experiments.lcrseg
experiments.lcrseg.__path__=[str(Path(cfg['code'])/'experiments/lcrseg'),str(Path(cfg['overlay'])/'experiments/lcrseg'),*experiments.lcrseg.__path__]
mode=os.environ['TASK_MODE']
if mode=='checks':
    from experiments.lcrseg.gyro_native_v1.module import self_check
    assert self_check()['status']=='PASS'
    from experiments.lcrseg.gyro_source_repair_v11.core import configure_lr
    from types import SimpleNamespace
    opt=SimpleNamespace(param_groups=[dict(lr=.03,fixed_lr=.001),dict(lr=.001,fixed_lr=.03)])
    configure_lr(opt,0);assert [x['lr'] for x in opt.param_groups]==[.001,.03]
    (Path(cfg['root'])/'SYNTHETIC_CHECKS.json').write_text(json.dumps(dict(status='PASS',optimizer_updates=0,spectral_contracts=True,group_lr_contract=True)))
else:
    if mode=='controller':
        from experiments.lcrseg.gyro_source_repair_v11.controller import main
    elif mode=='qualify':
        from experiments.lcrseg.gyro_source_repair_v11.qualification import qualify as main
    elif mode=='train':
        from experiments.lcrseg.gyro_source_repair_v11.run import main
    else:raise PermissionError('mode')
    main(cfg)

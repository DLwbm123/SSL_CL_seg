"""Copied to a neutral operational path; settings supplied only by environment."""
import json,os,sys,time,traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='3';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
cfg=json.loads(Path(os.environ['TASK_SETTINGS']).read_text());sys.path.insert(0,cfg['base'])
for variable,name in [('TMPDIR','tmp'),('XDG_CACHE_HOME','xdg'),('TORCH_HOME','torch'),('HF_HOME','hf'),('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda')]:
    p=Path(cfg['root'])/'scratch'/name;p.mkdir(parents=True,exist_ok=True);os.environ[variable]=str(p)
import experiments.lcrseg
experiments.lcrseg.__path__=[str(Path(p)/'experiments/lcrseg') for p in cfg['code_roots']]+list(experiments.lcrseg.__path__)
mode=os.environ['TASK_MODE']
if mode=='watchdog':
    from experiments.lcrseg.gyro_spatial_memory_v1.watchdog import main
elif mode=='audit':
    from experiments.lcrseg.gyro_spatial_memory_v1.audit import main
elif mode=='controller':
    from experiments.lcrseg.gyro_spatial_memory_v1.controller import main
else:
    import torch
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if mode=='accesscheck':
        from experiments.lcrseg.gyro_spatial_memory_v1.access_check import main
    elif mode=='poolcheck':
        from experiments.lcrseg.gyro_spatial_memory_v1.pool_check import main
    elif mode=='source':
        from experiments.lcrseg.gyro_spatial_memory_v1.source import run as main
    elif mode in ('train','qualify','evaluate'):
        from experiments.lcrseg.gyro_spatial_memory_v1 import target
        main={'train':target.main,'qualify':target.qualify,'evaluate':target.evaluate}[mode]
    else:raise ValueError('unknown operation')
try:main(cfg)
except BaseException as e:
    if mode=='source':
        try:
            from experiments.lcrseg.gyro_spatial_memory_v1.source import preserve_dirty
            preserve_dirty(cfg)
        except BaseException:
            pass
    out=Path(cfg.get('output',cfg['root']));out.mkdir(exist_ok=True,parents=True)
    (out/'FAILURE.json').write_text(json.dumps(dict(error=str(e),traceback=traceback.format_exc(),time=time.time())))
    raise

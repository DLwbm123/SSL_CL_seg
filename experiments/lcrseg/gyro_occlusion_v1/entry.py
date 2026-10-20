import json,os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='3';os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
cfg=json.loads(Path(os.environ['TASK_SETTINGS']).read_text());sys.path.insert(0,cfg['base'])
import experiments.lcrseg
experiments.lcrseg.__path__=[str(Path(p)/'experiments/lcrseg') for p in cfg['code_roots']]+list(experiments.lcrseg.__path__)
# Wrapper probes the approved volume; all new caches/temp files then stay in this round.
for variable,name in [('TMPDIR','tmp'),('XDG_CACHE_HOME','xdg'),('TORCH_HOME','torch'),('HF_HOME','hf'),('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda')]:
    path=Path(cfg['root'])/'scratch'/name;path.mkdir(parents=True,exist_ok=True);os.environ[variable]=str(path)
mode=os.environ['TASK_MODE']
if mode=='controller':
    from experiments.lcrseg.gyro_occlusion_v1.controller import main
elif mode=='checks':
    from experiments.lcrseg.gyro_occlusion_v1.contracts import main
elif mode=='qualify':
    from experiments.lcrseg.gyro_occlusion_v1.run import qualify as main
elif mode=='audit':
    from experiments.lcrseg.gyro_occlusion_v1.summarize import main as audit
    main=lambda cfg:audit(cfg['root'])
elif mode=='train':
    from experiments.lcrseg.gyro_occlusion_v1.run import main
else:raise PermissionError('mode')
main(cfg)

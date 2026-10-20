import json,os,sys
from pathlib import Path
assert os.environ.get('TASK_MODE')=='diagnose'
os.environ['CUDA_VISIBLE_DEVICES']='3';os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
cfg=json.loads(Path(os.environ['TASK_SETTINGS']).read_text());sys.path.insert(0,cfg['base'])
import experiments.lcrseg
experiments.lcrseg.__path__=[str(Path(p)/'experiments/lcrseg') for p in cfg['code_roots']]+list(experiments.lcrseg.__path__)
for variable,name in [('TMPDIR','tmp'),('XDG_CACHE_HOME','xdg'),('TORCH_HOME','torch'),('HF_HOME','hf'),('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda')]:
    p=Path(cfg['root'])/'scratch'/name;p.mkdir(parents=True,exist_ok=True);os.environ[variable]=str(p)
from experiments.lcrseg.gyro_mechanism_diag_v1.run import main
main(cfg)

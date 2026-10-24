"""Neutral deployment launcher; paths are supplied through private environment settings."""
import os,json,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='3';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
cfg=json.loads(Path(os.environ['TASK_SETTINGS']).read_text())
sys.path.insert(0,cfg['base'])
for key,name in [('TMPDIR','tmp'),('TORCH_HOME','torch'),('XDG_CACHE_HOME','xdg'),('HF_HOME','hf'),('CUDA_CACHE_PATH','cuda'),('TRITON_CACHE_DIR','triton')]:
    p=Path(cfg['root'])/'scratch'/name;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
import experiments.lcrseg
experiments.lcrseg.__path__=[str(Path(cfg['root'])/'code/experiments/lcrseg')]+[str(Path(p)/'experiments/lcrseg') for p in cfg['code_roots']]+list(experiments.lcrseg.__path__)
from importlib import import_module
import_module('experiments.lcrseg.apos_v2.'+cfg.get('module','controller')).main(cfg)

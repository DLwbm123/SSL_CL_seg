import os,json,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='3';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
cfg=json.loads(Path(os.environ['TASK_SETTINGS']).read_text());sys.path.insert(0,cfg['base'])
for variable,name in [('TMPDIR','tmp'),('XDG_CACHE_HOME','xdg'),('TORCH_HOME','torch'),('HF_HOME','hf'),('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda')]:
 p=Path(cfg['root'])/'scratch'/name;p.mkdir(parents=True,exist_ok=True);os.environ[variable]=str(p)
import experiments.lcrseg
experiments.lcrseg.__path__=[str(Path(p)/'experiments/lcrseg') for p in cfg['code_roots']]+list(experiments.lcrseg.__path__)
import experiments.lcrseg.gyro_spatial_memory_v1 as package
package.__path__.insert(0,str(Path(cfg['root'])/'code'))
package.__path__.append(str(Path(cfg["regional_root"])/"code"))
package.__path__.append(str(Path(cfg["compatible_root"])/"code"))
package.__path__.append(str(Path(cfg["group_risk_root"])/"code"))
package.__path__.append(str(Path(cfg["reliable_root"])/"code"))
package.__path__.append(str(Path(cfg["decision_root"])/"code"))
package.__path__.append(str(Path(cfg["ema_root"])/"code"))
from importlib import import_module
mod=import_module('experiments.lcrseg.gyro_spatial_memory_v1.'+cfg['module'])
if os.environ.get('TASK_MODE')=='cpucheck':
 import torch
 torch.set_num_threads(4)
 (Path(cfg['root'])/'CPU_SELF_CHECK.json').write_text(json.dumps(mod.self_check(),indent=2))
else:mod.main(cfg)

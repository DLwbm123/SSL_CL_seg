"""Copy to a neutral launcher path; all private paths arrive via environment/config."""
import json
import os
import sys
from pathlib import Path

if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,'3'):
    raise PermissionError('outside frozen GPU scope')
os.environ['CUDA_VISIBLE_DEVICES']='3'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
cfg=json.loads(Path(os.environ['TASK_SETTINGS']).read_text())
sys.path.insert(0,cfg['base'])
import experiments.lcrseg
experiments.lcrseg.__path__=[str(Path(cfg['code'])/'experiments/lcrseg'),
    str(Path(cfg['overlay'])/'experiments/lcrseg'),*experiments.lcrseg.__path__]
mode=os.environ['TASK_MODE']
if mode=='statistics':
    from experiments.lcrseg.semantic_condition_v1.statistics import main
elif mode=='qualify':
    from experiments.lcrseg.semantic_condition_v1.qualification import qualify as main
elif mode=='train':
    from experiments.lcrseg.semantic_condition_v1.run import main
elif mode=='controller':
    from experiments.lcrseg.semantic_condition_v1.controller import main
elif mode=='summarize':
    from experiments.lcrseg.semantic_condition_v1.summarize import main
    cfg=cfg['root']
else:
    raise ValueError('unregistered task mode')
main(cfg)

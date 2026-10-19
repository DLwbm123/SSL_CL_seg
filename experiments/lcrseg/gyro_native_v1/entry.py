"""Neutral entry; task paths and names exist only in environment, never argv."""
import json
import os
import sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='3'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
cfg=json.loads(Path(os.environ['TASK_SETTINGS']).read_text())
sys.path.insert(0,cfg['base'])
import experiments.lcrseg
experiments.lcrseg.__path__=[str(Path(cfg['code'])/'experiments/lcrseg'),str(Path(cfg['overlay'])/'experiments/lcrseg'),*experiments.lcrseg.__path__]
mode=os.environ['TASK_MODE']
if mode=='checks':
    from experiments.lcrseg.gyro_native_v1.module import self_check
    (Path(cfg['root'])/'SYNTHETIC_CHECKS.json').write_text(json.dumps(self_check(),indent=2))
else:
    if mode=='controller':
        from experiments.lcrseg.gyro_native_v1.controller import main
    elif mode=='qualify':
        from experiments.lcrseg.gyro_native_v1.qualification import qualify as main
    elif mode=='train':
        from experiments.lcrseg.gyro_native_v1.run import main
    else: raise PermissionError('unknown execution mode')
    main(cfg)

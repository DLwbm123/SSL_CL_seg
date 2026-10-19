import json,os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='3';os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
cfg=json.loads(Path(os.environ['TASK_SETTINGS']).read_text());sys.path.insert(0,cfg['base'])
import experiments.lcrseg
experiments.lcrseg.__path__=[str(Path(cfg['code'])/'experiments/lcrseg'),str(Path(cfg['previous_code'])/'experiments/lcrseg'),str(Path(cfg['overlay'])/'experiments/lcrseg'),*experiments.lcrseg.__path__]
mode=os.environ['TASK_MODE']
if mode=='checks':
    from experiments.lcrseg.gyro_source_repair_v12.contracts import check
    check(cfg)
elif mode=='controller':
    from experiments.lcrseg.gyro_source_repair_v12.controller import main
    main(cfg)
elif mode=='qualify':
    from experiments.lcrseg.gyro_source_repair_v12.run import qualify
    qualify(cfg)
elif mode=='train':
    from experiments.lcrseg.gyro_source_repair_v12.run import main
    main(cfg)
else:raise PermissionError('mode')

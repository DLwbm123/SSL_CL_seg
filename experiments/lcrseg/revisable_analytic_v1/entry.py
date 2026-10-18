"""Copy this entry to a neutral operational path before launching on GPU3."""
import os
import sys
import json
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
cfg=json.loads(Path(os.environ['TASK_SETTINGS']).read_text())
sys.path.insert(0,cfg['base'])
import experiments.lcrseg
experiments.lcrseg.__path__=[str(Path(cfg['code'])/'experiments/lcrseg'),str(Path(cfg['support'])/'experiments/lcrseg'),*experiments.lcrseg.__path__]
mode=os.environ.get('TASK_MODE','run')
if mode=='run':
    from experiments.lcrseg.revisable_analytic_v1.run import main
    main(cfg)
elif mode=='summary':
    from experiments.lcrseg.revisable_analytic_v1.summarize import main
    main(Path(cfg['output']).parent)
elif mode=='geometry':
    from experiments.lcrseg.revisable_analytic_v1.diagnose import main
    main(cfg)
else: raise PermissionError('unregistered mode')

"""Neutral process entry; paths and task identifiers arrive only via environment."""
import json
import os
from pathlib import Path
import sys
os.environ['CUDA_VISIBLE_DEVICES']='3'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
cfg=json.loads(Path(os.environ['TASK_SETTINGS']).read_text())
sys.path.insert(0,cfg['base'])
import experiments.lcrseg
experiments.lcrseg.__path__=[str(Path(cfg['code'])/'experiments/lcrseg'),
                           str(Path(cfg['overlay'])/'experiments/lcrseg'),*experiments.lcrseg.__path__]
mode=os.environ['TASK_MODE']
if mode=='checks':
    from experiments.lcrseg.functional_evidence_v1.state import self_check
    (Path(cfg['root'])/'SYNTHETIC_CHECKS.json').write_text(json.dumps(self_check(),indent=2))
else:
    if mode=='bank':
        from experiments.lcrseg.functional_evidence_v1.bank import main
    elif mode=='qualify':
        from experiments.lcrseg.functional_evidence_v1.qualification import qualify as main
    elif mode=='train':
        from experiments.lcrseg.functional_evidence_v1.run import main
    elif mode=='controller':
        from experiments.lcrseg.functional_evidence_v1.controller import main
    elif mode=='summarize':
        from experiments.lcrseg.functional_evidence_v1.summarize import main
    else: raise PermissionError('unregistered mode')
    main(cfg)

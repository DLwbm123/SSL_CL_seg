"""Validate the two new full-pipeline seeds and fixed paired target settings."""
import json,os
from pathlib import Path
import torch
from .model import initialize
from .train import setup
from sc_bayeseg.window_v1.r_start import write

def run(s):
    setup();tasks={t[0]:t for t in s['tasks']};assert len(tasks)==len(s['tasks'])==6
    assert not s['training_seeds'] and not s['source_paths']
    states=[]
    for seed in (166,167):
        src=f'source_B_s{seed}';assert tasks[src]==[src,'B','source',True,False]
        for arm in ('B10','BHALF'):
            name=f'{arm}_s{seed}';assert tasks[name]==[name,'B','target',True,arm=='BHALF']
            assert s['source_tasks'][name]==src
            assert s['task_options'][name]['history_strength']==(.5 if arm=='BHALF' else 1.)
        for name in (src,f'B10_s{seed}',f'BHALF_s{seed}'):
            assert s['task_options'][name]['initialization_seed']==s['task_options'][name]['schedule_seed']==seed
        m=initialize(dict(s,initialization_seed=seed),'B')
        states.append({k:v.detach().cpu().clone() for k,v in m.backbone.state_dict().items()});del m
    assert any(not torch.equal(v,states[1][k]) for k,v in states[0].items())
    write(Path(s['root'])/'INTEGRATION.json',dict(status='PASS',optimizer_updates=0,between_seed_backbone_different=True,source_dependencies_match=True,fixed_half_strength=True,target_RNG_restore=True,inherited_real_data_integration='sslcl_prostate_halfstyle_20261022/INTEGRATION.json'))
if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))

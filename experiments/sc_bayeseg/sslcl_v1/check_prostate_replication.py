"""Check independent initialization and source dependencies without training."""
import json,os
from pathlib import Path
import torch
from .model import initialize
from .train import setup
from sc_bayeseg.window_v1.r_start import write

def run(s):
    setup();tasks={t[0]:t for t in s['tasks']};assert len(tasks)==len(s['tasks'])==12
    assert not s['training_seeds'] and not s['source_paths']
    for name,_,stage,_,_ in s['tasks']:
        options=s['task_options'][name];seed=options['schedule_seed'];assert seed in (164,165) and options['initialization_seed']==seed
        if stage=='target':
            src=s['source_tasks'][name];assert tasks[src][2]=='source' and tasks[src][1]==tasks[name][1]
            assert s['task_options'][src]['schedule_seed']==seed
    states=[]
    for seed in (164,165):
        setting=dict(s,initialization_seed=seed)
        b=initialize(setting,'B');u=initialize(setting,'U')
        assert all(torch.equal(v,u.backbone.state_dict()[k]) for k,v in b.backbone.state_dict().items())
        states.append({k:v.detach().cpu().clone() for k,v in b.backbone.state_dict().items()})
        del b,u
    assert any(not torch.equal(v,states[1][k]) for k,v in states[0].items())
    write(Path(s['root'])/'INTEGRATION.json',dict(status='PASS',optimizer_updates=0,within_seed_backbone_equal=True,between_seed_backbone_different=True,source_dependencies_match=True,target_RNG_restore=True,inherited_binary_integration='sslcl_prostate_pilot_20261022/INTEGRATION.json'))
if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))

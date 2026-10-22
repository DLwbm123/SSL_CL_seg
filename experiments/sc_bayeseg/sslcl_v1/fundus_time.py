"""Exactly six student checkpoints per seed; no optimizer or EMA inference."""
import json,os,time
from pathlib import Path
import torch
from .train import setup
from .model import initialize
from .fundus import evaluate_fundus
from sc_bayeseg.window_v1.r_start import write


def run(s):
    setup();out=Path(s['output']);out.mkdir(exist_ok=False)
    assert json.loads((Path(s['pairing'])/'CASE_PAIRING.json').read_text())['status']=='PASS'
    seed=s['diagnostic_seed'];model=initialize(s,'U');results={};count=0
    # The existing evaluator invokes probability exactly once per image.
    probability=model.probability
    def counted(x):
        nonlocal count
        count+=len(x)
        assert count<=840
        write(out/'FORWARD_COUNT.json',dict(image_forwards=count,optimizer_updates=0))
        return probability(x)
    model.probability=counted
    for arm in ('F0','FG','FH'):
        for step in (1000,3000):
            assert time.time()-s['original_started_unix']<85200
            ck=torch.load(Path(s['baseline'])/f'{arm}_s{seed}'/f'S{step:04}.pt',map_location='cpu',weights_only=False)
            assert ck['step']==step;model.load_state_dict(ck['model']);del ck
            folder=out/f'{arm}_S{step:04}';folder.mkdir(exist_ok=False)
            results[f'{arm}_S{step:04}']=evaluate_fundus(model,s,folder)
    assert count==840
    write(out/'RESULTS.json',dict(status='COMPLETE',metrics=results,image_forwards=count,optimizer_updates=0,seed=seed))
    write(out/'STATUS.json',dict(status='COMPLETE',image_forwards=count))
if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))

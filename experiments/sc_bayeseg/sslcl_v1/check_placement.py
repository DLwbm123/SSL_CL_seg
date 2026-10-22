"""Validate paired full-strength branch interventions without optimization."""
import copy,json,os
from pathlib import Path
import torch
from .data import Schedule
from .model import initialize
from .train import setup,step_loss
from sc_bayeseg.window_v1.r_start import write

def run(s):
    setup();assert len(s['tasks'])==len(s['source_paths'])==12 and not s['training_seeds'];seen=set();results={}
    x,y,_=Schedule(s['data'],'BMC',dataset='prostate').batch(5001)
    u,uy,_=Schedule(s['data'],'BMC',True,dataset='prostate').batch(5001);assert uy is None
    for name,family,stage,use_u,history in s['tasks']:
        assert name not in seen;seen.add(name);opt=s['task_options'][name];seed=opt['schedule_seed'];kind=name[1:3]
        assert seed in (163,164,165) and opt['initialization_seed']==seed
        assert stage=='target' and use_u and history and opt['history_strength']==1.
        assert opt['history_l']==(kind=='HL') and opt['history_u']==(kind=='HU') and kind in ('HL','HU')
        suffix='' if seed==163 else f'_s{seed}';assert Path(s['source_paths'][name]).parent.name=='source_'+family+suffix
        ck=torch.load(s['source_paths'][name],map_location='cpu',weights_only=False);assert ck['step']==5000 and ck['family']==family
        if seed!=163:continue
        m=initialize(s,family);m.load_state_dict(ck['model']);t=copy.deepcopy(m).eval().requires_grad_(False);t.load_state_dict(ck['teacher'])
        loss,parts=step_loss(m,t,x,y,u,s['styles']['BMC'],s['styles']['RUNMC'],1000,history_l=opt['history_l'],history_u=opt['history_u'])
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
        assert all(p.grad is None for p in t.parameters());results[family+kind]=dict(loss=float(loss.detach()),parts=parts)
    write(Path(s['root'])/'INTEGRATION.json',dict(status='PASS',optimizer_updates=0,source_mappings=12,branch_configs_checked=True,finite_backward=results))
if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))

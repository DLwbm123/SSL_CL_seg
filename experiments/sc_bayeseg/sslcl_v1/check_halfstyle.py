"""Verify half-strength views and real-data gradients without optimizer updates."""
import copy,json,math,os
from pathlib import Path
import torch
from .data import Schedule,view
from .model import initialize
from .train import setup,step_loss
from sc_bayeseg.window_v1.r_start import write

def run(s):
    setup();current=s['styles']['BMC'];history=s['styles']['RUNMC']
    x,y,_=Schedule(s['data'],'BMC',dataset='prostate').batch(5001)
    u,uy,_=Schedule(s['data'],'BMC',True,dataset='prostate').batch(5001);assert uy is None
    assert torch.equal(view(x,current,history,1,history_strength=0),view(x,current,None,1))
    half=[current[0]+.5*(history[0]-current[0]),current[1]*math.exp(.5*math.log(history[1]/current[1]))]
    assert torch.equal(view(x,current,history,1,history_strength=.5),view(x,current,half,1))
    assert len(s['tasks'])==len(s['source_paths'])==6 and not s['training_seeds']
    rows={}
    for name,family,stage,use_u,hist in s['tasks']:
        assert stage=='target' and use_u and hist and s['task_options'][name]['history_strength']==.5
        seed=s['task_options'][name]['schedule_seed'];expected=('' if seed==163 else '_s'+str(seed))
        assert Path(s['source_paths'][name]).parent.name=='source_'+family+expected
        ck=torch.load(s['source_paths'][name],map_location='cpu',weights_only=False);assert ck['step']==5000 and ck['family']==family
        if seed!=163:continue
        m=initialize(s,family);m.load_state_dict(ck['model']);t=copy.deepcopy(m).eval().requires_grad_(False);t.load_state_dict(ck['teacher'])
        loss,parts=step_loss(m,t,x,y,u,current,history,1000,history_strength=.5)
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
        assert all(p.grad is None for p in t.parameters())
        rows[family]=dict(loss=float(loss.detach()),parts=parts)
    write(Path(s['root'])/'INTEGRATION.json',dict(status='PASS',optimizer_updates=0,zero_strength_view_exact=True,half_strength_matches_interpolated_stats_exact=True,source_mapping_checked=6,finite_backward=rows))
if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))

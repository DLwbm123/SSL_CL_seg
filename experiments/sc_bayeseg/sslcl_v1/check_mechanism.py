"""Zero-update integration of frozen history routing and style controls."""
import copy
import json
import os
import math
from pathlib import Path
import torch
from .data import Schedule,view
from .model import initialize
from .train import setup,step_loss
from sc_bayeseg.window_v1.r_start import write


def run(s):
    setup();current=s['styles']['Philips'];history=s['styles']['Siemens']
    x,y,_=Schedule(s['data'],'Philips',seed=164).batch(5001)
    u,_,_=Schedule(s['data'],'Philips',True,seed=164).batch(5001)
    a=view(x,current,None,1,164);b=view(x,current,current,1,164)
    assert torch.equal(a,b)
    w=view(x,current,None,1,164,'wide');assert torch.equal(w,view(x,current,[999.,999.],1,164,'wide'))
    assert torch.equal(w,view(x,current,None,1,164,'wide'))
    reverse=[2*current[0]-history[0],current[1]**2/history[1]]
    assert abs((reverse[0]-current[0])+(history[0]-current[0]))<1e-10
    assert abs(math.log(reverse[1]/current[1])+math.log(history[1]/current[1]))<1e-10
    ck=torch.load(s['source_paths']['B'],map_location='cpu',weights_only=False)
    model=initialize(s,'B');model.load_state_dict(ck['model']);teacher=copy.deepcopy(model).eval().requires_grad_(False);teacher.load_state_dict(ck['teacher'])
    rows={}
    for name in ('HL','HU','WIDE','REVERSE'):
        torch.manual_seed(164);model.zero_grad(set_to_none=True)
        options=s['task_options'][name];h=None if name=='WIDE' else history
        loss,parts=step_loss(model,teacher,x,y,u,current,h,1,164,**options)
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        assert all(p.grad is None for p in teacher.parameters())
        views={}
        for label,z,enabled,step in [('L',x,options.get('history_l',True),1),('U',u,options.get('history_u',True),500001)]:
            v=view(z,current,h if enabled else None,step,164,options.get('style_mode','history') if enabled else 'history')
            assert torch.isfinite(v).all()
            views[label]=dict(RMS_change=float((v-z).square().mean().sqrt()),minimum=float(v.min()),maximum=float(v.max()))
        rows[name]=dict(loss=float(loss.detach()),parts=parts,views=views)
    write(Path(s['root'])/'INTEGRATION.json',dict(status='PASS',optimizer_updates=0,real_backward_checks=4,identity_exact=True,wide_independent_of_history=True,reverse_equal_logstat_amplitude=True,arms=rows))


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))

"""B11 legacy grayscale parity and RGB mechanism qualification; no optimizer updates."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import torch
from sc_bayeseg.sslcl_v1.model import Model
from sc_bayeseg.sslcl_v1.train import setup, step_loss
from sc_bayeseg.sslcl_v1.data import Schedule
from lcrseg.engine.checkpoint import restore_rng_state


def run(s):
    setup()
    spec = importlib.util.spec_from_file_location('legacy_b11', s['legacy_model'])
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    results = {}
    for key in ('C0', 'P0'):
        t = s['task_options'][key]
        ck = torch.load(s['source_paths'][key], map_location='cpu', weights_only=False)
        assert ck['step'] == 5000 and ck['family'] == 'B'
        a = old.Model('B', t['num_classes']).cuda(); b = Model('B', t['num_classes']).cuda()
        a.load_state_dict(ck['model']); b.load_state_dict(ck['model'])
        teacher = copy.deepcopy(b).eval().requires_grad_(False); teacher.load_state_dict(ck['teacher'])
        x,y,_ = Schedule(t['data'], t['domain'], dataset=t['dataset']).batch(5001)
        u = Schedule(t['data'], t['domain'], True, dataset=t['dataset']).batch(5001)[0]
        losses=[]; grads=[]
        for model in (a,b):
            restore_rng_state(ck['rng'])
            loss, parts = step_loss(model,teacher,x,y,u,t['styles'][t['domain']],t['styles'][t['source_domain']],1)
            loss.backward(); losses.append(loss.detach().clone())
            grads.append({n:p.grad.detach().clone() for n,p in model.named_parameters() if p.requires_grad})
            assert all(torch.isfinite(g).all() for g in grads[-1].values())
        assert torch.equal(*losses)
        assert grads[0].keys()==grads[1].keys() and all(torch.equal(grads[0][n],grads[1][n]) for n in grads[0])
        assert all(p.grad is None for p in teacher.parameters())
        results[key] = dict(real_L_U_loss=float(losses[0]), exact_loss_and_all_gradients=True, teacher_no_gradient=True, source_B_step=5000)
        del a,b,teacher,grads,ck
    torch.manual_seed(163)
    b=Model('B',3,3).cuda(); teacher=copy.deepcopy(b).eval().requires_grad_(False)
    t=s['task_options']['FS']
    x,y,_=Schedule(t['data'],'REFUGE',dataset='fundus').batch(1)
    u=Schedule(t['data'],'REFUGE',True,dataset='fundus').batch(1)[0]
    loss,parts=step_loss(b,teacher,x,y,u,s['styles']['REFUGE'],None,1)
    loss.backward()
    gradient_groups={prefix: all(p.grad is not None and torch.isfinite(p.grad).all() for n,p in b.named_parameters() if n.startswith(prefix) and p.requires_grad) for prefix in ('backbone','hc','hr','hs','hm','g','f')}
    assert all(gradient_groups.values()) and torch.isfinite(loss)
    with torch.no_grad():
        p=b.probability(torch.nn.functional.interpolate(x[:1],(384,384),mode='bilinear',align_corners=False))
    assert p.shape==(1,3,384,384) and torch.isfinite(p).all() and torch.allclose(p.sum(1),torch.ones_like(p[:,0]),atol=1e-6)
    results['RGB']=dict(real_L_U_loss=float(loss), finite_gradients=gradient_groups, native_probability_shape=list(p.shape), parts=parts)
    report=dict(status='PASS', checks=results, optimizer_updates=0, backwards=5, test_access=False)
    (Path(s['root'])/'INTEGRATION.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))

if __name__=='__main__': run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))

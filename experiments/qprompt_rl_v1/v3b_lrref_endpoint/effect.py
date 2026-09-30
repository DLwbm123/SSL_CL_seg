"""Native Adam previews; feature equations inherited from V3A2."""
import copy
import torch
from torch.optim.adam import adam

def preview(opt, parameters, gradients):
    assert type(opt) is torch.optim.Adam
    index={id(p):i for i,p in enumerate(parameters)}
    values=[p.detach().clone() for p in parameters]; states={}
    with torch.no_grad():
        for group in opt.param_groups:
            assert not group['differentiable'] and not group['capturable']
            ps=[]; gs=[]; ms=[]; vs=[]; mx=[]; steps=[]
            for p in group['params']:
                i=index[id(p)]; g=gradients[i]
                if g is None: continue
                assert p.dtype==torch.float32 and not g.is_sparse and torch.isfinite(g).all()
                state=opt.state.get(p,{})
                q=copy.deepcopy(state) if state else dict(step=torch.tensor(0.),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
                if group['amsgrad'] and 'max_exp_avg_sq' not in q:q['max_exp_avg_sq']=torch.zeros_like(p)
                states[i]=q; ps.append(values[i]);gs.append(g.detach().clone());ms.append(q['exp_avg']);vs.append(q['exp_avg_sq']);steps.append(q['step'])
                if group['amsgrad']:mx.append(q['max_exp_avg_sq'])
            adam(ps,gs,ms,vs,mx,steps,amsgrad=group['amsgrad'],beta1=group['betas'][0],beta2=group['betas'][1],lr=group['lr'],weight_decay=group['weight_decay'],eps=group['eps'],maximize=group['maximize'],foreach=group['foreach'],capturable=group['capturable'],differentiable=False,fused=group.get('fused'),grad_scale=None,found_inf=None,has_complex=False)
    return values,states

def dot(a,b):
    result=0.
    for x,y in zip(a,b):
        if x is None or y is None:continue
        x=x.reshape(-1);y=y.reshape(-1)
        for start in range(0,x.numel(),262144):result+=float((x[start:start+262144].double()*y[start:start+262144].double()).sum())
    return result

def difference(a,b,parameters):return [(torch.zeros_like(p) if x is None else x)-(torch.zeros_like(p) if y is None else y) for x,y,p in zip(a,b,parameters)]
def features(parameters,names,g,q,updates):
    deltas=[[u-p.detach() for u,p in zip(update,parameters)] for update in updates];out=[]
    body=[n.startswith('parent.native.') and n.endswith(('.A','.B')) for n in names];assert all(body), 'unmapped trainable parameter'
    for action in (0,1):
        for use_body in (True,False):
            ids=[i for i,yes in enumerate(body) if yes==use_body];par=[parameters[i] for i in ids];ga=[g[action][i] for i in ids];gf=[g[2][i] for i in ids];qq=[q[i] for i in ids];df=[deltas[2][i] for i in ids];delta=[deltas[action][i]-deltas[2][i] for i in ids];dg=difference(ga,gf,par)
            norm=lambda v:dot(v,v)**.5
            nq,nd,nf,ng=norm(qq),norm(delta),norm(df),norm(gf);score=-dot(qq,delta)
            import math
            out.extend([score,score/(nq*nd) if nq*nd else 0.,math.log1p(nd/(nf+1e-12)),math.log1p(norm(dg)/(ng+1e-12)),float(nq>0 and nd>0),float(nf>0 and ng>0)])
    return out

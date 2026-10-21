"""Read-only endpoint precision and finite-fixture expressibility diagnostics."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import validate_nonlinear_stochastic as v


def main():
    root=Path(__file__).parent/'results'
    source=json.loads((root/'BAYESEG_G_NONLINEAR_STOCHASTIC.json').read_text())
    _,im=v.base.data(torch.float32);ii=im.double();rows=[]
    for run in source['runs']:
        w=torch.tensor(run['weights'],dtype=torch.float32,requires_grad=True)
        g=torch.autograd.grad(v.terms(w,im,torch.float64)[0].mean(),w)[0]
        wd=w.detach().double().requires_grad_(True)
        ref=torch.autograd.grad(v.terms(wd,ii,torch.float64)[0].mean(),wd)[0]
        error=float(((g.double()-ref).abs()/(1+ref.abs())).max())
        a,b,c,d,e,f=wd.detach().split(v.SIZES)
        z=ii.mean((-1,-2));z=torch.stack([(z[:,0]+.5)/1.5,z[:,1]/2],1)
        h1=torch.tanh(F.linear(z,a.reshape(32,2),b));h2=torch.tanh(F.linear(h1,c.reshape(32,32),d))
        design=torch.cat([h2,torch.ones(12,1,dtype=torch.float64)],1)
        sv=torch.linalg.svdvals(design)
        t=10**ii[:,0,0,0];x=ii[:,1,0,0]
        sd=torch.stack([torch.sqrt((1+t*t)/(2+t*t)),torch.sqrt(t*t/(1+t*t))],1)
        target=torch.column_stack([x/(2+t*t),x/(2+t*t),(torch.log(torch.expm1(sd-1e-6))-v.B0)/4,1/(1+t*t)-.01])
        # Diagnostic exact assignment only; never changes a recorded endpoint or starts optimization.
        coef=torch.linalg.lstsq(design,target,driver='gelsd').solution
        assignment=torch.cat([a,b,c,d,coef[:-1].T.reshape(-1),coef[-1]])
        control=v.evaluate(assignment,ii)
        rows.append(dict(compute=run['compute'],stochastic=run['stochastic'],mixed_endpoint_gradient_error=error,mixed_endpoint_gradient_pass=error<2e-6,feature_singular_values=sv.tolist(),feature_condition=float(sv.max()/sv.min()),feature_rank=int((sv>sv.max()*1e-12).sum()),hidden_abs_over_099=float((h2.abs()>.99).double().mean()),exact_assignment_max_KL=control['max_KL'],exact_assignment_pass= max(abs(q['KL']) for q in control['cases'])<1e-9))
    path=root/'BAYESEG_G_NONLINEAR_DIAGNOSTICS.json';assert not path.exists()
    path.write_text(json.dumps(dict(rows=rows,optimizer_updates=0,endpoint_gradient_requests=8,reference_assignments=4,limitation='Post-hoc finite-fixture expression control; no target assignment used in training or endpoint replacement'),indent=2)+'\n')
    print(json.dumps(rows))

if __name__=='__main__':main()

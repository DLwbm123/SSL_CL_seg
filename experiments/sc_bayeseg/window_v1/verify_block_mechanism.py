"""Read-only parameter-gradient check at the five recorded endpoints."""
import json
from pathlib import Path
import torch
import validate_nonlinear_stochastic as v

def main():
    root=Path(__file__).parent/'results';x=json.loads((root/'BAYESEG_G_BLOCK_MECHANISM.json').read_text());_,im=v.base.data(torch.float32);out=[]
    for r in x['runs']:
        dtype=torch.float64 if r['name']=='J64' else torch.float32
        w=torch.tensor(r['weights'],dtype=dtype,requires_grad=True)
        g=torch.autograd.grad(v.terms(w,im,torch.float64)[0].mean(),w)[0]
        wd=w.detach().double().requires_grad_(True)
        ref=torch.autograd.grad(v.terms(wd,im.double(),torch.float64)[0].mean(),wd)[0]
        error=float(((g.double()-ref).abs()/(1+ref.abs())).max())
        out.append(dict(name=r['name'],gradient_error=error,pass_original_gate=error<2e-6))
    p=root/'BAYESEG_G_BLOCK_VERIFY.json';assert not p.exists()
    p.write_text(json.dumps(dict(checks=out,gradient_requests=10,optimizer_updates=0),indent=2)+'\n')
if __name__=='__main__':main()

"""Post-run joint directional check; no optimization or medical inputs."""
import json
from pathlib import Path
import torch
import validate_neural_kappa as base


def main():
    root=Path(__file__).parent/'results'
    source=json.loads((root/'BAYESEG_G_NEURAL_KAPPA_JOINT_REPAIR.json').read_text())
    out=[]
    for run in source['runs'][1:]:
        w=torch.tensor(run['weights'],dtype=torch.float64,requires_grad=True)
        _,im=base.data(torch.float64);im.requires_grad_(True)
        g,gi=torch.autograd.grad(base.kl(w,im,'linear').mean(),(w,im))
        dw=torch.cos(torch.arange(85,dtype=torch.float64)).reshape(5,17)
        di=torch.sin(torch.arange(im.numel(),dtype=torch.float64)).reshape_as(im)
        h=1e-5
        with torch.no_grad():
            num=(base.kl(w+h*dw,im+h*di,'linear').mean()-base.kl(w-h*dw,im-h*di,'linear').mean())/(2*h)
        analytic=(g*dw).sum()+(gi*di).sum()
        error=float(abs(num-analytic)/(1+abs(num)));assert error<2e-6
        out.append(dict(name=run['name'],joint_directional_error=error,pass_original_gradient_gate=error<2e-6))
    path=root/'BAYESEG_G_NEURAL_KAPPA_JOINT_VERIFY.json';assert not path.exists()
    path.write_text(json.dumps(dict(checks=out,extra_kl_forwards=6,extra_gradient_requests=2,optimizer_updates=0),indent=2)+'\n')

if __name__=='__main__':main()

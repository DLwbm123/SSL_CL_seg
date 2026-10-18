"""Read-only, CPU spectral diagnostics of an already completed endpoint."""
import torch
from pathlib import Path
from .core import ARMS
from ..lctx_weight_memory_v0_1.core import LAYERS
from ..main_style_ssl_v1.runtime import write


def main(root):
    root=Path(root); result={}
    torch.set_num_threads(4)
    for arm in ARMS:
        initial=torch.load(root/'jobs'/arm/'initial.pt',map_location='cpu',weights_only=False)['model']
        checkpoint=torch.load(root/'jobs'/arm/'final.pt',map_location='cpu',weights_only=False)
        assert checkpoint['step']==checkpoint['physical_updates']==4000
        state=checkpoint['model']
        rows=[]
        for name in LAYERS:
            key='parent.native.'+name; module=key[:-7]
            base=initial[key].double().flatten(1)
            delta=(state[key].double().flatten(1)-base) if arm=='dense_fixmatch' else state[module+'.B'].double()@state[module+'.A'].double()
            sigma=torch.linalg.svdvals(delta); energy=sigma.square(); total=energy.sum()
            def erank(mass):
                p=mass/mass.sum().clamp_min(1e-300);p=p[p>0]
                return float((-(p*p.log()).sum()).exp()) if mass.sum()>0 else 0.
            rows.append(dict(layer=name,frobenius_norm=float(total.sqrt()),relative_to_base=float(total.sqrt()/base.norm()),
                             effective_rank_singular=erank(sigma),effective_rank_energy=erank(energy),
                             stable_rank=float(total/energy[0]) if total>0 else 0.))
        result[arm]=dict(layers=rows,mean_effective_rank_energy=sum(x['effective_rank_energy'] for x in rows)/len(rows),
                        mean_effective_rank_singular=sum(x['effective_rank_singular'] for x in rows)/len(rows),
                        total_frobenius_norm=sum(x['frobenius_norm']**2 for x in rows)**.5)
    write(root/'SPECTRAL_DIAGNOSTICS.json',dict(endpoint=4000,optimizer_calls=0,checkpoint_only=True,arms=result,
          limitation='student residual spectrum only; no causal attribution or EMA/optimizer-step spectral guarantee'))
    print({k:{a:b for a,b in v.items() if a!='layers'} for k,v in result.items()})

"""A fixed supervised-gradient norm bound on B U; Adam step is not bounded."""
from pathlib import Path
import torch
from ..f5_kl_output_u_v4.model import SelectiveUTrainer
from ..five_frameworks_v1.gate import digest


def cap_group(labeled,unlabeled):
    if labeled.shape!=unlabeled.shape:raise ValueError('gradient shapes differ')
    if not torch.isfinite(labeled).all() or not torch.isfinite(unlabeled).all():
        raise FloatingPointError('nonfinite norm-cap input')
    l,u=labeled.double(),unlabeled.double();nl,nu=l.norm(),u.norm()
    scale=torch.minimum(nl/nu.clamp_min(torch.finfo(torch.float64).tiny),nl.new_ones(())) if nu>0 else nl.new_ones(())
    filtered=unlabeled*scale.to(unlabeled);after=filtered.double().norm()
    if after>nl*(1+1e-6)+torch.finfo(torch.float64).tiny:raise FloatingPointError('B U norm bound violated')
    return filtered,dict(capped=bool(scale<1),scale=float(scale),L_B_norm=float(nl),U_B_norm_before=float(nu),U_B_norm_after=float(after))


class NormCapTrainer(SelectiveUTrainer):
    def __init__(self,*args,cap_enabled=True,**kwargs):
        self.cap_enabled=cap_enabled
        super().__init__(*args,**kwargs)
        self.telemetry.update(B_cap_calls=0,B_capped_calls=0,B_effective_nonzero_U_calls=0)

    def loss_contract(self):
        return {**super().loss_contract(),'equation':'A:L; B:L+norm_bounded_KL_U; R:L+KL_U+SWD_U',
                'B_cap_enabled':self.cap_enabled,'B_cap_implementation':digest(Path(__file__).read_text())}

    def transform_gradients(self,gradients):
        super().transform_gradients(gradients)
        params=self.model.parent.parameter_groups()['output_factors']
        if not self.cap_enabled or not any(gradients[id(p)]['U'] is not None for p in params):return
        def flat(key):return torch.cat([(torch.zeros_like(p) if gradients[id(p)][key] is None else gradients[id(p)][key]).flatten() for p in params])
        filtered,stats=cap_group(flat('L'),flat('U'));offset=0
        for p in params:
            pair=gradients[id(p)];u=filtered[offset:offset+p.numel()].reshape_as(p);offset+=p.numel()
            if pair['L'] is not None or pair['U'] is not None:
                p.grad=(torch.zeros_like(p) if pair['L'] is None else pair['L'])+u
        self.telemetry['B_cap_calls']+=1;self.telemetry['B_capped_calls']+=int(stats['capped'])
        self.telemetry['B_effective_nonzero_U_calls']+=int(stats['U_B_norm_after']>0)
        self.module_stats.update(stats)

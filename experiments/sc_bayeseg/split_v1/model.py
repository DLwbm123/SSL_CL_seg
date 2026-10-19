"""Same generative model; independent semantic and physical inference paths."""
import copy
import torch
from sc_bayeseg.model import Model as OriginalModel


class Model(OriginalModel):
    def __init__(self):
        super().__init__()
        # Append after all existing parameters so old Adam IDs retain their order.
        self.physical_encoder=copy.deepcopy(self.shape_encoder)
        self.proposal_stats={}

    def proposal(self,image,shifts=None,sample=True):
        shifts={} if shifts is None else shifts
        self.calls['proposal_images']+=len(image)
        v=self.shape_encoder(image).chunk(2,1)[0]
        x=self.gaussian(self.physical_encoder(image),sample,shifts.get('x'))
        m=self.gaussian(self.appearance_encoder(image),sample,shifts.get('m'))
        ell=self.gaussian(self.logit_encoder(v),sample,shifts.get('ell'))
        c=self.gaussian(self.code_encoder(ell[0]),sample,shifts.get('c'))
        fields=dict(zip(('x','m','ell','c'),(x,m,ell,c)))
        self.proposal_stats=dict(semantic_feature_rms=float(v.detach().square().mean().sqrt()),
                                 physical_x_rms=float(x[0].detach().square().mean().sqrt()))
        for name,values in fields.items():
            q=torch.quantile(values[3].detach().flatten(),torch.tensor([.05,.5,.95],device=image.device))
            for label,value in zip(('q05','q50','q95'),q):self.proposal_stats[name+'_logsd_'+label]=float(value)
        return {k:v[0] for k,v in fields.items()},sum(v[1] for v in fields.values())

    def bound(self,image,label=None):
        bound,terms,diag=super().bound(image,label)
        diag.update(self.proposal_stats)
        return bound,terms,diag


def semantic(name):
    return name.startswith(('shape_encoder.','logit_encoder.'))

"""No synthetic history branch; retain the shared frozen spatial coordinate."""
import torch
from unittest.mock import patch
from . import core,semantic_mass_target
from .runtime import attach
from ..siemens_ssl_reproduction_v1.methods import pixel_ce


def main(cfg):
    original_build=core.build
    def build(c):
        model=original_build(c)
        model.generator=None;model.code=None
        return model
    def setup(c):
        model=build(c)
        projection=torch.load(c['shared_projection'],map_location='cpu',weights_only=False)
        model.projection.load_state_dict(projection)
        model.parent.configure_stage_training();model.projection.requires_grad_(False).eval()
        assert model.generator is None and model.code is None
        groups=[dict(params=[p for p in model.parent.parameters() if p.requires_grad and (p.ndim>1)==matrix],lr=.001,fixed_lr=.001,name='native_'+str(matrix),weight_decay=.0005 if matrix else 0.) for matrix in (True,False)]
        opt=torch.optim.SGD(groups,lr=.001,momentum=.9,nesterov=True);attach(opt,c,c['arm'],c['phase'])
        return model.teacher(),model,model.teacher(),opt
    def losses(arm,model,data,step):
        assert model.generator is None and model.code is None
        x,y,_=data.batch('L',step);u,_,_=data.batch('U',step,14)
        ce=pixel_ce(model(x),y).mean()
        geo=core.geometry(model,torch.cat([x,u]),step,memory_count=0)
        return ce,.1*geo,dict(supervised=float(ce.detach()),unsupervised=float(.1*geo.detach()),accepted_class_pixels=[0]*4,U_pixels=0,geometry_rows=64,geometry_images=48,historical_raw_reads=0)
    with patch.object(core,'build',build),patch.object(core,'target_setup',setup),patch.object(core,'losses',losses):
        semantic_mass_target.main(cfg)

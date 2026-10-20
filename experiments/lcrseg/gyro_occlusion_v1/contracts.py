from pathlib import Path
import torch
from .augmentation import paired_view
from ..siemens_ssl_reproduction_v1.methods import strong_view as legacy
from ..gyro_fixed_projector_v1.contracts import main as fixed_checks
from ..main_style_ssl_v1.runtime import write

def main(cfg):
    fixed_checks(cfg)
    x=torch.linspace(-1,1,6*32*32).reshape(6,1,32,32)
    for step in [0,1,1999]:
        old,mask=paired_view(x,step,erase=True,return_masks=True)
        new,newmask=paired_view(x,step,erase=False,return_masks=True)
        assert torch.equal(old,legacy(x,step)) and torch.equal(mask,newmask)
        assert torch.equal(old[:,0][~mask],new[:,0][~mask])
    write(Path(cfg['root'])/'SYNTHETIC_CHECKS.json',dict(status='PASS',optimizer_calls=0,legacy_augmentation_exact=True,photometric_and_cutout_center_rng_exact=True,only_masked_pixels_may_differ=True,fixed_phi_input_gradient=True))

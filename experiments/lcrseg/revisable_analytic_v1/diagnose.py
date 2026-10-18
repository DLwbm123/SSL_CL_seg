"""One post-run source-L geometry check, with no fit or parameter update."""
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from ..space_lora_v1.core import source_model
from ..main_style_ssl_v1.data import Accessor
from ..main_style_ssl_v1.runtime import append, write


@torch.inference_mode()
def main(cfg):
    root=Path(cfg['output']); started=time.time(); torch.set_num_threads(4)
    model=source_model(cfg); item=Accessor(cfg['data'],'mnms','Siemens','train_labeled').read(0)
    x=item['image'][:1,None].cuda(); h=model.parent.features(x); patches=F.unfold(h,3).transpose(1,2)
    layer=model.parent.native.decoder.conv_logit.mu
    raw=patches@layer.weight.flatten(1).T
    if layer.bias is not None: raw+=layer.bias
    manual=F.interpolate(raw.transpose(1,2).reshape(1,4,h.shape[-2]-2,h.shape[-1]-2),size=x.shape[-2:],mode='bilinear',align_corners=True)
    native=model.parent.native_readout(h,x.shape[-2:]); error=float((native-manual).abs().max())
    assert torch.allclose(native,manual,atol=3e-5,rtol=3e-5)
    state=torch.load(root/'stage0_state.pt',map_location='cpu',weights_only=False)
    eig=torch.linalg.eigvalsh(state['gl'])
    report=dict(status='PASS',native_head_equals_unfold_patch_matrix_readout=True,maximum_logit_difference=error,
        source_L_class_mass=state['cl'][-1].tolist(),gram_min_eigenvalue=float(eig.min()),gram_max_eigenvalue=float(eig.max()),
        gram_ridge_condition=float((eig.max()+.001)/(eig.min()+.001)),ridge=.001,
        forward_slices=1,additional_solve_calls=0,additional_optimizer_updates=0,seconds=time.time()-started,
        interpretation='Spatial extraction/native readout check passed; this does not isolate why the RBF analytic source head fails. No tuning or rerun.')
    append(root/'exposure.jsonl',dict(kind='postrun_geometry',domain='Siemens',role='train_labeled',forward_slices=1))
    write(root/'GEOMETRY_DIAGNOSTIC.json',report)


"""One boundary pass over the original ten labeled Siemens patients; no U labels."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from .core import allocate, source_model
from ..lctx_weight_memory_v0_1.core import LAYERS
from ..main_style_ssl_v1.data import Accessor
from ..siemens_ssl_reproduction_v1.methods import pixel_ce
from ..five_frameworks_v1.checkpoint import atomic_save
from ..main_style_ssl_v1.runtime import write


def main(cfg):
    output=Path(cfg['statistics'])
    if output.exists(): raise FileExistsError(output)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    model=source_model(cfg)
    access=Accessor(cfg['data'],'mnms','Siemens','train_labeled')
    if len(access.patients)!=10: raise ValueError('wrong source labeled population')
    grams={}; fishers={}; inputs={}; counts={}; handles=[]
    for name in LAYERS:
        layer=model.parent.native.get_submodule(name[:-7]); d=layer.weight.flatten(1).shape[1]
        grams[name]=torch.zeros(d,d,device='cuda',dtype=torch.float64)
        fishers[name]=torch.zeros_like(grams[name]); counts[name]=0
        def forward(module,args,result,key=name):
            x=args[0].detach()
            def backward(gradient):
                patches=F.unfold(x,module.kernel_size,dilation=module.dilation,
                                 padding=module.padding,stride=module.stride)[0].double()
                delta=gradient[0].flatten(1).double()
                # Reassociate to avoid a spatial P x P matrix.
                g=delta@patches.T
                grams[key].add_(patches@patches.T); fishers[key].add_(g.T@g)
                counts[key]+=patches.shape[1]
            result.register_hook(backward)
        handles.append(layer.register_forward_hook(forward))
    slices=0
    try:
        for index in range(len(access.rows)):
            item=access.read(index)
            for z in range(len(item['image'])):
                x=item['image'][z:z+1,None].cuda().requires_grad_(True)
                y=item['label'][z:z+1].cuda()
                pixel_ce(model(x),y).mean().backward()
                slices+=1
            print(json.dumps(dict(source_cases=index+1,source_slices=slices,optimizer_updates=0)),flush=True)
    finally:
        for handle in handles: handle.remove()
    if slices!=230 or any(p.grad is not None for p in model.parameters()):
        raise AssertionError('source coverage/freeze violation')
    bases={}; report={}
    for name in LAYERS:
        bases[name],report[name]=allocate((grams[name]/counts[name]).cpu(),(fishers[name]/slices).cpu())
    atomic_save(dict(bases=bases,allocation=report,source_patients=10,source_slices=slices,
                     optimizer_updates=0,source_unlabeled_access=False),output)
    write(output.with_suffix('.json'),dict(status='COMPLETE',source_slices=slices,
          source_patients=10,optimizer_updates=0,allocation=report))

"""Window storage admission and common-random-number objective readout."""
import os,json
from pathlib import Path
import torch
from lcrseg.engine.checkpoint import capture_rng_state,restore_rng_state


def storage(settings,reserve=100_000_000):
    apparent=allocated=0
    for root in settings['space_roots']:
        if not Path(root).exists():continue
        for folder,dirs,files in os.walk(root):
            for name in files:
                p=Path(folder)/name
                if p.is_symlink():continue
                st=p.stat();apparent+=st.st_size;allocated+=st.st_blocks*512
    used=max(apparent,allocated)
    assert used+reserve<18_000_000_000,dict(apparent=apparent,allocated=allocated,reserve=reserve)
    return dict(apparent_bytes=apparent,allocated_bytes=allocated,conservative_bytes=used,hard_limit_bytes=20_000_000_000)


def objective(model,image,y,replay,noises=(1,25)):
    rng=capture_rng_state();vals=[]
    try:
        with torch.no_grad():
            for noise in noises:
                p=torch.load(Path(replay)/f'S{noise-1:02}.pt',map_location='cpu',weights_only=False)
                restore_rng_state(p['rng']);model.draw()
                try:
                    bound,_,_=model.bound(image,y);v=float(-bound.mean()/128**2+model.kl()/(2*128**2));assert torch.isfinite(torch.tensor(v));vals.append(v)
                finally:model.clear_draw()
        return vals
    finally:restore_rng_state(rng)


def joint_pass(values,baseline):return all(a<=b+1e-5 for a,b in zip(values,baseline)) and sum(values)/len(values)<sum(baseline)/len(baseline)-1e-5

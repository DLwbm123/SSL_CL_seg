"""Locate initialization semantic coverage without a patient or native-model forward."""
import json,time,subprocess
from pathlib import Path
import torch
from .module import JointGenerator,unit

def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def load(p):return torch.load(p,map_location='cpu',weights_only=False)

def metrics(code):
    z=unit(code).double();flat=z.flatten(1);flat=flat/flat.norm(dim=1,keepdim=True)
    sim=flat@flat.T;off=~torch.eye(len(z),dtype=torch.bool)
    return dict(mean_code_pair_cosine=float(sim[off].mean()),min_code_pair_cosine=float(sim[off].min()),mean_position_cross_code_variance=float(z.var(0,unbiased=False).mean()),mean_position_norm=float(z.norm(dim=1).mean()))

@torch.inference_mode()
def main(cfg):
    root=Path(cfg['diagnostic_root']);parent=Path(cfg['root']);calls=0
    def guard(new=False):
        elapsed=time.time()-cfg['diagnostic_started_epoch'];assert elapsed<540 and (not new or elapsed<480),'original clock'
        used=int(subprocess.check_output(['du','-s','-B1',str(root)],text=True).split()[0]);assert used+2_000_000<50_000_000 and used<20_000_000
        return dict(elapsed_seconds=elapsed,allocated_bytes=used,metadata_reserve_bytes=2_000_000)
    try:
        guard(new=True);torch.set_num_threads(4);torch.manual_seed(163);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        initial=load(parent/'initial_codes.pt')['code'];codec=load(parent/'codec/final.pt')['model']
        gen={k[len('generator.'):]:v for k,v in codec.items() if k.startswith('generator.')};model=JointGenerator(1).cuda().eval().requires_grad_(False);model.load_state_dict(gen)
        result={'initial_code':metrics(initial),'final_codes':{}}
        for stage in ['C0','C1']:
            start=load(parent/stage/'initial.pt')['model'];end=load(parent/stage/'final.pt')['model']
            assert torch.equal(start['code'],initial)
            assert all(torch.equal(v,start[k]) and torch.equal(v,end[k]) for k,v in codec.items() if k.startswith(('generator.','projection.')))
            final=unit(end['code']);result['final_codes'][stage]=dict(**metrics(end['code']),MSE_from_initial=float((final-unit(initial)).square().mean()))
        z=unit(initial).cuda();soft=torch.zeros(4,dtype=torch.float64);hard=torch.zeros(4,dtype=torch.float64);nonempty=torch.zeros(4,dtype=torch.long)
        for i in range(0,32,2):
            guard();write(root/'COUNTER.json',dict(executed_generator_samples=calls,pending_generator_samples=2,native_forwards=0,optimizer=0,functional=0))
            image,mask=model(z[i:i+2],(384,384));assert torch.isfinite(image).all() and torch.isfinite(mask).all();calls+=len(mask)
            soft+=mask.double().sum((0,2,3)).cpu();labels=mask.argmax(1)
            for c in range(4):hard[c]+=(labels==c).sum().cpu();nonempty[c]+=(labels==c).flatten(1).any(1).sum().cpu()
            write(root/'COUNTER.json',dict(executed_generator_samples=calls,pending_generator_samples=0,native_forwards=0,optimizer=0,functional=0))
        result['initial_mask']=dict(class_order=['background','LV','MYO','RV'],soft_class_mass_fraction=(soft/soft.sum()).tolist(),hard_class_pixel_fraction=(hard/hard.sum()).tolist(),codes_with_nonempty_hard_class=nonempty.tolist())
        old=json.loads(Path(cfg['final_mask_receipt']).read_text());result['final_mask_reused']=old['metrics'];result.update(status='COMPLETE',generator_samples=calls,native_forwards=0,optimizer_updates=0,functional_updates=0,resources=guard())
        assert calls==32;write(root/'RESULTS.json',result);write(root/'EXIT.json',dict(exit_code=0))
    except BaseException as e:
        import traceback
        write(root/'FAILURE.json',dict(error=str(e),traceback=traceback.format_exc(),generator_samples=calls));raise

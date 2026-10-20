"""One frozen source-marginal repair; reuses the original condensation engine."""
import json,time,subprocess
from pathlib import Path
import torch
from . import source,core,runtime
from .data import Domain
from .module import unit,retained_bytes

def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

def main(cfg):
    root=Path(cfg['root']);old_guard=runtime.guard
    def guard(c,planned=0,new_stage=False):
        elapsed=time.time()-cfg['started_epoch']
        if elapsed>=6300 or (new_stage and elapsed>=5400):raise TimeoutError('repair original 2h clock')
        used=int(subprocess.check_output(['du','-s','-B1',str(root)],text=True).split()[0])+2_000_000
        if used+planned>=800_000_000:raise OSError('repair 800MB protection plus 200MB reserve')
        return dict(elapsed_seconds=elapsed,allocated_bytes_including_metadata=used)
    runtime.guard=guard;source.guard=guard
    try:
        guard(cfg,new_stage=True);torch.set_num_threads(4);torch.manual_seed(163);torch.cuda.manual_seed_all(163)
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        data=Domain(cfg,'Siemens');hist=torch.zeros(4,dtype=torch.float64);slices=0
        for i,depth in enumerate(data._l.depths):
            for z in range(depth):
                y=data._l.read(i,z)['label'];valid=(y>=0)&(y<4);hist+=torch.bincount(y[valid].flatten(),minlength=4).double();slices+=1
        p=(hist/hist.sum()).cuda();assert torch.all(p>0)
        write(root/'SOURCE_MARGINAL.json',dict(class_order=['background','LV','MYO','RV'],fractions=p.cpu().tolist(),patients=10,volumes=20,slices=slices,U_labels_used=False))
        original=source.code_loss
        def repaired(c,model,data,step,stage,phase,paths=False):
            total,stats=original(c,model,data,step,stage,phase,paths)
            idx=core.code_indices(step);_,mask=model.generator(unit(model.code[idx]),(384,384))
            q=mask.double().mean((0,2,3)).clamp_min(1e-12);penalty=(p*(p.log()-q.log())).sum()
            stats['source_marginal_KL']=float(penalty.detach());return total+penalty,stats
        source.code_loss=repaired
        for phase,out in [('engineering',root/'qualification/C0'),('formal',root/'C0')]:
            guard(cfg,new_stage=True);write(root/'STATUS.json',dict(status='RUNNING',phase=phase))
            source.run(dict(cfg,stage='C0',phase=phase,output=str(out)))
        model=source.load_codec(cfg);saved=torch.load(root/'C0/final.pt',map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);model.eval().requires_grad_(False)
        soft=torch.zeros(4,dtype=torch.float64);nonempty=torch.zeros(4,dtype=torch.long)
        with torch.inference_mode():
            for i in range(0,32,2):
                guard(cfg);_,mask=model.generator(unit(model.code[i:i+2]),(384,384));assert torch.isfinite(mask).all()
                soft+=mask.double().sum((0,2,3)).cpu();labels=mask.argmax(1)
                for c in range(4):nonempty[c]+=(labels==c).flatten(1).any(1).sum().cpu()
        q=soft/soft.sum();target=p.cpu();coverage=bool(((q[1:]>=.25*target[1:])&(q[1:]<=4*target[1:])).all() and (nonempty[1:]>=8).all())
        memory_bytes=json.loads((root/'C0/MEMORY.json').read_text())['bytes'];counter=json.loads((root/'budget_counter.json').read_text());assert counter['pending']==0 and counter['total']<=544 and counter['by_phase']['C0:formal']==512
        result=dict(status='COMPLETE',semantic_coverage_gate='PASS' if coverage else 'SCIENTIFIC_STOP',source_fractions=target.tolist(),generated_fractions=q.tolist(),hard_nonempty_code_counts=nonempty.tolist(),persistent_bytes=memory_bytes,cost=counter,quality_generator_samples=32,quality_native_forwards=0,target_training_started=False,resources=guard(cfg))
        write(root/'RESULTS.json',result);write(root/'STATUS.json',dict(status='COMPLETE',semantic_coverage_gate=result['semantic_coverage_gate']));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException as e:
        import traceback
        write(root/'FAILURE.json',dict(error=str(e),traceback=traceback.format_exc()));write(root/'STATUS.json',dict(status='FAILED'));raise

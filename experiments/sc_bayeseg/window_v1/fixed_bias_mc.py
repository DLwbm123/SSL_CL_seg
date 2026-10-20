"""Fresh high-precision posterior readout at a previously frozen scalar bias."""
import json
import math
import os
import time
import traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.r_start import cpu,write,save,draw_g,local_sample,metrics
from sc_bayeseg.window_v1.same_family_oracle import posterior
from sc_bayeseg.window_v1.spatial_residual import SpatialResidualModel
from sc_bayeseg.window_v1.fullview import flip
from sc_bayeseg.window_v1.common import storage

SEEDS=(1420163,1430163,1440163,1450163)
BIASES={'ZERO':.5167211891076328,'R1_WARM':.4934826761721231}


def calibrate(p,b):
    value=p.clone();value[:,3]*=math.exp(b);return value/value.sum(1,keepdim=True)


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);started=time.time()
    costs=dict(generator_forward=0,generator_image_forward=0,theta=0,reverse=0,optimizer=0,parameter_applications=0)
    try:
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        write(root/'CONTEXT.json',s);storage(s,reserve=2_000_000_000)
        proof=Path(s['bias_source']);assert json.loads((proof/'AUDIT.json').read_text())['status']=='PASS'
        previous=json.loads((proof/'RESULTS.json').read_text())
        for arm,b in BIASES.items():
            row=next(r for r in previous['rows'] if r['candidate']==arm+'_PIXEL_CE' and r['mode']=='MIXTURE')
            interval=sorted(row['feasible_intervals'],key=lambda v:(-(v[1]-v[0]),v[0]))[0]
            assert abs(sum(interval)/2-b)<1e-14
        sources=[Path(s['checkpoint'])]+[Path(s['allocation_source'])/(a+'_T2L2')/'S0500.pt' for a in BIASES]
        meta={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in sources+[proof/'RESULTS.json']}
        source=torch.load(sources[0],map_location='cpu',weights_only=False);assert source['cursor']==8675
        model=SpatialResidualModel().cuda().requires_grad_(False);model.load_state_dict(source['model']);model.eval();frozen=cpu(model.state_dict())
        rows=[]
        for arm,file in zip(BIASES,sources[1:]):
            state=torch.load(file,map_location='cpu',weights_only=False);assert state['source_step']==1500
            q=posterior(tuple(v.cuda() for v in state['raw']));flows=[]
            with torch.no_grad():
                for seed in SEEDS:
                    assert time.time()-s['original_started_unix']<85500
                    tg=torch.Generator(device='cuda').manual_seed(seed);lg=torch.Generator(device='cuda').manual_seed(seed+1000000)
                    value=torch.zeros((8,4,128,128),device='cuda')
                    for _ in range(64):
                        draw_g(model,tg);costs['theta']+=1
                        for _ in range(8):
                            c,r=local_sample(q,lg);value+=model.spatial_logits(c,r).softmax(1)
                            costs['generator_forward']+=1;costs['generator_image_forward']+=8
                    flows.append((value/512).cpu())
                    write(root/'STATUS.json',dict(status='RUNNING',arm=arm,seed=seed,costs=costs))
            save(root/(arm+'_PRED.pt'),dict(flows=flows,seeds=list(SEEDS),fixed_bias=BIASES[arm]))
            # Predictions are sealed before training-label evaluation.
            sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text())
            ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
            lb=collate_labeled([ds[i] for i in sel['train_indices']])
            y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long();y=torch.cat([flip(y,k) for k in range(4)])
            for mode,b in (('UNCALIBRATED',0.),('FIXED_BIAS',BIASES[arm])):
                mixed=torch.stack(flows).mean(0)
                main=metrics(calibrate(mixed,b),y);perflow=[metrics(calibrate(p,b),y) for p in flows]
                rows.append(dict(arm=arm,mode=mode,bias=b,metrics=main,flow_metrics=perflow,
                  robust_qualified24=main['qualified_24'] and all(v['qualified_24'] for v in perflow)))
            model.clear_draw()
        assert costs==dict(generator_forward=4096,generator_image_forward=32768,theta=512,reverse=0,optimizer=0,parameter_applications=0)
        assert same(frozen,model.state_dict()) and all(v.grad is None for v in model.parameters())
        assert all((p.stat().st_size,p.stat().st_mtime_ns)==st for p,st in meta.items())
        write(root/'RESULTS.json',dict(status='COMPLETE',rows=rows,costs=costs,source_frozen=True,biases_frozen_before_new_draws=True,
          phase_elapsed_seconds=time.time()-started,original_elapsed_seconds=time.time()-s['original_started_unix'],space=storage(s)))
        write(root/'STATUS.json',dict(status='COMPLETE',costs=costs));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),costs=costs));write(root/'EXIT.json',dict(exit_code=1));raise

if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))

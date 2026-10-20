"""Four fixed readouts of existing oracle states; no optimization."""
import json
import os
import time
import traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset, collate_labeled
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.r_start import STREAMS, cpu, write, save, draw_g, local_sample, metrics
from sc_bayeseg.window_v1.same_family_oracle import posterior, ARMS
from sc_bayeseg.window_v1.spatial_residual import SpatialResidualModel
from sc_bayeseg.window_v1.fullview import flip
from sc_bayeseg.window_v1.common import storage


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);started=time.time()
    costs=dict(optimizer=0,reverse=0,generator_forward=0,generator_image_forward=0,theta=0,reused_predictions=0)
    try:
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        write(root/'CONTEXT.json',s);storage(s,reserve=2_000_000_000)
        old=Path(s['oracle_source']);assert json.loads((old/'AUDIT.json').read_text())['status']=='PASS'
        files=[Path(s['checkpoint'])]+[old/a/'S1000.pt' for a in ARMS]
        metadata={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in files}
        payload=torch.load(files[0],map_location='cpu',weights_only=False);assert payload['cursor']==8675
        model=SpatialResidualModel().cuda().requires_grad_(False);model.load_state_dict(payload['model']);model.eval()
        source=cpu(model.state_dict());rows=[]
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text())
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        def means():
            for layer in model.anatomy:
                layer.sampled=tuple(getattr(layer,'mean_'+p).detach() for p in ('w','b'))
        def forward(c,r):
            costs['generator_forward']+=1;costs['generator_image_forward']+=8
            return model.spatial_logits(c,r).softmax(1)
        with torch.no_grad():
            for arm in ARMS:
                state=torch.load(old/arm/'S1000.pt',map_location='cpu',weights_only=False)
                q=posterior(tuple(v.cuda() for v in state['raw']))
                for mode in ('MM','SM','MS','SS'):
                    flows=[]
                    if mode=='SS':
                        flows=torch.load(old/arm/'PRED_1000.pt',map_location='cpu',weights_only=False)['generator_flows']
                        costs['reused_predictions']+=1
                    elif mode=='MM':
                        means();flows=[forward(q[0],q[2]).cpu()]
                    else:
                        for seed in STREAMS:
                            tg=torch.Generator(device='cuda').manual_seed(seed)
                            lg=torch.Generator(device='cuda').manual_seed(seed+1000000)
                            prob=torch.zeros((8,4,128,128),device='cuda')
                            count=8 if mode=='SM' else 64
                            for _ in range(count):
                                if mode=='SM':
                                    draw_g(model,tg);costs['theta']+=1;c,r=q[0],q[2]
                                else:
                                    means();c,r=local_sample(q,lg)
                                prob+=forward(c,r)
                            flows.append((prob/count).cpu())
                    save(root/f'{arm}_{mode}.pt',dict(flows=flows))
                    # Label access only after probabilities are sealed.
                    lb=collate_labeled([ds[i] for i in sel['train_indices']])
                    y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
                    y=torch.cat([flip(y,k) for k in range(4)])
                    main=metrics(torch.stack(flows).mean(0),y)
                    dice=[v for row in main['per_view_image'] for v in row['dice'][1:]]
                    rows.append(dict(arm=arm,mode=mode,metrics=main,flow_metrics=[metrics(p,y) for p in flows],
                        passed24=sum(v>=.8 for v in dice),minimum_foreground_dice=min(dice),
                        interpretation='Label-conditioned oracle diagnostic; mean readouts do not replace SS posterior prediction'))
                    write(root/'STATUS.json',dict(status='RUNNING',arm=arm,mode=mode,costs=costs))
                model.clear_draw()
        assert costs==dict(optimizer=0,reverse=0,generator_forward=578,generator_image_forward=4624,theta=64,reused_predictions=2)
        assert same(source,model.state_dict()) and all(v.grad is None for v in model.parameters())
        assert all((p.stat().st_size,p.stat().st_mtime_ns)==st for p,st in metadata.items())
        write(root/'RESULTS.json',dict(status='COMPLETE',rows=rows,costs=costs,source_frozen=True,
          original_elapsed_seconds=time.time()-s['original_started_unix'],phase_elapsed_seconds=time.time()-started,space=storage(s)))
        write(root/'STATUS.json',dict(status='COMPLETE',costs=costs));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),costs=costs))
        write(root/'EXIT.json',dict(exit_code=1));raise

if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))

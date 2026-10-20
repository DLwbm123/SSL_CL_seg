"""Crossed parameter/local-noise gradients at fixed oracle states."""
import json
import os
import time
import traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset, collate_labeled
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.r_start import cpu, write, save, append, draw_g, local_sample
from sc_bayeseg.window_v1.same_family_oracle import posterior, ARMS
from sc_bayeseg.window_v1.spatial_residual import SpatialResidualModel
from sc_bayeseg.window_v1.fullview import flip
from sc_bayeseg.window_v1.common import storage


def decompose(values):
    h=torch.tensor(values,dtype=torch.float64);t,l=h.shape
    mean=h.mean();row=h.mean(1);col=h.mean(0)
    residual=h-row[:,None]-col[None,:]+mean
    ss=dict(theta=float(l*(row-mean).square().sum()),local=float(t*(col-mean).square().sum()),interaction=float(residual.square().sum()))
    total=float((h-mean).square().sum());assert abs(sum(ss.values())-total)<1e-18
    ms=dict(theta=ss['theta']/(t-1),local=ss['local']/(l-1),interaction=ss['interaction']/((t-1)*(l-1)))
    v=dict(theta=(ms['theta']-ms['interaction'])/l,local=(ms['local']-ms['interaction'])/t,interaction=ms['interaction'])
    positive={k:max(0.,x) for k,x in v.items()}
    proxy={f'{T}x{L}':positive['theta']/T+(positive['local']+positive['interaction'])/(T*L) for T,L in ((2,2),(4,1),(1,4))}
    ratio=proxy['4x1']/proxy['2x2'] if proxy['2x2']>0 else None
    return dict(mean=float(mean),minimum=float(h.min()),maximum=float(h.max()),negative_cells=int((h<0).sum()),
      row_means=row.tolist(),column_means=col.tolist(),sum_squares=ss,total_sum_squares=total,
      variance_components_raw=v,nested_variance_proxy=proxy,ratio_4x1_to_2x2=ratio)


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);started=time.time()
    costs=dict(reverse=0,optimizer=0,parameter_applications=0,generator_forward=0,generator_image_forward=0,theta=0)
    try:
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        write(root/'CONTEXT.json',s);storage(s,reserve=2_000_000_000)
        old=Path(s['oracle_source']);groot=Path(s['gradient_source'])
        assert json.loads((groot/'AUDIT.json').read_text())['status']=='PASS'
        files=[Path(s['checkpoint']),groot/'GRADIENTS.pt']+[old/a/'S1000.pt' for a in ARMS]
        metadata={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in files}
        gradient=torch.load(files[1],map_location='cpu',weights_only=False)
        payload=torch.load(files[0],map_location='cpu',weights_only=False);assert payload['cursor']==8675
        model=SpatialResidualModel().cuda().requires_grad_(False);model.load_state_dict(payload['model']);model.eval();source=cpu(model.state_dict())
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text())
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in sel['train_indices']])
        y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long();y=torch.cat([flip(y,k) for k in range(4)]).cuda()
        results={};private={}
        for arm in ARMS:
            state=torch.load(old/arm/'S1000.pt',map_location='cpu',weights_only=False)
            original=state['raw'];raw=tuple(v.cuda().detach().requires_grad_(True) for v in original)
            reference=torch.stack([v['Q_eval'] for v in gradient[arm]['flows']]).mean(0);gk=gradient[arm]['K']
            tg=torch.Generator(device='cuda').manual_seed(1220163);draws=[];grid=[]
            for theta in range(8):
                assert time.time()-s['original_started_unix']<85500
                draw_g(model,tg);costs['theta']+=1
                lg=torch.Generator(device='cuda').manual_seed(1320163);gv=[];hv=[]
                for local in range(8):
                    c,r=local_sample(posterior(raw),lg);loss=F.cross_entropy(model.spatial_logits(c,r),y,reduction='none').mean()
                    grad=torch.autograd.grad(loss,raw);costs['reverse']+=1
                    g=torch.cat([v.detach().flatten().cpu() for v in grad]);assert torch.isfinite(g).all()
                    gv.append(g);hv.append(float(-torch.dot(reference,g.double()+gk)))
                    costs['generator_forward']+=1;costs['generator_image_forward']+=8
                    append(root/'reverse.jsonl',dict(index=costs['reverse'],arm=arm,theta=theta,local=local))
                draws.append(torch.stack(gv));grid.append(hv)
                write(root/'STATUS.json',dict(status='RUNNING',arm=arm,theta=theta,costs=costs))
            results[arm]=dict(projection_grid=grid,statistics=decompose(grid),reference_risk_norm=float(reference.norm()))
            private[arm]=dict(data_gradients=torch.stack(draws),reference=reference,KL=gk)
            assert same(cpu(raw),original);model.clear_draw()
        assert costs==dict(reverse=128,optimizer=0,parameter_applications=0,generator_forward=128,generator_image_forward=1024,theta=16)
        assert same(source,model.state_dict()) and all(v.grad is None for v in model.parameters())
        assert all((p.stat().st_size,p.stat().st_mtime_ns)==st for p,st in metadata.items())
        save(root/'SAMPLES.pt',private)
        write(root/'RESULTS.json',dict(status='COMPLETE',arms=results,costs=costs,source_frozen=True,raw_unchanged=True,
          allocation_control_gate=all(v['statistics']['ratio_4x1_to_2x2'] is not None and v['statistics']['ratio_4x1_to_2x2']<=.8 for v in results.values()),
          phase_elapsed_seconds=time.time()-started,original_elapsed_seconds=time.time()-s['original_started_unix'],space=storage(s)))
        write(root/'STATUS.json',dict(status='COMPLETE',costs=costs));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),costs=costs));write(root/'EXIT.json',dict(exit_code=1));raise

if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))

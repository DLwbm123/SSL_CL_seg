import os,json,csv,time,copy
from pathlib import Path
import torch,h5py,numpy as np
from torch.nn import functional as F
from lcrseg.models.unet import UNet2D
from experiments.sc_bayeseg.hbsi_v1.config import Config
from experiments.sc_bayeseg.hbsi_v1.data_adapter import CurrentTrainingData,confined,restore_probability
from experiments.sc_bayeseg.hbsi_v1.real_diagnostic import write
from experiments.sc_bayeseg.hbsi_v1.budgets import ResourceLedger
from sc_bayeseg.sslcl_v1.prostate_metrics import binary_phase_metrics


def loss(model,x,y):
    z=model(x).logits;p=z.softmax(1)[:,1];t=(y==1).to(p.dtype)
    count=t.sum((1,2));dice=(2*(p*t).sum((1,2))+1e-6)/(p.sum((1,2))+count+1e-6)
    return -z.log_softmax(1).gather(1,y[:,None]).mean()+((1-dice)*(count>0)).mean()


def main():
    root=Path(os.environ['RUN_ROOT']);parent=Path(os.environ['PARENT_RUN']);p=json.loads((parent/'PRIVATE_PROTOCOL.json').read_text());cfg=Config(**p['config']);dr=Path(p['data_root'])
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(cfg.seed);model=UNet2D(1,2,base_channels=16);model.projection_head=torch.nn.Identity();model=model.double().cuda();optimizer=torch.optim.Adam(model.parameters(),lr=cfg.outer_lr)
    prepared=CurrentTrainingData.inspect_manifest(dr,'prostate','RUNMC',p['data_version'],0,roles=('L',));assert len(prepared['selected'])==4
    data=CurrentTrainingData(dr,'prostate','RUNMC','unet',cfg,data_version=p['data_version'],prepared=prepared);assert not data.records['U']
    ledger=ResourceLedger(root/'results',max_seconds=7200,max_updates=1604,max_bytes=512*1024**2)
    def save(step):
        tmp=root/'checkpoint.tmp';torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,seed=cfg.seed,data_seed=p['data_seed']),tmp);os.replace(tmp,root/'PRIVATE_CHECKPOINT.pt')
        assert sum(v.stat().st_size for v in root.rglob('*') if v.is_file())<450*1024**2, 'storage reserve reached'
    with ledger.attempt('preflight02',seconds=120,updates=0):
        initial=copy.deepcopy(model.state_dict());x=torch.randn(1,1,128,128,device='cuda',dtype=torch.float64);y=torch.zeros(1,128,128,device='cuda',dtype=torch.long);y[:,40:80,40:80]=1
        def step():
            optimizer.zero_grad();v=loss(model,x,y);v.backward();assert torch.isfinite(v) and all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters());optimizer.step()
        step();checkpoint=copy.deepcopy(dict(model=model.state_dict(),optimizer=optimizer.state_dict()));step();expected=copy.deepcopy(model.state_dict());model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer']);step();assert all(torch.equal(expected[k],v) for k,v in model.state_dict().items())
        model.load_state_dict(initial);optimizer=torch.optim.Adam(model.parameters(),lr=cfg.outer_lr)
        b=data.batch('L',1,torch.Generator().manual_seed(p['data_seed']));torch.cuda.synchronize();t=time.monotonic();v=loss(model,b['x'].cuda(),b['observations']['y_obs'].long().argmax(1).cuda());v.backward();torch.cuda.synchronize();elapsed=time.monotonic()-t;optimizer.zero_grad()
        assert all(v.grad is None for v in model.parameters());assert elapsed*1600*2<6500
        write(root/'PREFLIGHT.json',dict(resume_exact=True,fixture_updates=3,real_updates=0,step_seconds=elapsed,estimated_train_seconds=elapsed*1600,parameters=sum(v.numel() for v in model.parameters())))
    with ledger.attempt('train',seconds=6500,updates=1600):
        model.train();save(0)
        for i in range(1600):
            ledger.check();b=data.batch('L',1,torch.Generator().manual_seed(p['data_seed']+2*i));x=b['x'].cuda();y=b['observations']['y_obs'].long().argmax(1).cuda()
            optimizer.zero_grad();value=loss(model,x,y);value.backward();assert torch.isfinite(value) and all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters());optimizer.step()
            if (i+1)%100==0:
                save(i+1);tmp=root/'PROGRESS.tmp';tmp.write_text(json.dumps(dict(step=i+1,loss=float(value.detach()),updated=time.time())));os.replace(tmp,root/'PROGRESS.json')
        save(1600)
    records=[r for r in csv.DictReader((dr/'manifests/training/lcrseg_v1_seed0.csv').open()) if r['dataset']=='prostate' and r['site_or_vendor']=='RUNMC' and r['primary_20pct_split']=='val'];assert len(records)==4
    rows=[];model.eval()
    with ledger.attempt('evaluate',seconds=500,updates=0):
        for i,r in enumerate(records):
            with h5py.File(confined(dr/'h5/v1',r['image_h5_relpath']),'r') as f:x=torch.from_numpy(f['image'][...].copy())[:,None].double();spacing=json.loads(f.attrs['processed_spacing'])
            with torch.no_grad():q=torch.cat([model(F.interpolate(v[None],(128,128),mode='bilinear',align_corners=False).cuda()).logits.softmax(1).cpu() for v in x])
            q=restore_probability(q,x.shape[-2:])
            with h5py.File(confined(dr/'h5/v1',r['label_h5_relpath']),'r') as f:y=f['label'][...]
            rows.append(dict(case_index=i,metrics=binary_phase_metrics(q,y,spacing)))
        mm=[r['metrics'] for r in rows];cells=[v['classes'][0] for v in mm]
        summary=dict(Dice=float(np.mean([c['Dice'] for c in cells])),NLL=float(np.mean([v['NLL'] for v in mm])),Brier=float(np.mean([v['Brier'] for v in mm])),HD95_penalized_mm=float(np.mean([c['HD95_penalized_mm'] for c in cells])),empty_predictions=sum(c['empty_prediction_failure'] for c in cells))
        write(root/'SUMMARY.json',dict(summary=summary,rows=rows,real_updates=1600,fixture_updates=3,patients=4,deterministic=True,U_used=False,test_access=0))
        print(json.dumps(summary),flush=True)

if __name__=='__main__':main()

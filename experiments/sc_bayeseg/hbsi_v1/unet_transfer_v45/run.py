"""Frozen standard U-Net comparator and current-method target adaptation."""
import copy,csv,json,os,time
from dataclasses import replace
from pathlib import Path
import h5py,numpy as np,torch
from torch.nn import functional as F
from lcrseg.models.unet import UNet2D
from ..config import Config
from ..stream_state import StreamState,digest
from ..paired_semantic_v19.run import data,evaluate
from ..data_adapter import confined,restore_probability
from ..budgets import ResourceLedger,Counters
from ..training import run_stage
from ..local_inference import Refiner
from ..qualification import fingerprint
from ..real_diagnostic import write
from ..semantic_alignment_v23.method import Objective
from ...sslcl_v1.prostate_metrics import binary_phase_metrics


def risk(model,x,y):
    # Same standard supervised objective as v17, including its empty-GT convention.
    z=model(x).logits;p=z.softmax(1)[:,1];t=(y==1).to(p.dtype);count=t.sum((1,2))
    dice=(2*(p*t).sum((1,2))+1e-6)/(p.sum((1,2))+count+1e-6)
    return -z.log_softmax(1).gather(1,y[:,None]).mean()+((1-dice)*(count>0)).mean()


def unet():
    torch.manual_seed(20261023);m=UNet2D(1,2,base_channels=16);m.projection_head=torch.nn.Identity()
    return m.double().cuda()


def batch(d,step,seed):
    b=d.batch('L',1,torch.Generator().manual_seed(seed+2*step))
    return b['x'].cuda(),b['observations']['y_obs'].long().argmax(1).cuda()


def train_unet(m,opt,d,start,end,seed,name,ledger,seconds):
    m.train()
    with ledger.attempt(name,seconds=seconds,updates=end-start) as out:
        def save(step):
            tmp=out/'checkpoint.tmp';torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),step=step),tmp);tmp.replace(out/'PRIVATE_CHECKPOINT.pt')
        save(start);clock=time.monotonic()
        with (out/'PRIVATE_TRAIN_LOG.jsonl').open('x') as log:
            for step in range(start,end):
                ledger.check();x,y=batch(d,step,seed);opt.zero_grad(set_to_none=True);loss=risk(m,x,y);loss.backward()
                assert torch.isfinite(loss) and all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters())
                opt.step();log.write(json.dumps(dict(step=step+1,loss=float(loss)))+'\n');log.flush()
                if (step+1)%100==0:save(step+1)
        assert end%100==0
        write(out/'RECEIPT.json',dict(status='COMPLETE',steps=end-start,start=start,end=end,seconds=time.monotonic()-clock,formal_reverse_calls=end-start))
    return out/'PRIVATE_CHECKPOINT.pt'


def evaluate_unet(p,m,domain,out):
    out.mkdir(exist_ok=False);m.eval();summaries=[]
    records=[r for r in csv.DictReader((Path(p['data_root'])/'manifests/training/lcrseg_v1_seed0.csv').open()) if r['dataset']=='prostate' and r['site_or_vendor']==domain and r['primary_20pct_split']=='val']
    assert len(records)==4
    for i,r in enumerate(records):
        with h5py.File(confined(Path(p['data_root'])/'h5/v1',r['image_h5_relpath']),'r') as f:
            x=torch.from_numpy(f['image'][...].copy())[:,None].double();spacing=json.loads(f.attrs['processed_spacing'])
        with torch.no_grad():q=torch.cat([m(F.interpolate(v[None],(128,128),mode='bilinear',align_corners=False).cuda()).logits.softmax(1).cpu() for v in x])
        torch.save(q,out/f'PRIVATE_PREDICTION_{i}.pt')
        with h5py.File(confined(Path(p['data_root'])/'h5/v1',r['label_h5_relpath']),'r') as f:truth=f['label'][...]
        metrics=binary_phase_metrics(restore_probability(q,x.shape[-2:]),truth,spacing)
        write(out/f'PRIVATE_METRICS_{i}.json',dict(patient=r['patient_id'],metrics=[metrics]));summaries.append(metrics)
    cs=[m['classes'][0] for m in summaries]
    repeat=dict(Dice=float(np.mean([v['Dice'] for v in cs])),NLL=float(np.mean([v['NLL'] for v in summaries])),Brier=float(np.mean([v['Brier'] for v in summaries])),HD95_penalized_mm=float(np.mean([v['HD95_penalized_mm'] for v in cs])),empty_predictions=sum(v['empty_prediction_failure'] for v in cs))
    result=dict(domain=domain,patients=4,repeats=[repeat],predictor='deterministic_standard_UNet',complete=True)
    write(out/'SUMMARY.json',result);return result


def main():
    root=Path(os.environ['RUN_ROOT']);p=json.loads((root/'PRIVATE_PROTOCOL.json').read_text());assert p['purpose']=='UNET_TRANSFER_V45'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ledger=ResourceLedger(root/'results',max_seconds=7200,max_updates=3600,max_bytes=3_000_000_000);counter=Counters();cfg=replace(Config(**p['config']),seed=20261023)
    def status(stage,**kw):write(root/'STATUS.tmp',dict(stage=stage,time=time.time(),**kw));(root/'STATUS.tmp').replace(root/'STATUS.json')
    def load_method():
        s=StreamState.load(Path(p['v23_root'])/'results/L_attempt001/PRIVATE_SEALED_STATE.pt')
        assert s.active is None and s.receipts['RUNMC']['steps']==1600 and s.config.seed==20261023
        d=data(p,s.config,'BMC','L');s.begin_task('BMC',d.manifest_hash,d.evidence);s.to_device('cuda')
        s.phi.s_head.requires_grad_(False);s.phi.b_head.requires_grad_(False)
        return s,d
    def objective(s,ref):return Objective(s,ref,False)
    try:
        assert json.loads((Path(p['v23_root'])/'EXIT.json').read_text())['exit_code']==0
        source=data(p,cfg,'RUNMC','L');target=data(p,cfg,'BMC','L')
        assert source.counts==dict(L=80,U=0) and target.counts['U']==0
        assert cfg.height==cfg.width==128 and cfg.outer_lr==3e-4 and cfg.dtype=='float64'
        status('PREFLIGHT')
        with ledger.attempt('preflight',seconds=300,updates=0):
            m=unet();initial=digest(m.state_dict());checks=[]
            for d,seed in ((source,163000),(target,263000)):
                x,y=batch(d,0,seed);v=risk(m,x,y);v.backward();assert torch.isfinite(v) and all(q.grad is None or torch.isfinite(q.grad).all() for q in m.parameters());checks.append(float(v));m.zero_grad(set_to_none=True)
                assert digest(m.state_dict())==initial
            z=torch.tensor([[[[.4,.7]],[[.6,-.1]]]],dtype=torch.float64);y=torch.tensor([[[0,1]]])
            torch.testing.assert_close(-z.log_softmax(1).gather(1,y[:,None]).mean(),F.cross_entropy(z,y))
            s,d=load_method();before=digest(s.payload());rng=s.generator.get_state();L=d.batch('L',1,torch.Generator().manual_seed(263000));L['x']=L['x'].cuda();L['observations']={k:v.cuda() for k,v in L['observations'].items()}
            v,parts=objective(s,Refiner(s.model,s.phi,counter)).estimate(L,None,d.counts,s.generator);v.backward()
            assert torch.isfinite(v) and all(q.grad is None or torch.isfinite(q.grad).all() for g in s.optimizer.param_groups for q in g['params'])
            assert all(any(q.grad is not None and q.grad.norm()>0 for q in g['params']) for g in s.optimizer.param_groups)
            assert all(q.grad is None for q in s.memories['RUNMC'].parameters())
            s.optimizer.zero_grad(set_to_none=True);s.generator.set_state(rng);assert digest(s.payload())==before
            # Exact same step-indexed L batch for the method and U-Net target paths.
            ux,uy=batch(target,0,263000);assert torch.equal(ux,L['x']) and torch.equal(uy,L['observations']['y_obs'].long().argmax(1))
            write(root/'PREFLIGHT.json',dict(status='PASS',model_reverse_calls=3,optimizer_updates=0,unet_losses=checks,method_loss=float(v),method_entry_restored=True,paired_target_batch=True,CE_equivalence=True,unet_parameters=sum(q.numel() for q in m.parameters()),method_q_parameters=sum(q.numel() for q in s.q.parameters()),method_phi_parameters=sum(q.numel() for q in s.phi.parameters()),old_memory_no_gradient=True))
            del m,s,d,L,v
        cells={}
        def eval_u(m,arm):
            for domain in ('RUNMC','BMC'):
                status('EVALUATE',arm=arm,domain=domain)
                with ledger.attempt('eval_'+arm+'_'+domain,seconds=450,updates=0) as out:cells[arm+'_'+domain]=evaluate_unet(p,m,domain,out/'validation')
        m=unet();opt=torch.optim.Adam(m.parameters(),lr=cfg.outer_lr)
        status('TRAIN',arm='U1600');cp=train_unet(m,opt,source,0,1600,163000,'U1600',ledger,1800);eval_u(m,'U1600')
        status('TRAIN',arm='U2000');train_unet(m,opt,source,1600,2000,163000,'U2000',ledger,600);eval_u(m,'U2000')
        payload=torch.load(cp,map_location='cpu',weights_only=True);m.load_state_dict(payload['model']);opt=torch.optim.Adam(m.parameters(),lr=cfg.outer_lr)
        assert not opt.state;status('TRAIN',arm='U_B800');train_unet(m,opt,target,0,800,263000,'U_B800',ledger,1200);eval_u(m,'U_B800');del m,opt,payload,source,target
        s,d=load_method();old_memory=digest(s.memories['RUNMC'].state_dict());unused={k:digest(getattr(s.phi,k).state_dict()) for k in ('s_head','b_head')}
        write(root/'METHOD_ENTRY.json',dict(source='v23_L1600',source_steps=1600,target_steps=800,fresh_Adam=not bool(s.optimizer.state),source_posterior_prior=True,counts=d.counts,entry_payload=digest(s.payload())))
        status('TRAIN',arm='M_B800');run_stage(s,d,ledger,counter,'M_B800',800,2400,fingerprint(p),data_seed=263000,checkpoint_stride=400,objective_factory=objective)
        assert digest(s.memories['RUNMC'].state_dict())==old_memory and all(digest(getattr(s.phi,k).state_dict())==v for k,v in unused.items())
        for domain in ('RUNMC','BMC'):
            status('EVALUATE',arm='M_B800',domain=domain)
            with ledger.attempt('eval_M_B800_'+domain,seconds=450,updates=0) as out:cells['M_B800_'+domain]=evaluate(p,s,domain,domain,out/'validation')
        write(root/'RESULTS.json',dict(cells=cells,new_formal_updates=3600,unet_updates=2800,method_updates=800,preflight_reverse_calls=3,preflight_updates=0,method_counter=counter.export(),old_memory_and_Sb_unchanged=True,reused=['v23_L1600','v25_A2000'],unet_training_seed=20261023))
        status('COMPLETE')
    except BaseException as exc:status('FAILED',error_type=type(exc).__name__,error=str(exc));raise

if __name__=='__main__':main()

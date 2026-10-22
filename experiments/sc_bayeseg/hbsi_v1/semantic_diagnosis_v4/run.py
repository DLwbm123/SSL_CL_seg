import json, os, time
from pathlib import Path
import torch
from experiments.sc_bayeseg.hbsi_v1.budgets import ResourceLedger, Counters
from experiments.sc_bayeseg.hbsi_v1.stream_state import StreamState
from experiments.sc_bayeseg.hbsi_v1.data_adapter import CurrentTrainingData, leaf_observations
from experiments.sc_bayeseg.hbsi_v1.local_inference import Refiner
from experiments.sc_bayeseg.hbsi_v1.real_diagnostic import prepare_data, write
from torch.nn import functional as F

def main():
    root=Path(os.environ['RUN_ROOT']);old=Path(os.environ['PARENT_RUN'])
    p=json.loads((old/'PRIVATE_PROTOCOL.json').read_text())
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ledger=ResourceLedger(root/'results',max_seconds=300,max_updates=0,max_bytes=64*1024**2)
    checkpoints={'warm800':old/'results/warm_attempt001/ACTIVE_STEP_000800.pt',
                 'warm1200':old/'results/warm_attempt001/PRIVATE_SEALED_STATE.pt',
                 'control1200':old/'results/control_attempt001/PRIVATE_SEALED_STATE.pt'}
    rows=[]
    with ledger.attempt('diagnosis',seconds=300,updates=0) as out:
        for arm,path in checkpoints.items():
            state=StreamState.load(path).to_device('cuda');cfg=state.config
            p['arms']=['L'];data=prepare_data(p,cfg,'L');theta=state.q.mean.detach()
            memory=state.current if state.active is not None else next(iter(state.memories.values()))
            a=memory.mean(theta).detach()
            ref=Refiner(state.model,state.phi)
            prior=[]
            with torch.no_grad():
                g=torch.Generator().manual_seed(71001)
                for _ in range(16):
                    z={'c':torch.randn(1,cfg.c_dim,generator=g,dtype=torch.float64).cuda(),
                       'R':torch.randn(1,cfg.r_channels,cfg.r_height,cfg.r_width,generator=g,dtype=torch.float64).cuda()}
                    prob=state.model.class_probabilities(z,theta)[:,1]
                    prior.append(dict(probability=float(prob.mean()),foreground=float((prob>.5).double().mean())))
            for K,labelled in [(0,False),(3,False)]:
                cases={}
                for i,record in enumerate(data.records['L']):
                    ledger.check()
                    x=data.read(record,'image')[None,None].double()
                    y=data.read(record,'label')[None,None].double()
                    x=F.interpolate(x,(128,128),mode='bilinear',align_corners=False).cuda()
                    y=F.interpolate(y,(128,128),mode='nearest')[:,0].long().cuda()
                    obs=leaf_observations(y,2) if labelled else None
                    ctx,offsets=ref.refine(x,theta,a,obs,K=K,training=False,
                        generator=torch.Generator().manual_seed(81001+i))
                    probs=[];gen=torch.Generator().manual_seed(91001+i)
                    with torch.no_grad():
                        for _ in range(8):
                            z,_=state.phi.rsample_and_log_prob(ctx,state.phi.noise(1,x,gen),offsets,state.model,theta)
                            probs.append(state.model.class_probabilities(z,theta)[:,1])
                    q=torch.stack(probs).mean(0);pred=q>.5;truth=y==1
                    totals=cases.setdefault(str(record['image']),[0.,0.,0.,0.,0.,0.])
                    values=[float((pred&truth).sum()),float(pred.sum()),float(truth.sum()),
                            float(q[truth].sum()),float(q[~truth].sum()),float((~truth).sum())]
                    for j,v in enumerate(values):totals[j]+=v
                t=torch.tensor(list(cases.values()),dtype=torch.float64)
                row=dict(arm=arm,K=K,label_conditioned=labelled,patients=len(cases),slices=len(data.records['L']),
                    macro_Dice=float((2*t[:,0]/(t[:,1]+t[:,2])).mean()),
                    predicted_fg_fraction=float(t[:,1].sum()/(t[:,2]+t[:,5]).sum()),
                    GT_fg_fraction=float(t[:,2].sum()/(t[:,2]+t[:,5]).sum()),
                    probability_on_foreground=float(t[:,3].sum()/t[:,2].sum()),
                    probability_on_background=float(t[:,4].sum()/t[:,5].sum()),
                    generator_prior=prior,interpretation='training proposal diagnostic, not formal posterior prediction')
                rows.append(row);write(out/f'{arm}_K{K}_labels{labelled}.json',row)
                print(json.dumps({k:v for k,v in row.items() if k!='generator_prior'}),flush=True)
        write(root/'SUMMARY.json',dict(rows=rows,optimizer_updates=0,MCMC_calls=0,validation_access=0,test_access=0))

if __name__=='__main__':main()

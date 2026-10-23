"""Training-only trajectory audit of the unchanged local refinement rule."""
import json,os
from pathlib import Path
import torch
from experiments.sc_bayeseg.hbsi_v1.budgets import ResourceLedger
from experiments.sc_bayeseg.hbsi_v1.stream_state import StreamState
from experiments.sc_bayeseg.hbsi_v1.real_diagnostic import prepare_data,write
from experiments.sc_bayeseg.hbsi_v1.local_inference import Refiner


def main():
    root=Path(os.environ['RUN_ROOT']);old=Path(os.environ['PARENT_RUN'])
    p=json.loads((old/'PRIVATE_PROTOCOL.json').read_text())
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ledger=ResourceLedger(root/'results',max_seconds=300,max_updates=0,max_bytes=64*1024**2)
    rows=[]
    with ledger.attempt('trajectories',seconds=300,updates=0):
        for arm in ('pure','mixed'):
            s=StreamState.load(old/f'results/{arm}_attempt001/PRIVATE_SEALED_STATE.pt').to_device('cuda')
            data=prepare_data(p,s.config,arm);ref=Refiner(s.model,s.phi)
            for i in range(16):
                ledger.check();b=data.batch('L',1,torch.Generator().manual_seed(560001+i));x=b['x'].cuda()
                outer=torch.Generator().manual_seed(660001+i)
                theta=s.q.rsample(outer).detach();a=next(iter(s.memories.values())).rsample(theta,outer).detach()
                context=s.phi.context(x,theta,a).detach();offsets=s.phi.initial_offsets(1,x)
                fit=torch.Generator().manual_seed(760001+i);estimate=torch.Generator().manual_seed(860001+i)
                noises=[s.phi.noise(1,x,fit) for _ in range(3)];held=[s.phi.noise(1,x,estimate) for _ in range(4)]
                truth=b['observations']['y_obs'].cuda()[:,1].bool();trajectory=[];component_gradients={}
                for k in range(4):
                    # The same four held noises evaluate all steps; labels never enter refinement.
                    probs=[];values=[]
                    with torch.no_grad():
                        for noise in held:
                            elbo,z,parts,logq=ref.draw(x,theta,a,context,offsets,noise)
                            probs.append(s.model.class_probabilities(z,theta))
                            values.append({**{key:float(v.mean()) for key,v in parts.items()},'negative_logq':float(-logq.mean()),'elbo':float(elbo.mean())})
                        probability=torch.stack(probs).mean(0);mask=probability[:,1]>.5
                        dice=float((2*(mask&truth).sum()+1e-6)/(mask.sum()+truth.sum()+1e-6))
                        nll=float(-(b['observations']['y_obs'].cuda()*probability.clamp_min(1e-12).log()).sum(1).mean())
                    trajectory.append(dict(K=k,Dice=dice,NLL=nll,terms={key:sum(v[key] for v in values)/4 for key in values[0]},offset_norms={key:float(v.detach().norm()) for key,v in offsets.items()}))
                    if k==3:break
                    elbo,_,parts,logq=ref.draw(x,theta,a,context,offsets,noises[k])
                    if k==0:
                        for key,value in {**parts,'negative_logq':-logq}.items():
                            if not value.requires_grad:continue
                            gradients=torch.autograd.grad(value.sum(),tuple(offsets.values()),retain_graph=True,allow_unused=True)
                            component_gradients[key]={name:0. if g is None else float(g.norm()) for name,g in zip(offsets,gradients)}
                    gradients=torch.autograd.grad(elbo.sum(),tuple(offsets.values()))
                    offsets={name:(v+s.config.inner_lr*g).detach().requires_grad_(True) for (name,v),g in zip(offsets.items(),gradients)}
                    assert all(torch.isfinite(v).all() for v in offsets.values())
                if i==0:
                    _,expected=ref.refine(x,theta,a,K=3,training=False,noises=noises)
                    assert all(torch.equal(offsets[key],expected[key]) for key in offsets), 'trace differs from original refiner'
                rows.append(dict(arm=arm,batch=i,trajectory=trajectory,initial_component_gradient_norms=component_gradients))
        summary={}
        for arm in ('pure','mixed'):
            subset=[r for r in rows if r['arm']==arm]
            summary[arm]=dict(steps=[dict(K=k,Dice=sum(r['trajectory'][k]['Dice'] for r in subset)/16,NLL=sum(r['trajectory'][k]['NLL'] for r in subset)/16,elbo=sum(r['trajectory'][k]['terms']['elbo'] for r in subset)/16) for k in range(4)],
                dice_decreased=sum(r['trajectory'][3]['Dice']<r['trajectory'][0]['Dice'] for r in subset),
                held_elbo_decreased=sum(r['trajectory'][3]['terms']['elbo']<r['trajectory'][0]['terms']['elbo'] for r in subset))
        write(root/'SUMMARY.json',dict(summary=summary,rows=rows,optimizer_updates=0,validation_access=0,test_access=0,MCMC_calls=0,original_refiner_equivalence=True))
        print(json.dumps(summary),flush=True)

if __name__=='__main__':main()

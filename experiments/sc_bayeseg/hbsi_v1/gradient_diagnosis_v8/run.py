import json, os, time
from pathlib import Path
import torch
from experiments.sc_bayeseg.hbsi_v1.budgets import ResourceLedger, Counters
from experiments.sc_bayeseg.hbsi_v1.stream_state import StreamState
from experiments.sc_bayeseg.hbsi_v1.real_diagnostic import prepare_data, write
from experiments.sc_bayeseg.hbsi_v1.semantic_initialization import SemanticCurriculum
from experiments.sc_bayeseg.hbsi_v1.objective import SequentialObjective
from experiments.sc_bayeseg.hbsi_v1.local_inference import Refiner


def main():
    root=Path(os.environ['RUN_ROOT']);old=Path(os.environ['PARENT_RUN'])
    p=json.loads((old/'PRIVATE_PROTOCOL.json').read_text())
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ledger=ResourceLedger(root/'results',max_seconds=300,max_updates=0,max_bytes=64*1024**2)
    rows=[]
    with ledger.attempt('gradients',seconds=300,updates=0):
        s=StreamState.load(old/'results/mixed_attempt001/ACTIVE_STEP_001200.pt').to_device('cuda')
        data=prepare_data(p,s.config,'mixed');counter=Counters();ref=Refiner(s.model,s.phi,counter)
        semantic=SemanticCurriculum(s,ref,warm_steps=10**9,boundary=-1,balanced_ce=True)
        full=SequentialObjective(s,ref)
        named=[('global_mean',s.q.mean)]+list(s.phi.named_parameters())
        params=[v for _,v in named if v.requires_grad];named=[(k,v) for k,v in named if v.requires_grad]
        for i in range(16):
            ledger.check();batches=[]
            for j,role in enumerate(('L','U')):
                b=data.batch(role,1,torch.Generator().manual_seed(260001+2*i+j));b['x']=b['x'].cuda()
                if 'observations' in b:b['observations']={k:v.cuda() for k,v in b['observations'].items()}
                batches.append(b)
            L,U=batches
            a,_=semantic.estimate(L,U,data.counts,torch.Generator().manual_seed(360001+i))
            ga=torch.autograd.grad(a,params,allow_unused=True)
            b,_=full.estimate(L,U,data.counts,torch.Generator().manual_seed(460001+i))
            b=b/(sum(data.counts.values())*s.config.height*s.config.width)
            gb=torch.autograd.grad(b,params,allow_unused=True)
            groups={'G':([],[]),'recognition_shared':([],[])}
            for (name,param),x,y in zip(named,ga,gb):
                if x is None:continue
                y=torch.zeros_like(x) if y is None else y
                if name=='global_mean':
                    xx=s.q.unpack(x);yy=s.q.unpack(y)
                    for key in xx:
                        if key.startswith('g.'):
                            groups['G'][0].append(xx[key].flatten());groups['G'][1].append(yy[key].flatten())
                elif not name.startswith(('s_head','b_head')):
                    groups['recognition_shared'][0].append(x.flatten());groups['recognition_shared'][1].append(y.flatten())
            row={'batch':i,'semantic_loss':float(a),'normalized_elbo':float(b),'groups':{}}
            for name,(x,y) in groups.items():
                x=torch.cat(x);y=torch.cat(y);dot=x.dot(y);nx=x.norm();ny=y.norm()
                row['groups'][name]=dict(cosine=float(dot/(nx*ny).clamp_min(1e-30)),semantic_norm=float(nx),elbo_norm=float(ny),dot=float(dot))
            rows.append(row)
        summary={k:dict(negative_cosine=sum(r['groups'][k]['cosine']<0 for r in rows),
                       mean_cosine=sum(r['groups'][k]['cosine'] for r in rows)/16,
                       mean_norm_ratio=sum(r['groups'][k]['elbo_norm']/max(r['groups'][k]['semantic_norm'],1e-30) for r in rows)/16)
                 for k in groups}
        write(root/'SUMMARY.json',dict(summary=summary,rows=rows,optimizer_updates=0,validation_access=0,test_access=0,MCMC_calls=0))
        print(json.dumps(summary),flush=True)

if __name__=='__main__':main()

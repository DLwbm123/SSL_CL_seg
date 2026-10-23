"""Single bounded read-only diagnostic on training L/U and frozen states."""
import json, os, time
from pathlib import Path
import torch
from torch.nn import functional as F
from ..stream_state import StreamState, digest
from ..paired_semantic_v19.run import data
from ..paired_semantic_v19.method import balanced_risk
from ..semantic_evidence_v22.method import ImageEvidence, sample_semantics
from ..data_adapter import leaf_observations
from ..budgets import ResourceLedger
from ..real_diagnostic import write


def components(e,x,prob,theta):
    mean=e.gamma.exp()[None,:,None,None]*e.model.structure_mean(prob,theta)+e.beta[None,:,None,None]
    r=(x-mean).transpose(0,1);dr=e.solve(r)
    v=torch.einsum('khw,cbhw->cbk',e.W,dr)
    correction=(v*torch.cholesky_solve(v.transpose(-1,-2),e.chol).transpose(-1,-2)).sum(-1)
    quad=-.5*((r*dr).sum((-2,-1))-correction).sum(0)
    logdet=-.5*e.logdet.sum().expand(len(x))
    return quad,logdet


def read_L(d,i):
    r=d.records['L'][i];x=d.read(r,'image')[None,None].double()
    x=F.interpolate(x,(128,128),mode='bilinear',align_corners=False).cuda()
    y=F.interpolate(d.read(r,'label')[None,None].double(),(128,128),mode='nearest')[:,0].long().cuda()
    return x,leaf_observations(y,2)['y_obs'].double()


def binary(f):return torch.cat([1-f,f],1)
def soft(p):return .98*p+.02/p.shape[1]
def dice(p,t):return ((2*(p[:,1]*t[:,1]).sum()+1e-6)/(p[:,1].sum()+t[:,1].sum()+1e-6))

def main():
    root=Path(os.environ['RUN_ROOT']);p=json.loads((root/'PRIVATE_PROTOCOL.json').read_text())
    assert p['purpose']=='SEMANTIC_ALIGNMENT_D0_V23'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ledger=ResourceLedger(root/'results',max_seconds=900,max_updates=0,max_bytes=500_000_000)
    def status(stage,**kw):
        f=root/'STATUS.tmp';f.write_text(json.dumps(dict(stage=stage,time=time.time(),**kw)));f.replace(root/'STATUS.json')
    paths={'L400':Path(p['v19_root'])/'results/s20261023_L_A_attempt001/ACTIVE_STEP_000400.pt',
           'L1600':Path(p['v22_root'])/'results/L_attempt001/ACTIVE_STEP_001600.pt',
           'LU1600':Path(p['v22_root'])/'results/LU_attempt001/ACTIVE_STEP_001600.pt'}
    rows=[];gradrows=[];calls=0;checks={}
    try:
      with ledger.attempt('once',seconds=900,updates=0):
        for name,path in paths.items():
          status('DIAGNOSE',state=name)
          s=StreamState.load(path);s.to_device('cuda');before=digest(s.payload())
          d=data(p,s.config,'RUNMC','LU');params=list(s.phi.parameters())+list(s.q.parameters())+list(s.current.parameters())
          lengths=[sum(v.numel() for v in s.phi.parameters()),sum(v.numel() for v in s.q.parameters()),sum(v.numel() for v in s.current.parameters())]
          def grads(loss):
            nonlocal calls
            calls+=1;assert calls<=64
            gg=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
            vec=torch.cat([(torch.zeros_like(v) if g is None else g).flatten() for v,g in zip(params,gg)])
            assert torch.isfinite(vec).all()
            return vec.split(lengths)
          for j,i in enumerate([0,10,20,30,40,50,60,70]):
            x,truth=read_L(d,i);rng=torch.Generator().manual_seed(230000+j)
            theta=s.q.rsample(rng);a=s.current.rsample(theta,rng);e=ImageEvidence(s.model,a)
            prob,local=sample_semantics(s,x,theta,a,rng);fg=truth[:,1:2]
            hard=F.one_hot(prob.argmax(1),2).permute(0,3,1,2).double()
            candidates={'hard_prediction':hard,'soft_GT':soft(truth),'half_corrected':.5*(prob.detach()+truth),
                        'eroded_GT':soft(binary(-F.max_pool2d(-fg,7,1,3))),
                        'dilated_GT':soft(binary(F.max_pool2d(fg,7,1,3))),
                        'shifted_GT':soft(torch.roll(truth,8,-1))}
            with torch.no_grad():
              base=e.score(x,prob,theta).item();bd=dice(prob,truth).item()
              for candidate,q in candidates.items():
                rows.append(dict(state=name,candidate=candidate,empty=not bool(fg.any()),different=bool((q-prob).abs().max()>1e-12),
                                 score_delta=e.score(x,q,theta).item()-base,dice_delta=dice(q,truth).item()-bd))
            if j<2:
              task=grads(balanced_risk(prob,truth).mean())[0]
              quad,ld=components(e,x,prob,theta)
              for key,v in [('quadratic',quad),('logdet',ld),('local_KL',local)]:
                # Gradient of loss, so all component signs match optimization.
                gs=grads(-.01*v.mean()/16384)
                cos=None if gs[0].norm()==0 or task.norm()==0 else float(torch.dot(gs[0],task)/(gs[0].norm()*task.norm()))
                gradrows.append(dict(state=name,role='L',component=key,value_per_pixel=float(v.detach().mean()/16384),
                                     norms=[float(g.norm()) for g in gs],task_phi_cosine=cos))
          for j in range(2):
            b=d.batch('U',1,torch.Generator().manual_seed(73001+j));assert 'observations' not in b
            x=b['x'].cuda();rng=torch.Generator().manual_seed(231000+j)
            theta=s.q.rsample(rng);a=s.current.rsample(theta,rng);e=ImageEvidence(s.model,a)
            prob,local=sample_semantics(s,x,theta,a,rng);quad,ld=components(e,x,prob,theta)
            for key,v in [('quadratic',quad),('logdet',ld),('local_KL',local)]:
              gs=grads(-.01*d.counts['U']/d.counts['L']*v.mean()/16384)
              gradrows.append(dict(state=name,role='U',component=key,value_per_pixel=float(v.detach().mean()/16384),norms=[float(g.norm()) for g in gs]))
          checks[name]=digest(s.payload())==before;assert checks[name]
          del s,d
        aggregates=[]
        for name in paths:
          for candidate in candidates:
            rr=[r for r in rows if r['state']==name and r['candidate']==candidate]
            eligible=[r for r in rr if abs(r['dice_delta'])>1e-8]
            aggregates.append(dict(state=name,candidate=candidate,n=len(rr),empty=sum(r['empty'] for r in rr),
                quality_comparisons=len(eligible),concordant=sum(r['score_delta']*r['dice_delta']>0 for r in eligible),
                score_delta_mean=sum(r['score_delta'] for r in rr)/len(rr),dice_delta_mean=sum(r['dice_delta'] for r in rr)/len(rr)))
        write(root/'DIAGNOSTIC_RESULTS.json',dict(status='COMPLETE',candidate_aggregates=aggregates,gradient_probes=gradrows,
              reverse_calls=calls,formal_updates=0,discarded_Adam_steps=0,source_unchanged=checks,L_probes=24,U_probes=6,
              limits=dict(reverse_calls=64,seconds=900,bytes=500000000),gradient_group_order=['phi','q','domain']))
        status('COMPLETE')
    except BaseException as exc:status('FAILED',error_type=type(exc).__name__,error=str(exc));raise

if __name__=='__main__':main()

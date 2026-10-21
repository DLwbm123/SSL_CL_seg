"""Bounded joint optimizer repair on the recorded synthetic endpoint."""
import json, time, signal
from pathlib import Path
import torch
import validate_neural_kappa as base
import validate_neural_kappa_repair as repair

ROOT=Path(__file__).parent

def decompose(w):
    result=base.evaluate(repair.mapped(w,4,torch.float64),'linear')
    for row in result['cases']:
        t,x,k=row['tau'],row['observation'],row['kappa']
        d=1+1/t**2;c=1+k*k+(1-k)**2/t**2
        delta=torch.tensor(row['means'],dtype=torch.float64)-x/(2+t*t)
        mean=.5*(delta.square().sum()+delta.sum().square()/t**2)
        coupling=.5*torch.log(torch.tensor(c*d/(1+2/t**2),dtype=torch.float64))
        v=torch.tensor([c*row['sd'][0]**2,d*row['sd'][1]**2],dtype=torch.float64)
        scale=.5*(v-1-v.log()).sum()
        row.update(mean_KL=float(mean),coupling_KL=float(coupling),scale_KL=float(scale))
        assert abs(float(mean+coupling+scale)-row['KL'])<1e-7
    return result


def main():
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('600s')))
    signal.alarm(600)
    path=ROOT/'results/BAYESEG_G_NEURAL_KAPPA_JOINT_REPAIR.json';assert not path.exists()
    source=json.loads((ROOT/'results/BAYESEG_G_NEURAL_KAPPA_REPAIR.json').read_text())['runs'][-1]
    assert source['gain']==4 and source['compute']=='torch.float64'
    mapped=torch.tensor(source['weights'],dtype=torch.float64)
    coord=mapped.clone();coord[2:4,-1]-=repair.B0;coord[2:4]/=4
    # All arms start from the same rounded coordinates.
    coord=coord.float()
    _,im=base.data(torch.float64)
    def objective(w):return repair.loss(w,im,4,torch.float64).mean()
    out=dict(protocol='NEURAL_KAPPA_JOINT_REPAIR_PREREG_ZH.md',formal_training=False,real_data_reads=0,runs=[],initial=decompose(coord))
    out['restored_weight_error']=float((repair.mapped(coord,4,torch.float64)-mapped).abs().max())
    assert out['restored_weight_error']<2e-6
    w=coord.double().requires_grad_(True)
    h=torch.autograd.functional.hessian(objective,w).reshape(85,85)
    eig=torch.linalg.eigvalsh(h);positive=eig[eig>1e-8]
    out['local_hessian']=dict(eigenvalues=eig.tolist(),positive_condition=float(positive.max()/positive.min()),threshold=1e-8,negative_below_minus_1e8=int((eig < -1e-8).sum()),note='85 coordinates, only 12 observations per head; structural null directions expected')
    def save():path.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    save()
    try:
        for name,dtype in [('Adam',torch.float32),('LBFGS_FP32',torch.float32),('LBFGS_FP64',torch.float64)]:
            w=coord.to(dtype).clone().requires_grad_(True);calls=0;beg=time.monotonic()
            if name=='Adam':
                opt=torch.optim.Adam([w],lr=3e-4,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
            else:
                opt=torch.optim.LBFGS([w],lr=1,max_iter=200,max_eval=400,history_size=50,line_search_fn='strong_wolfe',tolerance_grad=1e-10,tolerance_change=1e-12)
            def closure():
                nonlocal calls
                if calls>=400:raise RuntimeError('objective call budget reached')
                opt.zero_grad();v=objective(w);v.backward();calls+=1
                assert torch.isfinite(v) and torch.isfinite(w.grad).all()
                return v
            failure=None
            try:
                if name=='Adam':
                    for _ in range(200):
                        closure();torch.nn.utils.clip_grad_norm_([w],1.);opt.step()
                else:opt.step(closure)
            except RuntimeError as e:failure=str(e)
            g=torch.autograd.grad(objective(w),w)[0]
            r=dict(name=name,objective_gradient_calls=calls,iterations=200 if name=='Adam' else opt.state[w].get('n_iter',0),seconds=time.monotonic()-beg,failure=failure,block_gradient_max=g.abs().max(1).values.tolist(),**decompose(w.detach()))
            out['runs'].append(r);save()
            print({k:r[k] for k in ['name','objective_gradient_calls','iterations','passed','max_KL','failure']},flush=True)
        out['status']='COMPLETED_SYNTHETIC_ONLY';save()
    except BaseException as e:
        out['status']='FAILED';out['failure']=repr(e);save();raise
    finally:signal.alarm(0)

if __name__=='__main__':main()

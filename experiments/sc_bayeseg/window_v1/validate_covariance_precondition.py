"""Observation-feature preconditioning; no posterior oracle in the transform."""
import json,signal,time
from pathlib import Path
import torch
from validate_covariance_coordinates import Fixture,v
from bounded_lbfgs import make_optimizer,accepted_step
ROOT=Path(__file__).parent

def main():
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('600s')));signal.alarm(600)
    path=ROOT/'results/BAYESEG_G_COVARIANCE_PRECONDITION.json';assert not path.exists()
    src=json.loads((ROOT/'results/BAYESEG_G_BLOCK_MECHANISM.json').read_text());w=torch.tensor(src['mean_diagnostic']['weights'],dtype=torch.float64);_,im=v.base.data(torch.float32)
    f=Fixture(w,im);original=Fixture(w,im);u,s,vt=torch.linalg.svd(f.h,full_matrices=False)
    assert len(s)==12 and float(s.min())>1e-10
    transform=(vt.T/s)@vt;f.h=f.h@transform
    out=dict(protocol='COVARIANCE_PRECONDITION_PREREG_ZH.md',formal_training=False,real_data_reads=0,feature_condition_before=float(s.max()/s.min()),feature_condition_after=float(torch.linalg.cond(f.h)),transform=transform.tolist(),runs=[])
    def save():path.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    save()
    try:
        for kind in ['K','L']:
            p=torch.zeros(3,33,dtype=torch.float32,requires_grad=True);counter={'calls':0};opt=make_optimizer(p);trace=[];beg=time.monotonic();stop='iteration_limit'
            check=f.gradient_check(torch.sin(torch.arange(99)).reshape(3,33)*1e-4,kind,True)
            assert torch.equal(f.values(p,kind)[0],original.values(p,kind)[0])
            for i in range(200):
                try:move=accepted_step(opt,p,lambda q:f.values(q,kind)[0].mean(),counter,400)
                except RuntimeError as e:stop=str(e);break
                if move==0:stop='no displacement';break
                ev=f.evaluate(p,kind);trace.append(dict(iteration=i+1,calls=counter['calls'],max_KL=ev['max_KL'],mean_KL=ev['mean_KL'],passed=ev['passed']))
            mapped=p.detach().double()@transform.T
            mapping_error=float((f.values(p,kind)[0]-original.values(mapped,kind)[0]).abs().max().detach());assert mapping_error<1e-10
            r=dict(kind=kind,calls=counter['calls'],iterations=len(trace),seconds=time.monotonic()-beg,stop=stop,checks=check,endpoint_checks=f.gradient_check(p,kind),trace=trace,endpoint=f.evaluate(p,kind),mapped_FP32=original.evaluate(mapped.float(),kind),mapping_KL_error=mapping_error,weights=p.detach().tolist(),mapped_weights=mapped.tolist())
            out['runs'].append(r);save();print(kind,counter['calls'],len(trace),r['endpoint']['passed'],r['endpoint']['max_KL'],r['mapped_FP32']['passed'],flush=True)
        out['status']='COMPLETED_SYNTHETIC_ONLY';save()
    except BaseException as e:out['failure']=repr(e);out['status']='FAILED';save();raise
    finally:signal.alarm(0)

if __name__=='__main__':main()

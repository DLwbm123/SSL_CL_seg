"""One bounded generated CPU suite. No native model, real data or CUDA calls."""
import copy
import json
import os
import random
import sys
from pathlib import Path
import torch
from .core import parts, weights, hierarchical_kl
from .trainer import HierarchicalTrainer
from ..five_frameworks_v1.kernels import masked_kl
from ..five_frameworks_v1.model import Model
from ..five_frameworks_v1.parent_bridge import SyntheticParentBridge
from ..five_frameworks_v1.recipes import SyntheticCurrentDomain
from ..five_frameworks_v1.train_stage import StageTrainer
from ..five_frameworks_v1 import checkpoint


def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def make(arm=None,initialize=True):
    torch.manual_seed(163);random.seed(163)
    parent=SyntheticParentBridge()
    with torch.no_grad():parent.readout.bias.copy_(torch.tensor([-3.,1.,1.]))
    model=Model(parent,'B2_PARENT_PAS_KL')
    provider=SyntheticCurrentDomain(seed=163,order=1,stage=2,size=8)
    opts=dict(total_steps=5,PAS_confidence=0.,PAS_cosine=-1.)
    return StageTrainer(model,provider,opts,initialize=initialize) if arm is None else HierarchicalTrainer(model,provider,opts,arm,initialize)


def snapshot(t):
    return copy.deepcopy(dict(student=t.model.state_dict(),ema=t.ema.state_dict(),optimizer=t.optimizer.state_dict(),
        scheduler=t.scheduler.state_dict(),scaler=t.scaler.state_dict(),prototypes=t.prototypes.values,
        support=t.prototypes.support,step=t.step,cursor=t.cursor,telemetry=t.telemetry,last=t.last,
        torch_rng=torch.get_rng_state(),python_rng=random.getstate(),reads=[t.provider.l_reads,t.provider.u_reads]))


def run(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    old=list(root.glob('attempt_*'))
    if len(old)>=2:raise RuntimeError('CPU attempts exhausted')
    out=root/f'attempt_{len(old)+1}';out.mkdir()
    counts=dict(optimizer_calls=0,autograd_calls=0,expected_failures=0)
    checks=[]
    def record():
        (out/'COUNTS.json').write_text(json.dumps(counts,indent=2))
    step=torch.optim.Adam.step;grad=torch.autograd.grad
    def tracked_step(*a,**kw):
        if counts['optimizer_calls']>=16:raise RuntimeError('CPU attempt cap')
        counts['optimizer_calls']+=1;record();return step(*a,**kw)
    def tracked_grad(*a,**kw):
        if counts['autograd_calls']>=128:raise RuntimeError('CPU gradient-call cap')
        counts['autograd_calls']+=1;record();return grad(*a,**kw)
    torch.optim.Adam.step=tracked_step;torch.autograd.grad=tracked_grad
    try:
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        for dtype,tol in [(torch.float32,2e-6),(torch.float64,1e-12)]:
            torch.manual_seed(7)
            z=torch.randn(2,3,4,4,dtype=dtype,requires_grad=True)
            q=torch.randn_like(z).softmax(1).requires_grad_(True);m=torch.ones(2,4,4,dtype=torch.bool)
            p,c=parts(z,q,m);a=masked_kl(z,q,m);b=(p+c).mean()
            torch.testing.assert_close(a,b,atol=tol,rtol=tol)
            ga,=torch.autograd.grad(a,z,retain_graph=True);gb,=torch.autograd.grad(b,z)
            torch.testing.assert_close(ga,gb,atol=tol,rtol=tol)
            equal(hierarchical_kl(z,q,m,'C0'),masked_kl(z,q,m))
        checks.append('KL identity value and student gradients float32/float64; C0 exact path')
        q=torch.tensor([[[[0.,1.,.1,.1]],[[.5,0.,.9,0.]],[[.5,0.,0.,.9]]]],requires_grad=True)
        z=torch.tensor([[[[1000.,-1000.,0.,0.]],[[0.,0.,1000.,-1000.]], [[-1000.,1000.,-1000.,1000.]]]],requires_grad=True)
        m=torch.ones(1,1,4,dtype=torch.bool)
        for arm in ('C1','C2'):
            loss=hierarchical_kl(z,q,m,arm);gq,gz=torch.autograd.grad(loss,(q,z),allow_unused=True)
            assert gq is None and torch.isfinite(gz).all() and torch.isfinite(loss)
            w=weights(q,m,arm);assert w.min()>=.5 and w.max()<=1 and not w.requires_grad
        assert weights(q,m,'C2')[0,0,0]==.5
        assert weights(q,m,'C2')[0,0,1]==1
        empty=m&False;loss=hierarchical_kl(z,q,empty,'C2');g,=torch.autograd.grad(loss,z)
        assert loss==0 and torch.count_nonzero(g)==0
        checks.append('zero mass, pure classes, extreme logits, empty support, teacher/weights stop-gradient')
        torch.manual_seed(12);q=torch.randn(2,3,5,5).softmax(1);q[:,0]*=.01;q/=q.sum(1,keepdim=True)
        m=torch.rand(2,5,5)>.2;w1=weights(q,m,'C1');w2=weights(q,m,'C2');dq=q[:,1:].sum(1)
        e=m&(dq>=.9);cls=q.argmax(1)
        for i in range(2):
            for k in (1,2):
                g=e[i]&(cls[i]==k)
                torch.testing.assert_close((dq[i][g]*w1[i][g]).sum(),(dq[i][g]*w2[i][g]).sum())
                if g.any():assert w1[i][g].unique().numel()==1
        assert (w1[~e]==1).all() and (w2[~e]==1).all()
        checks.append('C1 per-image/per-predicted-class teacher mass matching and scope')
        for bad in ['C3','unknown']:
            try:weights(q,m,bad)
            except ValueError:pass
            else:raise AssertionError('unregistered arm admitted')
        t=make(None)
        for _ in range(2):t.update()
        expected=snapshot(t);t=make('C0')
        for _ in range(2):t.update()
        equal(expected,snapshot(t));checks.append('C0 shared-engine two-update bitwise no-op including EMA/Adam/prototypes/RNG')
        for arm in ('C1','C2'):
            t=make(arm);t.update();ident=dict(family=t.model.family,seed=163,order=1,stage=2,arm=arm)
            path=out/(arm+'.pt');checkpoint.save(t,path,ident)
            for _ in range(2):t.update()
            expected=snapshot(t);r=make(arm,initialize=False);checkpoint.restore(r,path,ident)
            for _ in range(2):r.update()
            equal(expected,snapshot(r))
            wrong=make('C2' if arm=='C1' else 'C1',initialize=False)
            try:checkpoint.restore(wrong,path,ident)
            except ValueError:pass
            else:raise AssertionError('cross-arm restore admitted')
        checks.append('C1/C2 checkpoint continuation: model/EMA/Adam/scheduler/scaler/prototypes/reads/RNG exact; cross-arm rejected')
        t=make('C2')
        try:t.update(fault='after_optimizer')
        except RuntimeError as e:assert 'injected after_optimizer' in str(e)
        else:raise AssertionError('fault absent')
        counts['expected_failures']+=1;record()
        assert t.requires_restore and t.step==0 and t.physical_optimizer_updates==1
        try:t.update()
        except RuntimeError:pass
        else:raise AssertionError('failed state reused')
        try:checkpoint.save(t,out/'invalid.pt',{})
        except RuntimeError:pass
        else:raise AssertionError('uncommitted checkpoint allowed')
        checks.append('one charged after-optimizer injection; no automatic retry or uncommitted checkpoint')
        assert counts['optimizer_calls']==15
        result=dict(status='PASS',checks=checks,counts=counts,torch=torch.__version__,scope='generated CPU fixture only',native_runs=0,patient_reads=0,CUDA_calls=0)
    except BaseException as e:
        result=dict(status='FAIL',checks=checks,counts=counts,error=repr(e));raise
    finally:
        torch.optim.Adam.step=step;torch.autograd.grad=grad
        (out/'REPORT.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':run(os.environ['TASK_EVIDENCE'])

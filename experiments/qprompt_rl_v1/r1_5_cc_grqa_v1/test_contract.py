"""Zero optimizer-call formula, solver and gradient regression."""
import itertools
import torch
from qprompt.losses import PrototypeBank, grqa_loss
from .method import exact_assignment,routed_grqa,weight


def main():
    # Execute the actual diagnostic merge expressions, including overlapping fields.
    import ast
    from pathlib import Path
    tree=ast.parse(Path(__file__).with_name('runner.py').read_text())
    merge=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='diag' for t in n.targets) and isinstance(n.value,ast.Dict) and n.value.keys==[None,None,None])
    vals={'categorical_kl':.2,'Lseg':.4};routing={'categorical_kl':.2};grads={'norm_seg':1.}
    diag=eval(compile(ast.Expression(merge),'merge','eval'),{},dict(vals=vals,routing=routing,grads=grads))
    assert diag=={'categorical_kl':.2,'Lseg':.4,'norm_seg':1.} and vals['Lseg']==.4
    call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='dict' and n.args and isinstance(n.args[0],ast.Name) and n.args[0].id=='diag')
    call.keywords=[ast.keyword(arg='Lseg',value=ast.Constant(.5))];ast.fix_missing_locations(call)
    assert eval(compile(ast.Expression(call),'record','eval'),{},dict(diag=diag))['Lseg']==.5
    torch.manual_seed(261)
    for classes in (1,2,3):
        supported=torch.arange(3)<classes
        for trial in range(4):
            s=torch.randint(-2,3,(1,6,3)).double().requires_grad_()
            lo,hi={1:(6,6),2:(2,4),3:(1,3)}[classes]
            feasible=[p for p in itertools.product(range(classes),repeat=6) if all(lo<=p.count(c)<=hi for c in range(classes))]
            expected=min(feasible,key=lambda p:(-sum(float(s[0,i,c]) for i,c in enumerate(p)),p))
            actual=exact_assignment(s,supported,(lo,hi))
            assert tuple(actual[0].tolist())==expected and not actual.requires_grad
        s=torch.zeros(2,12,3);a=exact_assignment(s,supported)
        assert torch.equal(a,exact_assignment(s,supported))
        for row in a:
            counts=torch.bincount(row,minlength=3);lo,hi={1:(12,12),2:(4,8),3:(3,6)}[classes]
            assert all(lo<=int(counts[c])<=hi for c in range(classes))
            assert not counts[classes:].any()
    bank=PrototypeBank(8);bank.vectors.copy_(torch.nn.functional.normalize(torch.randn(3,8),dim=-1));bank.supported.fill_(True)
    q=torch.randn(2,12,8,requires_grad=True);r=torch.randn_like(q,requires_grad=True)
    a=grqa_loss(q,r,bank);b,_=routed_grqa(q,r,bank,False,True)
    for k in ('total','group','paper_selected_k3_surrogate','categorical_kl'):torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    ga=torch.autograd.grad(a['total'],q,retain_graph=True)[0];gb=torch.autograd.grad(b['total'],q)[0]
    torch.testing.assert_close(ga,gb,rtol=0,atol=0)
    cc,info=routed_grqa(q,r,bank,True,True);cc['total'].backward();assert r.grad is None and torch.isfinite(q.grad).all()
    bank.supported.zero_();zero,_=routed_grqa(q,r,bank,True);assert zero['total'].requires_grad and zero['total']==0
    assert weight('A4',1)==5/300 and weight('A2',150)==2.5 and weight('A4',300)==5 and weight('A4',1000)==5
    assert weight('A0',1000)==0 and weight('A3',1)==5
    print('PASS exact solver vs exhaustive small-K; deterministic ties; support cases; original objective/gradient equality; detach; zero-support; ramp. Optimizer calls=0.',flush=True)

if __name__=='__main__':main()

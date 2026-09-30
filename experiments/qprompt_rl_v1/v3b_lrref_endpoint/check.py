"""Small runnable check for changed Adam semantics, including None vs zero."""
import copy
import torch
from .effect import preview

def main():
    torch.manual_seed(7)
    ps=[torch.nn.Parameter(torch.randn(4,3)) for _ in range(3)]
    opt=torch.optim.Adam([{'params':ps[:2],'lr':.0005},{'params':ps[2:],'lr':.0007}],weight_decay=4e-5)
    for round in range(2):
        gs=[torch.randn_like(ps[0]),None,torch.zeros_like(ps[2])]
        before=copy.deepcopy(opt.state_dict());params=[p.detach().clone() for p in ps]
        values,states=preview(opt,ps,gs)
        assert len(opt.state)==len(before['state'])
        for p,v in zip(ps,params):assert torch.equal(p,v)
        for p,g in zip(ps,gs):p.grad=g
        opt.step()
        for i,(p,v) in enumerate(zip(ps,values)):
            assert torch.equal(p,v)
            if i==1:assert p not in opt.state
            else:
                for k,x in states[i].items():assert torch.equal(x,opt.state[p][k])
    print('PASS native Adam preview: empty/nonempty state, coupled decay, group LR, None versus zero')
if __name__=='__main__':main()

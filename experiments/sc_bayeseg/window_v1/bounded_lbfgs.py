"""One accepted step at a time; restore parameters on interrupted line search."""
import torch

class BudgetExhausted(RuntimeError):
    pass


def make_optimizer(param):
    return torch.optim.LBFGS([param],lr=1,max_iter=1,max_eval=400,history_size=50,line_search_fn='strong_wolfe',tolerance_grad=1e-10,tolerance_change=1e-12)


def accepted_step(opt,param,objective,counter,limit):
    saved=param.detach().clone()
    def closure():
        if counter['calls']>=limit:raise BudgetExhausted('objective gradient budget')
        opt.zero_grad();counter['calls']+=1
        loss=objective(param);loss.backward()
        if not torch.isfinite(loss) or not torch.isfinite(param.grad).all():raise RuntimeError('nonfinite objective/gradient')
        return loss
    try:
        opt.step(closure)
        if not torch.isfinite(param).all():raise RuntimeError('nonfinite accepted parameter')
    except BaseException:
        with torch.no_grad():param.copy_(saved)
        # Caller stops/discards this optimizer: interrupted history is not reusable.
        raise
    return float((param.detach()-saved).norm())


def self_check():
    p=torch.tensor([2.],dtype=torch.float64,requires_grad=True);o=make_optimizer(p);c={'calls':0}
    try:accepted_step(o,p,lambda q:(q-7).square().sum(),c,1)
    except BudgetExhausted:pass
    else:raise AssertionError('budget did not interrupt')
    assert torch.equal(p.detach(),torch.tensor([2.],dtype=torch.float64))
    o=make_optimizer(p);c={'calls':0};accepted_step(o,p,lambda q:(q-7).square().sum(),c,20)
    saved=p.detach().clone()
    try:accepted_step(o,p,lambda q:(q-7).square().sum(),c,c['calls']+1)
    except BudgetExhausted:pass
    else:raise AssertionError('second budget did not interrupt')
    assert torch.equal(p.detach(),saved)
    count=[0];o=make_optimizer(p)
    def fail(q):
        count[0]+=1
        if count[0]>1:raise RuntimeError('injected line search failure')
        return (q-9).square().sum()
    try:accepted_step(o,p,fail,{'calls':0},20)
    except RuntimeError:pass
    assert torch.equal(p.detach(),saved)
    return dict(first_step_budget_restore=True,later_step_budget_restore=True,exception_restore=True)
